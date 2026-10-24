#include "rk_npu_attention_f16_internal.h"
#include "rk_npu_trace.h"
#include <condition_variable>
#include <exception>
#include <mutex>
#include <system_error>
#include <thread>

namespace rknpu2_matmul_open::attention {
bool pipeline_policy()noexcept{
    const char* value=std::getenv("RK_NPU_ATTN_PIPELINE");
    return !value || std::strcmp(value,"0");
}

class Pipeline {
    struct Block {int first,rows;};
    struct Slot {
        Workspace* work=nullptr;
        Tail tail;
        int first=0,rows=0,start=0;
        rk_npu_attention_f16_timings timings{};
    };
    Workspace& owner;
    Workspace alternate;
    Slot slots[2];
    std::array<Block,128> blocks{};
    int block_count=0;
    std::mutex mutex;
    std::condition_variable ready;
    Slot* job=nullptr;
    bool job_qk=false,pending=false,done=true,stop=false;
    std::exception_ptr error;
    std::thread thread;

    void worker()noexcept{
        std::unique_lock<std::mutex> lock(mutex);
        for(;;){
            ready.wait(lock,[&]{return pending || stop;});
            if(stop)return;
            Slot& slot=*job;const bool qk=job_qk;pending=false;lock.unlock();
            std::exception_ptr failure;
            try{
                RK_NPU_TRACE_EVENT("rknpu.detail", "attention.pipeline_submit",
                    "qk", qk, "query_offset", slot.first, "query_rows", slot.rows);
                const auto begin=Clock::now();auto& graph=*slot.work->graph;
                slot.timings.hardware_us+=qk?graph.submit():graph.submit_pv(slot.start);
                slot.timings.submit_us+=elapsed(begin,Clock::now());
            }catch(...){failure=std::current_exception();}
            lock.lock();error=failure;done=true;ready.notify_all();
        }
    }
    void launch(Slot& slot,bool qk){
        std::lock_guard<std::mutex> lock(mutex);
        job=&slot;job_qk=qk;done=false;pending=true;ready.notify_one();
    }
    void wait(){
        std::unique_lock<std::mutex> lock(mutex);
        ready.wait(lock,[&]{return done;});
        auto failure=error;error=nullptr;lock.unlock();
        if(failure)std::rethrow_exception(failure);
    }
    void plan(Cache& cache,int rows,int start,int block_rows){
        block_count=0;
        for(int first=0;first<rows;){
            const int count=owner.causal_rows(cache,std::min(block_rows,rows-first),start+first);
            blocks[block_count++]={first,count};first+=count;
        }
    }
    void prepare_slot(Cache& cache,int index,const uint16_t* query,int rows,int start){
        auto& slot=slots[index%2];const auto block=blocks[index];
        slot.first=block.first;slot.rows=block.rows;slot.start=start+block.first;slot.timings={};
        const int end=owner.cfg.mask_mode==RK_NPU_ATTENTION_F16_CAUSAL?start+rows:cache.length;
        slot.work->geometry(cache,slot.rows,slot.start,slot.timings,end,&slot.tail);
        if(query){
            const auto begin=Clock::now();
            slot.work->graph->pack_q(query,rows,slot.first);
            slot.timings.query_pack_us=elapsed(begin,Clock::now());
        }
    }
    void from_device(Slot& slot){
        const auto begin=Clock::now();
        slot.work->graph->sync_scores(RK_NPU_SYNC_FROM_DEVICE);
        slot.timings.submit_us+=elapsed(begin,Clock::now());
    }
    void scores(Slot& slot,Cache& cache,const rk_npu_attention_f16_boolean_mask* mask){
        RK_NPU_TRACE_EVENT("rknpu.detail", "attention.pipeline_scores", "query_offset", slot.first);
        auto view=mask?*mask:rk_npu_attention_f16_boolean_mask{};
        if(mask && view.query_rows!=1)view.query_offset+=slot.first;
        const auto begin=Clock::now();
        slot.work->graph->process_scores(slot.start,mask?&view:nullptr,cache.length);
        slot.timings.submit_us+=elapsed(begin,Clock::now());
    }
    void finish(Slot& slot,void* output,int rows,bool half,rk_npu_attention_f16_timings& total){
        const auto begin=Clock::now();slot.work->graph->sync_output();const auto synced=Clock::now();
        slot.work->graph->finish(output,rows,slot.first,half);
        slot.timings.output_sync_us=elapsed(begin,synced);
        slot.timings.finish_us=elapsed(synced,Clock::now());add_timings(total,slot.timings);
    }
public:
    explicit Pipeline(Workspace& w):owner(w),alternate(w.lease.domain,w.cfg,w.shape){
        slots[0].work=&w;slots[1].work=&alternate;
        alternate.full_causal_prefill=w.full_causal_prefill;
        // One persistent blocking-submit worker. There is never more than one
        // NPU submit in flight, and its buffers outlive every completed ioctl.
        try{thread=std::thread(&Pipeline::worker,this);}
        catch(const std::system_error&){throw Error{RK_NPU_ERR_NOMEM};}
    }
    ~Pipeline(){
        std::unique_lock<std::mutex> lock(mutex);
        ready.wait(lock,[&]{return done;});stop=true;ready.notify_one();lock.unlock();thread.join();
    }
    void prepare(Cache& cache,int rows,int start,int block_rows){
        plan(cache,rows,start,block_rows);
        try{for(int index=0;index<block_count;++index)prepare_slot(cache,index,nullptr,rows,start);}
        catch(...){owner.poisoned=true;throw;}
    }
    rk_npu_attention_f16_timings run(Cache& cache,const uint16_t* query,int rows,int start,
        void* output,const rk_npu_attention_f16_boolean_mask* mask,bool half,int block_rows){
        RK_NPU_TRACE_EVENT("rknpu", "attention.pipeline_run", "query_rows", rows,
            "query_start", start, "kv_length", cache.length);
        const auto begin=Clock::now();plan(cache,rows,start,block_rows);
        rk_npu_attention_f16_timings total{};total.prepare_us=elapsed(begin,Clock::now());
        auto& a=slots[0];auto& b=slots[1];
        try{
            prepare_slot(cache,0,query,rows,start);launch(a,true);
            if(block_count>1)prepare_slot(cache,1,query,rows,start);
            wait();
            for(int index=0;index<block_count;index+=2){
                if(index+1==block_count){
                    from_device(a);scores(a,cache,mask);launch(a,false);wait();finish(a,output,rows,half,total);break;
                }
                // Invalidate first: overlap the next QK with real centering,
                // rather than consuming it during cache maintenance alone.
                from_device(a);launch(b,true);scores(a,cache,mask);wait();
                launch(a,false);from_device(b);scores(b,cache,mask);wait();
                // The next LUT/PV hides finish; the next pair's QK also hides
                // finish and packing, before either slot can be reused by PV.
                launch(b,false);finish(a,output,rows,half,total);
                if(index+2<block_count)prepare_slot(cache,index+2,query,rows,start);
                wait();
                if(index+2<block_count){
                    launch(a,true);finish(b,output,rows,half,total);
                    if(index+3<block_count)prepare_slot(cache,index+3,query,rows,start);
                    wait();
                }else finish(b,output,rows,half,total);
            }
        }catch(...){
            // A CPU sync/allocation can fail while the other slot is on the
            // device. Drain it before releasing the cache lock or any buffer.
            auto failure=std::current_exception();try{wait();}catch(...){}
            owner.poisoned=true;std::rethrow_exception(failure);
        }
        total.total_us=elapsed(begin,Clock::now());return total;
    }
};

Workspace::Workspace(rk_npu_iommu_domain* d,const rk_npu_attention_f16_config& c,Shape s)
    :lease(d),cfg(c),shape(s),pipeline_enabled(pipeline_policy()){}
Workspace::~Workspace()=default;
void Workspace::prepare_pipeline(Cache& cache,int rows,int start,int block_rows){
    if(!pipeline)pipeline=std::make_unique<Pipeline>(*this);
    pipeline->prepare(cache,rows,start,block_rows);
}
rk_npu_attention_f16_timings Workspace::run_pipeline(Cache& cache,const uint16_t* query,int rows,int start,
    void* output,const rk_npu_attention_f16_boolean_mask* mask,bool half,int block_rows){
    const auto begin=Clock::now();
    if(!pipeline)pipeline=std::make_unique<Pipeline>(*this);
    const auto created=Clock::now();auto result=pipeline->run(cache,query,rows,start,output,mask,half,block_rows);
    result.prepare_us+=elapsed(begin,created);return result;
}
} // namespace rknpu2_matmul_open::attention
