#include "mla_f16_test_utils.h"
#include "rk_npu_matmul_f16.h"
#include "rk_npu_quant_matmul.h"
#include "../src/rk_npu_internal.h"
using namespace mla_test;
struct I8Canary {
    struct PerCore {
        rk_npu_matmul_workspace* workspace=nullptr;
        rk_npu_i8i8i32_weights* weights=nullptr;
        ~PerCore(){rk_npu_matmul_workspace_free(workspace);rk_npu_i8i8i32_weights_free(weights);}
    };
    std::vector<std::unique_ptr<PerCore>> cores;
    std::vector<int8_t> a=std::vector<int8_t>(4*128,1);
    std::vector<int32_t> output=std::vector<int32_t>(4*32);
    I8Canary(Device& d,int mask){for(int core:{1,2,4})if(mask&core){
        auto p=std::make_unique<PerCore>();rk_npu_matmul_strategy s{};
        s.op_kind=RK_NPU_MATMUL_I8I8I32;s.M=4;s.N=32;s.K=128;s.k_tile=128;s.n_tile=32;
        s.a_layout=RK_NPU_MATMUL_A_LAYOUT_NATIVE_K16_M16;s.wave_count=1;s.n_groups=1;
        s.npu_core_mask=core;s.cpu_core_mask=1ull<<4;s.cpu_threads=1;
        p->workspace=rk_npu_matmul_workspace_create(d.domain,&s);
        rk_npu_matmul_weight_config cfg{128,32,128};std::vector<int8_t> b(128*32,1);
        p->weights=rk_npu_i8i8i32_weights_create(d.domain,&cfg,b.data());
        require(p->workspace&&p->weights,"INT8 canary prepare");cores.push_back(std::move(p));
    }}
    void run(){for(auto& p:cores){check(rk_npu_i8i8i32_run(p->workspace,p->weights,a.data(),output.data()));
        require(std::all_of(output.begin(),output.end(),[](int32_t x){return x==128;}),"MLA/INT8 mixed-state output");}}
};
void emitted_tasks(Device& d){
    for(const auto& shape:std::vector<std::vector<int>>{{1,128,4096,192},{8,128,129,512},{2,64,513,4096}}){
        const int batch=shape[0];rk_npu_matmul_f16_config cfg{};
        rk_npu_matmul_f16_config_init(&cfg,shape[1],shape[2],shape[3],RK_NPU_FUSE_NONE);
        cfg.a_layout=RK_NPU_F16_A_LAYOUT_NATIVE_K8_M8;cfg.d_layout=RK_NPU_F16_D_LAYOUT_NATIVE_N8_M8;
        rk_npu_matmul_sizes s{};check(rk_npu_matmul_f16_batch_query(batch,&cfg,&s));
        rk_npu_mem a{},b{},out{};check(rk_npu_mem_alloc(d.domain,s.input_bytes,RK_NPU_MEM_DATA_DEFAULT,&a));
        check(rk_npu_mem_alloc(d.domain,s.weight_bytes,RK_NPU_MEM_DATA_DEFAULT,&b));
        check(rk_npu_mem_alloc(d.domain,s.output_bytes,RK_NPU_MEM_DATA_DEFAULT,&out));
        auto* plan=rk_npu_matmul_f16_batch_prepare(d.domain,batch,&cfg);require(plan,"task equivalence prepare");
        std::vector<rknpu2_matmul_open::detail::RegisterTask> before,after;
        check(rknpu2_matmul_open::detail::export_f16_batch_tasks(plan,&a,&b,&out,before));
        check(rknpu2_matmul_open::detail::emit_f16_batch_tasks(d.domain,batch,&cfg,&a,&b,&out,after));
        require(before.size()==after.size(),"emit/export task count");
        for(size_t i=0;i<before.size();++i)require(before[i].body==after[i].body && before[i].op_idx==after[i].op_idx && before[i].enable_mask==after[i].enable_mask,"emit/export register bodies");
        rk_npu_matmul_f16_batch_plan_free(plan);rk_npu_mem_free(d.ctx,&out);rk_npu_mem_free(d.ctx,&b);rk_npu_mem_free(d.ctx,&a);
    }
    std::puts("CHECK direct task emission vs allocated plans bitexact PASS");
}
void cpu(){
    rk_npu_mla_f16_config c;rk_npu_mla_f16_config_init(&c);rk_npu_mla_f16_sizes s{};
    require(rk_npu_mla_f16_query(&c,4096,1,4096,&s)==RK_NPU_ERR_PARAM,"approximation opt-in");c.flags=1;
    check(rk_npu_mla_f16_query(&c,4096,1,4096,&s));require(s.physical_rows==4 && s.key_bytes==16ull*4096*192*2 && s.output_bytes==16*128*4,"expanded shape");
    c.mode=RK_NPU_MLA_F16_ABSORBED;check(rk_npu_mla_f16_query(&c,8192,5,4097,&s));
    require(s.compute_length==4128 && s.physical_rows==20 && s.key_bytes==8192ull*576*2 && s.value_bytes==8192ull*544*2,"shared latent shape");
    for(int mask:{0,1,2,3,4,7}){c.core_mask=mask;check(rk_npu_mla_f16_query(&c,8192,32,4096,&s));}
    for(int mask:{1,3,7}){c.core_mask=mask;check(rk_npu_mla_f16_query(&c,8192,1,4096,&s));
        require(s.physical_rows==(mask==1?16:mask==3?8:6),"decode head grouping");require(s.output_bytes==16*128*4,"logical output heads");}
    for(int mask:{5,6,8}){c.core_mask=mask;require(rk_npu_mla_f16_query(&c,8192,1,4096,&s)==RK_NPU_ERR_PARAM,"invalid core");}
    c.core_mask=1;c.max_query_rows=33;require(rk_npu_mla_f16_query(&c,8192,1,4096,&s)==RK_NPU_ERR_PARAM,"row limit");
    c.mode=RK_NPU_MLA_F16_EXPANDED;c.max_query_rows=128;
    for(int rows:{64,127,128})check(rk_npu_mla_f16_query(&c,8192,rows,4096,&s));
    c.max_query_rows=129;require(rk_npu_mla_f16_query(&c,8192,1,4096,&s)==RK_NPU_ERR_PARAM,"expanded row limit");
    rk_npu_mla_f16_workspace* p=reinterpret_cast<rk_npu_mla_f16_workspace*>(1);
    require(rk_npu_mla_f16_prepare(nullptr,&c,nullptr,&p)==RK_NPU_ERR_PARAM && !p,"failure handle");
    std::puts("CHECK MLA CPU contract PASS");
}
// Ordinary GEMM on every selected core catches inherited DPU/PC state errors.
void canary(Device& d,int mask){
    for(int core:{1,2,4})if(mask&core){
        rk_npu_matmul_f16_config cfg{};rk_npu_matmul_f16_config_init(&cfg,4,32,128,RK_NPU_FUSE_NONE);cfg.core_mask=core;
        rk_npu_matmul_sizes s{};check(rk_npu_matmul_f16_query(&cfg,&s));
        rk_npu_mem a{},b{},out{};check(rk_npu_mem_alloc(d.domain,s.input_bytes,RK_NPU_MEM_DATA_DEFAULT,&a));
        check(rk_npu_mem_alloc(d.domain,s.weight_bytes,RK_NPU_MEM_DATA_DEFAULT,&b));check(rk_npu_mem_alloc(d.domain,s.output_bytes,RK_NPU_MEM_DATA_DEFAULT,&out));
        std::vector<uint16_t> av(4*128,0x3000),bv(128*32,0x3400),ov(4*32);
        check(rk_npu_matmul_f16_pack_a(&cfg,av.data(),&a));check(rk_npu_matmul_f16_pack_b(&cfg,bv.data(),&b));
        check(rk_npu_mem_sync(d.ctx,&a,RK_NPU_SYNC_TO_DEVICE));check(rk_npu_mem_sync(d.ctx,&b,RK_NPU_SYNC_TO_DEVICE));
        auto* plan=rk_npu_matmul_f16_prepare(d.domain,&cfg);require(plan,"canary prepare");
        check(rk_npu_matmul_f16_run(d.ctx,plan,&a,&b,nullptr,&out));check(rk_npu_mem_sync(d.ctx,&out,RK_NPU_SYNC_FROM_DEVICE));
        check(rk_npu_matmul_f16_unpack_d(&cfg,&out,ov.data()));require(std::all_of(ov.begin(),ov.end(),[](uint16_t v){return v==0x4400;}),"GEMM state canary");
        rk_npu_matmul_f16_plan_free(plan);rk_npu_mem_free(d.ctx,&out);rk_npu_mem_free(d.ctx,&b);rk_npu_mem_free(d.ctx,&a);
    }
}
void board(int core,bool small,uint32_t flags=1){
    BoardLock l1("/tmp/rknpu_lowlevel_submit.lock"),l2("/tmp/rk3588_npu_submit.lock");Device dev;Fixture f(small?161:4160,128);auto w=weights(dev,f.w.data());
    emitted_tasks(dev);
    rk_npu_mla_f16_config cfg;rk_npu_mla_f16_config_init(&cfg);cfg.flags=flags;cfg.core_mask=core;cfg.max_capacity=8192;
    for(auto mode:{RK_NPU_MLA_F16_EXPANDED,RK_NPU_MLA_F16_ABSORBED}){
        cfg.mode=mode;cfg.max_query_rows=mode==RK_NPU_MLA_F16_EXPANDED?128:32;
        auto c=cache(dev,cfg,w.get());auto ws=work(dev,cfg,w.get());auto sc=cfg;sc.core_mask=1;auto single=work(dev,sc,w.get());
        for(int length:small?std::vector<int>{33,129}:std::vector<int>{33,129,4096,4097}){
            check(rk_npu_mla_f16_cache_load(dev.ctx,c.get(),f.latent.data(),f.rope.data(),length));
            std::vector<int> query_rows{1,5,17,32};
            if(mode==RK_NPU_MLA_F16_EXPANDED && length>=128)query_rows.insert(query_rows.end(),{64,127,128});
            for(int rows:query_rows){
                GuardOutput out(size_t(rows)*H*D);canary(dev,core);rk_npu_mla_f16_timings t{};
                check(rk_npu_mla_f16_run(dev.ctx,ws.get(),c.get(),f.q.data(),f.gates.data(),rows,length-rows,out.ptr(),&t,nullptr));
                auto saved=out.values();double e=error(saved,f.reference(length,rows,length-rows,cfg.mask_mode));require(e<.08,"MLA exact-softmax reference");
                canary(dev,core);check(rk_npu_mla_f16_run(dev.ctx,single.get(),c.get(),f.q.data(),f.gates.data(),rows,length-rows,out.ptr(),nullptr,nullptr));
                require(out.values()==saved,"single/multicore output");
                check(rk_npu_mla_f16_cache_reserve(dev.ctx,c.get(),8192));
                check(rk_npu_mla_f16_run(dev.ctx,ws.get(),c.get(),f.q.data(),f.gates.data(),rows,length-rows,out.ptr(),nullptr,nullptr));require(out.values()==saved,"reserve/rebind");
                std::printf("CHECK core=%d mode=%d L=%d Q=%d rel_l2=%.8f tasks=%d PASS\n",core,int(mode),length,rows,e,t.attention.tasks);
            }
        }
        // Mode switches, ungated output, tail padding, token-broadcast and
        // arbitrary head/query/key byte strides with an entirely hidden row.
        check(rk_npu_mla_f16_cache_load(dev.ctx,c.get(),f.latent.data(),f.rope.data(),129));
        for(int start:{0,5,10,15,50,3,124}){GuardOutput out(5*H*D);rk_npu_mla_f16_timings t{};
            check(rk_npu_mla_f16_run(dev.ctx,ws.get(),c.get(),f.q.data(),f.gates.data(),5,start,out.ptr(),&t,nullptr));
            require(t.attention.compute_length==160,"fixed valid-KV prefill span");
            require(error(out.values(),f.reference(129,5,start,cfg.mask_mode))<.08,"incremental/backward causal mask");
        }
        Fixture other(129,32);for(auto& x:other.latent)x^=0x8000;for(auto& x:other.cf)x=-x;
        auto layer_b=cache(dev,cfg,w.get());check(rk_npu_mla_f16_cache_load(dev.ctx,layer_b.get(),other.latent.data(),other.rope.data(),129));
        for(auto* chosen:{c.get(),layer_b.get(),c.get()}){GuardOutput out(5*H*D);
            check(rk_npu_mla_f16_run(dev.ctx,ws.get(),chosen,f.q.data(),f.gates.data(),5,17,out.ptr(),nullptr,nullptr));
            require(error(out.values(),(chosen==c.get()?f:other).reference(129,5,17,cfg.mask_mode))<.08,"A/B/A cache relocation");
        }
        for(auto mask_mode:{RK_NPU_ATTENTION_F16_NO_MASK,RK_NPU_ATTENTION_F16_BOOLEAN_MASK}){
            auto mc=cfg;mc.mask_mode=mask_mode;auto mw=work(dev,mc,w.get());GuardOutput out(5*H*D);
            std::vector<uint8_t> bytes(size_t(H)*7*129*2,0x55);
            for(int h=0;h<H;++h)for(int r=0;r<7;++r)for(int k=0;k<129;++k)bytes[((size_t(h)*7+r)*129+k)*2]=uint8_t(r!=1 && (k+r+h)%3!=0);
            rk_npu_attention_f16_boolean_mask mask;rk_npu_attention_f16_boolean_mask_init(&mask,bytes.data(),bytes.size(),H,7,129);
            mask.head_stride*=2;mask.query_stride*=2;mask.key_stride=2;mask.query_offset=1;mask.version=1;
            auto* mp=mask_mode==RK_NPU_ATTENTION_F16_BOOLEAN_MASK?&mask:nullptr;
            check(rk_npu_mla_f16_run(dev.ctx,mw.get(),c.get(),f.q.data(),nullptr,5,17,out.ptr(),nullptr,mp));
            require(error(out.values(),f.reference(129,5,17,mask_mode,mp,false))<.08,"ungated/mask oracle");
            {GuardOutput one(H*D);if(mp)mask.query_offset=2;
                check(rk_npu_mla_f16_run(dev.ctx,mw.get(),c.get(),f.q.data(),nullptr,1,128,one.ptr(),nullptr,mp));
                require(error(one.values(),f.reference(129,1,128,mask_mode,mp,false))<.08,"decode grouped arbitrary mask");
                if(mp)mask.query_offset=1;}
            if(mp){for(int h=0;h<H;++h)for(int j=0;j<D;++j)require(out.ptr()[h*D+j]==0,"hidden row");
                auto saved=out.values();mask.bytes=1;require(rk_npu_mla_f16_run(dev.ctx,mw.get(),c.get(),f.q.data(),nullptr,5,17,out.ptr(),nullptr,mp)==RK_NPU_ERR_PARAM,"mask bounds");require(out.values()==saved,"invalid mask changes output");
                mask.bytes=bytes.size();for(auto& b:bytes)b=0;++mask.version;
                check(rk_npu_mla_f16_run(dev.ctx,mw.get(),c.get(),f.q.data(),nullptr,5,17,out.ptr(),nullptr,mp));require(std::all_of(out.ptr(),out.ptr()+out.count,[](float x){return x==0;}),"versioned hidden mask");
                std::vector<uint8_t> tokens(129,1);rk_npu_attention_f16_boolean_mask_init(&mask,tokens.data(),tokens.size(),1,1,129);mask.version=1;
                check(rk_npu_mla_f16_run(dev.ctx,mw.get(),c.get(),f.q.data(),nullptr,5,17,out.ptr(),nullptr,&mask));require(error(out.values(),f.reference(129,5,17,mask_mode,&mask,false))<.08,"broadcast mask");
            }
        }
        if(mode==RK_NPU_MLA_F16_EXPANDED){
            auto wide_cfg=cfg;wide_cfg.mask_mode=RK_NPU_ATTENTION_F16_BOOLEAN_MASK;
            auto wide=work(dev,wide_cfg,w.get());std::vector<uint8_t> tokens(129,1);
            for(int j=0;j<129;++j)tokens[j]=uint8_t(j%3!=0);
            rk_npu_attention_f16_boolean_mask mask;rk_npu_attention_f16_boolean_mask_init(&mask,tokens.data(),tokens.size(),1,1,129);
            mask.version=1;GuardOutput out(128*H*D);
            check(rk_npu_mla_f16_run(dev.ctx,wide.get(),c.get(),f.q.data(),f.gates.data(),128,0,out.ptr(),nullptr,&mask));
            require(error(out.values(),f.reference(129,128,0,wide_cfg.mask_mode,&mask))<.08,"wide boolean mask");
            std::fill(tokens.begin(),tokens.end(),0);++mask.version;
            check(rk_npu_mla_f16_run(dev.ctx,wide.get(),c.get(),f.q.data(),f.gates.data(),128,0,out.ptr(),nullptr,&mask));
            require(std::all_of(out.ptr(),out.ptr()+out.count,[](float x){return x==0;}),"wide all hidden");
        }
        // Automatic growth and 32-token geometry/tail transitions.
        auto gc=cfg;gc.initial_capacity=32;auto growing=cache(dev,gc,w.get());check(rk_npu_mla_f16_cache_load(dev.ctx,growing.get(),f.latent.data(),f.rope.data(),31));
        for(int length=32;length<=65;++length){GuardOutput out(H*D);
            check(rk_npu_mla_f16_decode_step(dev.ctx,ws.get(),growing.get(),f.q.data(),f.gates.data(),f.latent.data()+size_t(length-1)*C,f.rope.data()+size_t(length-1)*R,out.ptr(),nullptr,nullptr));
            require(error(out.values(),f.reference(length,1,length-1,cfg.mask_mode))<.08,"growing oracle");
        }
        auto foreign=rk_npu_iommu_domain_create(dev.ctx,13);rk_npu_mla_f16_workspace* bad=nullptr;
        require(rk_npu_mla_f16_prepare(foreign,&cfg,w.get(),&bad)==RK_NPU_ERR_DOMAIN && !bad,"foreign domain");rk_npu_iommu_domain_free(foreign);
        canary(dev,core);std::printf("CHECK core=%d mode=%d masks/gate/growth/domain PASS\n",core,int(mode));
    }
    // Retained immutable weights survive caller releasing its own handle.
    auto c=cache(dev,cfg,w.get());auto ws=work(dev,cfg,w.get());w.reset();
    check(rk_npu_mla_f16_cache_load(dev.ctx,c.get(),f.latent.data(),f.rope.data(),33));GuardOutput out(H*D);
    check(rk_npu_mla_f16_run(dev.ctx,ws.get(),c.get(),f.q.data(),nullptr,1,32,out.ptr(),nullptr,nullptr));require(error(out.values(),f.reference(33,1,32,cfg.mask_mode,nullptr,false))<.08,"retained weights");
    std::puts("CHECK MLA board PASS");
}
void long_board(int core,int tile,uint32_t flags=RK_NPU_MLA_F16_FIXED_SHIFT_EXPERIMENTAL){
    BoardLock l1("/tmp/rknpu_lowlevel_submit.lock"),l2("/tmp/rk3588_npu_submit.lock");Device dev;
    I8Canary int8(dev,core);int8.run();
    Fixture f(32768,128);auto weights_handle=weights(dev,f.w.data());
    rk_npu_mla_f16_config cfg;rk_npu_mla_f16_config_init(&cfg);
    cfg.flags=flags;cfg.core_mask=core;cfg.kv_tile=tile;cfg.max_capacity=32768;
    const double tolerance=flags==RK_NPU_MLA_F16_STABLE_SOFTMAX?.006:.08;
    for(auto mode:{RK_NPU_MLA_F16_ABSORBED,RK_NPU_MLA_F16_EXPANDED}){
        cfg.mode=mode;cfg.max_query_rows=mode==RK_NPU_MLA_F16_EXPANDED?128:32;
        auto c=cache(dev,cfg,weights_handle.get());
        for(auto mask_mode:{RK_NPU_ATTENTION_F16_CAUSAL,RK_NPU_ATTENTION_F16_NO_MASK,RK_NPU_ATTENTION_F16_BOOLEAN_MASK}){
            cfg.mask_mode=mask_mode;auto ws=work(dev,cfg,weights_handle.get());auto sc=cfg;sc.core_mask=1;auto single=work(dev,sc,weights_handle.get());
            for(int length:{8191,8192,8193,16383,16384,16385,32767,32768}){
                check(rk_npu_mla_f16_cache_load(dev.ctx,c.get(),f.latent.data(),f.rope.data(),length));
                std::vector<int> qr{1};
                if(length==8193 || length==16385 || length==32768)qr.insert(qr.end(),{5,32});
                if(mode==RK_NPU_MLA_F16_EXPANDED && length==32768)qr.push_back(128);
                for(int rows:qr){
                    std::vector<uint8_t> bytes(size_t(rows+1)*length*2,0);
                    for(int r=0;r<rows;++r)for(int j=0;j<length;++j)
                        bytes[(size_t(r+1)*length+j)*2]=uint8_t((rows==1 || r) && (j+r)%3!=0);
                    rk_npu_attention_f16_boolean_mask mask;rk_npu_attention_f16_boolean_mask_init(&mask,bytes.data(),bytes.size(),1,rows+1,length);
                    mask.head_stride*=2;mask.query_stride*=2;mask.key_stride=2;mask.query_offset=1;mask.version=1;
                    auto* mp=mask_mode==RK_NPU_ATTENTION_F16_BOOLEAN_MASK?&mask:nullptr;
                    GuardOutput out(size_t(rows)*H*D);rk_npu_mla_f16_timings t{};canary(dev,core);
                    check(rk_npu_mla_f16_run(dev.ctx,ws.get(),c.get(),f.q.data(),f.gates.data(),rows,length-rows,out.ptr(),&t,mp));
                    int8.run();
                    auto saved=out.values();double e=error(saved,f.reference(length,rows,length-rows,mask_mode,mp));require(e<tolerance,"long MLA exact-softmax oracle");
                    rk_npu_mla_f16_sizes sizes{};check(rk_npu_mla_f16_query(&cfg,32768,rows,length,&sizes));require(t.attention.tasks==sizes.tasks,"long MLA task sizing");
                    check(rk_npu_mla_f16_run(dev.ctx,single.get(),c.get(),f.q.data(),f.gates.data(),rows,length-rows,out.ptr(),nullptr,mp));
                    int8.run();
                    require(out.values()==saved,"long MLA single/multicore bitexact");
                    if(mp && rows>1)for(int j=0;j<H*D;++j)require(saved[j]==0,"long MLA hidden row");
                    std::printf("CHECK long MLA core=%d tile=%d mode=%d mask=%d L=%d Q=%d rel_l2=%.8f tasks=%d PASS\n",core,tile,int(mode),int(mask_mode),length,rows,e,t.attention.tasks);std::fflush(stdout);
                }
            }
            if(mask_mode==RK_NPU_ATTENTION_F16_CAUSAL){
                // A preloaded future must remain masked when a decode graph
                // switches between a final query and an earlier one.
                for(int rows:{1,5})for(int start:{32768-rows,0,8190,8191,16382,16383,32768-rows}){
                    GuardOutput out(size_t(rows)*H*D);
                    check(rk_npu_mla_f16_run(dev.ctx,ws.get(),c.get(),f.q.data(),f.gates.data(),rows,start,out.ptr(),nullptr,nullptr));
                    require(error(out.values(),f.reference(32768,rows,start,mask_mode))<tolerance,"long MLA earlier/final query mask transition");
                }
                std::puts("CHECK long MLA earlier/final causal queries PASS");
            }
        }
        cfg.mask_mode=RK_NPU_ATTENTION_F16_CAUSAL;auto ws=work(dev,cfg,weights_handle.get());
        for(int boundary:{8192,16384}){
            auto gc=cfg;gc.initial_capacity=boundary;auto growing=cache(dev,gc,weights_handle.get());
            check(rk_npu_mla_f16_cache_load(dev.ctx,growing.get(),f.latent.data(),f.rope.data(),boundary-1));
            for(int length=boundary;length<=boundary+33;++length){
                GuardOutput out(H*D);check(rk_npu_mla_f16_decode_step(dev.ctx,ws.get(),growing.get(),f.q.data(),f.gates.data(),f.latent.data()+size_t(length-1)*C,f.rope.data()+size_t(length-1)*R,out.ptr(),nullptr,nullptr));
                require(error(out.values(),f.reference(length,1,length-1,cfg.mask_mode))<tolerance,"long MLA growing/rebound cache");
            }
            std::printf("CHECK long MLA core=%d tile=%d mode=%d growth=%d..%d PASS\n",core,tile,int(mode),boundary,boundary+33);std::fflush(stdout);
        }
        int8.run();canary(dev,core);
    }
    std::puts("CHECK long MLA board PASS");
}
void expanded_input_board(int core,uint32_t flags=1){
    BoardLock l1("/tmp/rknpu_lowlevel_submit.lock"),l2("/tmp/rk3588_npu_submit.lock");Device dev;
    Fixture f(65,5);rk_npu_mla_f16_config cfg;rk_npu_mla_f16_config_init(&cfg);
    cfg.flags=flags;cfg.core_mask=core;cfg.initial_capacity=32;cfg.max_capacity=96;
    const double tolerance=flags==RK_NPU_MLA_F16_STABLE_SOFTMAX?.006:.08;
    auto c=cache(dev,cfg,nullptr);auto ws=work(dev,cfg,nullptr);
    auto projected=[&](int first,int count){
        std::pair<std::vector<uint16_t>,std::vector<uint16_t>> kv;
        kv.first.resize(size_t(H)*count*QK);kv.second.resize(size_t(H)*count*D);
        for(int h=0;h<H;++h)for(int r=0;r<count;++r){
            const auto* latent=f.cf.data()+size_t(first+r)*C;
            auto* key=kv.first.data()+(size_t(h)*count+r)*QK;
            auto* value=kv.second.data()+(size_t(h)*count+r)*D;
            for(int j=0;j<D;++j){
                key[j]=bits::float_to_half(dot(latent,f.wf.data()+(size_t(h)*2*D+j)*C,C));
                value[j]=bits::float_to_half(dot(latent,f.wf.data()+(size_t(h)*2*D+D+j)*C,C));
            }
            std::copy_n(f.rope.data()+size_t(first+r)*R,R,key+D);
        }
        return kv;
    };
    check(rk_npu_mla_f16_cache_load_expanded(dev.ctx,c.get(),nullptr,nullptr,0));
    auto prefix=projected(0,31);
    check(rk_npu_mla_f16_cache_load_expanded(dev.ctx,c.get(),prefix.first.data(),prefix.second.data(),31));
    require(rk_npu_mla_f16_cache_append(dev.ctx,c.get(),f.latent.data(),f.rope.data(),1)==RK_NPU_ERR_PARAM,"weightless latent update rejected");
    require(rk_npu_mla_f16_cache_append_expanded(dev.ctx,c.get(),nullptr,nullptr,1)==RK_NPU_ERR_PARAM,"expanded null update rejected");
    for(int length=32;length<=65;++length){
        auto token=projected(length-1,1);GuardOutput output(H*D);
        check(rk_npu_mla_f16_cache_append_expanded(dev.ctx,c.get(),token.first.data(),token.second.data(),1));
        check(rk_npu_mla_f16_run(dev.ctx,ws.get(),c.get(),f.q.data(),f.gates.data(),1,length-1,output.ptr(),nullptr,nullptr));
        require(error(output.values(),f.reference(length,1,length-1,cfg.mask_mode))<tolerance,"expanded append/growth oracle");
    }
    GuardOutput output(5*H*D);
    check(rk_npu_mla_f16_run(dev.ctx,ws.get(),c.get(),f.q.data(),f.gates.data(),5,60,output.ptr(),nullptr,nullptr));
    auto expected=output.values();auto all=projected(0,65);
    check(rk_npu_mla_f16_cache_load_expanded(dev.ctx,c.get(),all.first.data(),all.second.data(),65));
    check(rk_npu_mla_f16_run(dev.ctx,ws.get(),c.get(),f.q.data(),f.gates.data(),5,60,output.ptr(),nullptr,nullptr));
    require(output.values()==expected,"expanded load vs append bitexact");
    require(error(expected,f.reference(65,5,60,cfg.mask_mode))<tolerance,"expanded prefill oracle");
    auto short_prefix=projected(0,17);
    check(rk_npu_mla_f16_cache_load_expanded(dev.ctx,c.get(),short_prefix.first.data(),short_prefix.second.data(),17));
    check(rk_npu_mla_f16_run(dev.ctx,ws.get(),c.get(),f.q.data(),nullptr,5,12,output.ptr(),nullptr,nullptr));
    require(error(output.values(),f.reference(17,5,12,cfg.mask_mode,nullptr,false))<tolerance,"expanded rewind oracle");
    rk_npu_mla_f16_cache_info info{};check(rk_npu_mla_f16_cache_get_info(c.get(),&info));
    require(info.length==17 && info.capacity>=65,"expanded reload preserves capacity");
    if(flags==RK_NPU_MLA_F16_STABLE_SOFTMAX){
        // Visible logits exceed the old fixed-shift FP16 exp range. Future
        // keys are much larger and must not enter the causal row maximum.
        std::vector<uint16_t> q(5*H*QK,0x3c00),k(size_t(H)*65*QK),v(size_t(H)*65*D);
        for(int h=0;h<H;++h)for(int token=0;token<65;++token){
            std::fill_n(k.data()+(size_t(h)*65+token)*QK,QK,bits::float_to_half(token<5?2.f:100.f));
            std::fill_n(v.data()+(size_t(h)*65+token)*D,D,bits::float_to_half(token/16.f+h/8.f));
        }
        check(rk_npu_mla_f16_cache_load_expanded(dev.ctx,c.get(),k.data(),v.data(),65));
        check(rk_npu_mla_f16_run(dev.ctx,ws.get(),c.get(),q.data(),nullptr,5,0,output.ptr(),nullptr,nullptr));
        auto values=output.values();
        for(int r=0;r<5;++r)for(int h=0;h<H;++h)for(int j=0;j<D;++j)
            require(std::abs(values[(r*H+h)*D+j]-(r/32.f+h/8.f))<.002,"stable large logits/causal future exclusion");
        auto bc=cfg;bc.mask_mode=RK_NPU_ATTENTION_F16_BOOLEAN_MASK;auto boolean=work(dev,bc,nullptr);
        std::vector<uint8_t> tokens(65,0);std::fill_n(tokens.data(),5,1);
        rk_npu_attention_f16_boolean_mask mask;rk_npu_attention_f16_boolean_mask_init(&mask,tokens.data(),tokens.size(),1,1,65);
        check(rk_npu_mla_f16_run(dev.ctx,boolean.get(),c.get(),q.data(),nullptr,5,0,output.ptr(),nullptr,&mask));
        values=output.values();
        for(int r=0;r<5;++r)for(int h=0;h<H;++h)for(int j=0;j<D;++j)
            require(std::abs(values[(r*H+h)*D+j]-(.125f+h/8.f))<.002,"stable large logits/boolean exclusion");
        std::fill(tokens.begin(),tokens.end(),0);
        check(rk_npu_mla_f16_run(dev.ctx,boolean.get(),c.get(),q.data(),nullptr,5,0,output.ptr(),nullptr,&mask));
        values=output.values();require(std::all_of(values.begin(),values.end(),[](float x){return x==0;}),"stable large logits all-hidden zero");
        // V outside the delayed-normalization bound must keep normalized
        // probabilities. Exercise each side of the bound, a mixed-head
        // cache, append increasing the bound, and load lowering it again.
        constexpr int bound_length=2048;
        auto bound_cfg=cfg;bound_cfg.initial_capacity=bound_cfg.max_capacity=bound_length;
        auto bound_cache=cache(dev,bound_cfg,nullptr);auto bound_work=work(dev,bound_cfg,nullptr);
        std::vector<uint16_t> bound_keys(size_t(H)*bound_length*QK,bits::float_to_half(2.f));
        std::vector<uint16_t> bound_values(size_t(H)*bound_length*D);
        for(float magnitude:{63.f,65.f,1024.f,60000.f,65504.f,1.f}){
            for(int h=0;h<H;++h)for(int token=0;token<bound_length;++token)
                std::fill_n(bound_values.data()+(size_t(h)*bound_length+token)*D,D,bits::float_to_half(h%2?magnitude:1.f));
            check(rk_npu_mla_f16_cache_load_expanded(dev.ctx,bound_cache.get(),bound_keys.data(),bound_values.data(),bound_length));
            check(rk_npu_mla_f16_run(dev.ctx,bound_work.get(),bound_cache.get(),q.data(),nullptr,5,bound_length-5,output.ptr(),nullptr,nullptr));
            values=output.values();
            for(int r=0;r<5;++r)for(int h=0;h<H;++h)for(int j=0;j<D;++j){
                const float expected=bits::half_to_float(bits::float_to_half(h%2?magnitude:1.f));
                require(std::abs(values[(r*H+h)*D+j]-expected)<=std::max(.002f,expected*.002f),"stable V bound/load transition");
            }
        }
        auto prefix=std::vector<uint16_t>(size_t(H)*(bound_length-1)*QK,bits::float_to_half(2.f));
        auto small=std::vector<uint16_t>(size_t(H)*(bound_length-1)*D,bits::float_to_half(1.f));
        check(rk_npu_mla_f16_cache_load_expanded(dev.ctx,bound_cache.get(),prefix.data(),small.data(),bound_length-1));
        check(rk_npu_mla_f16_run(dev.ctx,bound_work.get(),bound_cache.get(),q.data(),nullptr,5,bound_length-6,output.ptr(),nullptr,nullptr));
        auto last_key=std::vector<uint16_t>(size_t(H)*QK,bits::float_to_half(2.f));
        auto large=std::vector<uint16_t>(size_t(H)*D,bits::float_to_half(60000.f));
        check(rk_npu_mla_f16_cache_append_expanded(dev.ctx,bound_cache.get(),last_key.data(),large.data(),1));
        check(rk_npu_mla_f16_run(dev.ctx,bound_work.get(),bound_cache.get(),q.data(),nullptr,5,bound_length-5,output.ptr(),nullptr,nullptr));
        values=output.values();
        for(int r=0;r<5;++r)for(int h=0;h<H;++h)for(int j=0;j<D;++j){
            const float expected=r==4?(60000.f+bound_length-1)/bound_length:1.f;
            require(std::abs(values[(r*H+h)*D+j]-expected)<std::max(.002f,expected*.002f),"stable V bound append transition");
        }
        std::puts("CHECK stable delayed/normalized V bounds and transitions PASS");
    }
    cfg.mode=RK_NPU_MLA_F16_ABSORBED;rk_npu_mla_f16_cache* bad=nullptr;rk_npu_mla_f16_workspace* bad_work=nullptr;
    require(rk_npu_mla_f16_cache_create(dev.domain,&cfg,nullptr,&bad)==RK_NPU_ERR_PARAM && !bad,"absorbed requires weights");
    require(rk_npu_mla_f16_prepare(dev.domain,&cfg,nullptr,&bad_work)==RK_NPU_ERR_PARAM && !bad_work,"absorbed workspace requires weights");
    canary(dev,core);std::printf("CHECK expanded input core=%d load/append/growth/rewind/guards PASS\n",core);
}
int main(int argc,char** argv){try{cpu();if(argc>1 && std::string(argv[1])=="--board"){
    const int core=argc>2?std::stoi(argv[2]):1;
    if(argc>3 && std::string(argv[3])=="--expanded-input")expanded_input_board(core);
    else if(argc>3 && std::string(argv[3])=="--expanded-stable")expanded_input_board(core,RK_NPU_MLA_F16_STABLE_SOFTMAX);
    else if(argc>3 && std::string(argv[3])=="--stable")board(core,true,RK_NPU_MLA_F16_STABLE_SOFTMAX);
    else if(argc>3 && std::string(argv[3])=="--long")long_board(core,argc>4?std::stoi(argv[4]):512);
    else if(argc>3 && std::string(argv[3])=="--long-stable")long_board(core,argc>4?std::stoi(argv[4]):512,RK_NPU_MLA_F16_STABLE_SOFTMAX);
    else board(core,argc>3 && std::string(argv[3])=="--small");
}return 0;}
    catch(const std::exception& e){std::fprintf(stderr,"MLA FAIL: %s\n",e.what());return 1;}}
