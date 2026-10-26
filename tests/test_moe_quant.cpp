#include "rk_npu_moe_w8.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

static void require(bool ok,const char* msg) {if(!ok)throw std::runtime_error(msg);}
static float quantize(const std::vector<float>& x,std::vector<int8_t>& q) {
    float max=0;for(float v:x)max=std::max(max,std::abs(v));
    const float scale=max?max/127.f:1.f;
    q.resize(x.size());for(size_t i=0;i<x.size();++i)q[i]=int8_t(std::clamp(std::lrint(x[i]/scale),-127L,127L));
    return scale;
}

static void check_staged_slots(rk_npu_iommu_domain* domain,
        const rk_npu_moe_w8_config& config,rk_npu_moe_w8_weights* reference,
        rk_npu_moe_w8_workspace* reference_work) {
    using Weights=std::unique_ptr<rk_npu_moe_w8_weights,decltype(&rk_npu_moe_w8_weights_free)>;
    using Work=std::unique_ptr<rk_npu_moe_w8_workspace,decltype(&rk_npu_moe_w8_workspace_free)>;
    const int H=config.weights.hidden,E=config.weights.routed_experts,K=config.top_k;
    const int rows=5;
    auto one=config;one.top_k=1;
    Weights hot(rk_npu_moe_quant_slots_create(domain,&one.weights),rk_npu_moe_w8_weights_free);
    Weights stage(rk_npu_moe_quant_slots_create(domain,&one.weights),rk_npu_moe_w8_weights_free);
    Work work(rk_npu_moe_quant_workspace_create(domain,&one),rk_npu_moe_w8_workspace_free);
    require(hot&&stage&&work,"create staging test banks");
    const size_t bytes=rk_npu_moe_quant_expert_blob_bytes(reference);
    std::vector<std::vector<uint8_t>> records(E,std::vector<uint8_t>(bytes));
    for(int e=0;e<E;++e) {
        require(!rk_npu_moe_quant_export_expert(reference,e,records[e].data(),bytes),"export staging test record");
        require(!rk_npu_moe_quant_import_expert(hot.get(),e,records[e].data(),bytes),"initialize hot slot");
    }
    std::vector<float> input(rows*H),actual(rows*H),expected(rows*H);
    std::vector<float> routes(rows,1),reference_routes(rows*K,0);
    std::vector<int32_t> physical(rows),source(rows*K);
    for(size_t i=0;i<input.size();++i)input[i]=std::sin(float(i%97)*.13f)*.3f;
    for(int r=0;r<rows;++r){physical[r]=r%E;reference_routes[r*K]=1;}
    auto run_hot=[&] {
        require(!rk_npu_moe_w8_run_f32(work.get(),hot.get(),rows,input.data(),
            physical.data(),routes.data(),actual.data(),nullptr),"run hot slots");
    };
    // An odd number of exchanges leaves hot owning the original staging storage.
    for(int iteration=0;iteration<9;++iteration) {
        std::exception_ptr error;
        std::thread importer([&] {
            try {
                for(int slot=0;slot<E;++slot) {
                    rk_npu_moe_import_timings timing{};
                    const auto& record=records[(iteration+slot)%E];
                    require(!rk_npu_moe_quant_import_expert_timed(stage.get(),slot,
                        record.data(),bytes,&timing),"import while hot computes");
                    require(timing.copy_us>=0 && timing.validate_us>=0 &&
                        timing.gate_sync_us>=0 && timing.down_sync_us>=0 &&
                        timing.total_us>=timing.copy_us,"import timing fields");
                }
            } catch(...) {error=std::current_exception();}
        });
        try {run_hot();} catch(...) {importer.join();throw;}
        importer.join();
        if(error)std::rethrow_exception(error);
        for(int slot=0;slot<E;++slot)
            require(!rk_npu_moe_quant_swap_slots(hot.get(),slot,stage.get(),slot),"promote imported slot");
        for(int r=0;r<rows;++r)for(int k=0;k<K;++k)source[r*K+k]=(iteration+r%E)%E;
        require(!rk_npu_moe_w8_run_f32(reference_work,reference,rows,input.data(),
            source.data(),reference_routes.data(),expected.data(),nullptr),"run staging reference");
        run_hot();
        require(actual==expected,"staged promotion changed expert output");
    }

    std::atomic<bool> started{false},finished{false};
    std::exception_ptr error;
    std::thread importer([&] {
        started=true;
        try {
            for(int i=0;i<2048;++i) {
                require(rk_npu_moe_quant_expert_blob_bytes(stage.get())==bytes,"stable packed record length");
                require(!rk_npu_moe_quant_import_expert(stage.get(),1,records[0].data(),bytes),"import beside slot-zero exchange");
            }
        } catch(...) {error=std::current_exception();}
        finished=true;
    });
    while(!started)std::this_thread::yield();
    size_t swaps=0;
    try {
        while(!finished) {
            require(!rk_npu_moe_quant_swap_slots(hot.get(),0,stage.get(),0),"exchange beside another import");
            ++swaps;
        }
    } catch(...) {importer.join();throw;}
    importer.join();
    if(error)std::rethrow_exception(error);
    require(swaps>0,"concurrent slot exchanges exercised");
    if(swaps%2)require(!rk_npu_moe_quant_swap_slots(hot.get(),0,stage.get(),0),"restore hot slot");
    std::vector<uint8_t> exported(bytes);
    require(!rk_npu_moe_quant_export_expert(stage.get(),1,exported.data(),bytes),"export concurrent import");
    require(exported==records[0],"concurrent import changed packed bytes");
    stage.reset();
    run_hot();
    require(actual==expected,"hot storage did not survive staging-bank destruction");
    std::printf("Staged slot checks passed: 2048 imports, %zu concurrent exchanges\n",swaps);
}

int main() try {
    constexpr int H=2048,F=512,E=3,K=2;
    auto* ctx=rk_npu_open(nullptr);require(ctx,"open NPU");
    auto* domain=rk_npu_iommu_domain_create(ctx,15);require(domain,"create domain");
    std::vector<std::vector<int8_t>> gate(E),down(E);
    std::vector<std::vector<float>> gs(E),ds(E);
    std::vector<rk_npu_moe_w8_expert> experts(E);
    for(int e=0;e<E;++e) {
        gate[e].resize(H*2*F);down[e].resize(F*H);gs[e].resize(2*F);ds[e].resize(H);
        for(size_t i=0;i<gate[e].size();++i)gate[e][i]=int8_t(int((i*7+i/23+e*5)%15)-7);
        for(size_t i=0;i<down[e].size();++i)down[e][i]=int8_t(int((i*3+i/11+e*7)%15)-7);
        for(int i=0;i<2*F;++i)gs[e][i]=.001f*float(i%13+1);
        for(int i=0;i<H;++i)ds[e][i]=.001f*float(i%7+1);
        experts[e]={gate[e].data(),gs[e].data(),down[e].data(),ds[e].data()};
    }
    rk_npu_moe_w8_config cfg;rk_npu_moe_w8_config_init(&cfg,H,F,E,0,K,128);
    auto* weights=rk_npu_moe_quant_weights_create(domain,&cfg.weights,experts.data());
    auto* work=rk_npu_moe_quant_workspace_create(domain,&cfg);
    require(weights&&work,"create quantized FFN");
    for(int rows:{1,5,128,1}) {
        std::vector<float> input(rows*H),output(rows*H),routes(rows*K);
        std::vector<int32_t> ids(rows*K);
        for(size_t i=0;i<input.size();++i)input[i]=std::sin(float(i%97)*.13f)*.3f;
        for(int r=0;r<rows;++r)for(int k=0;k<K;++k){ids[r*K+k]=(r+k)%E;routes[r*K+k]=k?.375f:.625f;}
        require(rk_npu_moe_w8_run_f32(work,weights,rows,input.data(),ids.data(),routes.data(),output.data(),nullptr)==0,"run quantized FFN");
        double delta=0,energy=0;
        std::vector<int> samples{0};if(rows>1)samples.push_back(rows/2);if(rows>2)samples.push_back(rows-1);
        for(int row:samples) {
            std::vector<float> expected(H,0);
            for(int k=0;k<K;++k) {
                int e=ids[row*K+k];std::vector<float> x(H),g(2*F),hidden(F);std::vector<int8_t> q,hq;
                for(int i=0;i<H;++i)x[i]=input[row*H+i];
                float scale=quantize(x,q);
                for(int n=0;n<2*F;++n) {
                    int64_t acc=0;for(int i=0;i<H;++i)acc+=int(q[i])*int(gate[e][i*2*F+n]);
                    g[n]=(float(acc)*scale)*gs[e][n];
                }
                for(int n=0;n<F;++n)hidden[n]=(g[n]/(1.f+std::exp(-g[n])))*g[F+n];
                scale=quantize(hidden,hq);
                for(int n=0;n<H;++n) {
                    int64_t acc=0;for(int i=0;i<F;++i)acc+=int(hq[i])*int(down[e][i*H+n]);
                    float y=(float(acc)*scale)*ds[e][n];expected[n]=std::fma(y,routes[row*K+k],expected[n]);
                }
            }
            for(int n=0;n<H;++n){require(std::isfinite(output[row*H+n]),"nonfinite output");double d=output[row*H+n]-expected[n];delta+=d*d;energy+=double(expected[n])*expected[n];}
        }
        double error=std::sqrt(delta/std::max(energy,1e-30));
        std::printf("W4 rows=%d sampled_rows=%zu relL2=%.9g\n",rows,samples.size(),error);
        require(error<2e-5,"integer oracle mismatch");
    }
    auto one=cfg;one.top_k=1;
    auto* slots=rk_npu_moe_quant_slots_create(domain,&one.weights);
    auto* slot_work=rk_npu_moe_quant_workspace_create(domain,&one);
    require(slots&&slot_work,"create physical expert slot");
    const size_t bytes=rk_npu_moe_quant_expert_blob_bytes(weights);
    const size_t stride=(bytes+4095)/4096*4096;
    std::vector<uint8_t> blob(stride),roundtrip(stride);
    for(int expert:{0,2,1,0}) {
        const int physical=(expert+1)%E;
        require(rk_npu_moe_quant_export_expert(weights,expert,blob.data(),blob.size())==0,"export packed expert");
        require(rk_npu_moe_quant_import_expert(slots,physical,blob.data(),blob.size())==0,"import packed expert");
        require(rk_npu_moe_quant_export_expert(slots,physical,roundtrip.data(),roundtrip.size())==0,"export imported slot");
        require(blob==roundtrip,"packed expert byte roundtrip");
        const int rows=5;std::vector<float> input(rows*H),actual(rows*H),expected(rows*H);
        std::vector<int32_t> source_ids(rows*K,expert),slot_ids(rows,physical);
        std::vector<float> source_routes(rows*K,0),slot_routes(rows,1);
        for(int r=0;r<rows;++r)source_routes[r*K]=1;
        for(size_t i=0;i<input.size();++i)input[i]=std::sin(float(i%97)*.13f)*.3f;
        require(rk_npu_moe_w8_run_f32(work,weights,rows,input.data(),source_ids.data(),source_routes.data(),expected.data(),nullptr)==0,"run resident expert");
        require(rk_npu_moe_w8_run_f32(slot_work,slots,rows,input.data(),slot_ids.data(),slot_routes.data(),actual.data(),nullptr)==0,"run replaced slot");
        require(actual==expected,"slot replacement changed expert arithmetic");
    }
    {
        const int rows=5;std::vector<float> input(rows*H),actual(rows*H),expected(rows*H);
        std::vector<int32_t> source_ids(rows*K),slot_ids(rows);
        std::vector<float> source_routes(rows*K,0),slot_routes(rows,1);
        for(int r=0;r<rows;++r){source_ids[r*K]=source_ids[r*K+1]=r%E;source_routes[r*K]=1;slot_ids[r]=(r%E+1)%E;}
        for(size_t i=0;i<input.size();++i)input[i]=std::sin(float(i%97)*.13f)*.3f;
        require(rk_npu_moe_w8_run_f32(work,weights,rows,input.data(),source_ids.data(),source_routes.data(),expected.data(),nullptr)==0,"run resident permutation");
        require(rk_npu_moe_w8_run_f32(slot_work,slots,rows,input.data(),slot_ids.data(),slot_routes.data(),actual.data(),nullptr)==0,"run physical slot permutation");
        require(actual==expected,"physical expert slot permutation changed output");
        std::printf("W4 slot permutation checks passed\n");
    }
    require(rk_npu_moe_quant_import_expert(slots,0,blob.data(),bytes-1)==RK_NPU_ERR_PARAM,"reject short packed record");
    check_staged_slots(domain,cfg,weights,work);
    // Existing W4 cache records must remain usable when a resident shared W8
    // expert is attached. Compare imports/permutations with independently
    // packed mixed weights; the shared expert is never a replaceable slot.
    std::vector<int8_t> shared_gate(H*2*F),shared_down(F*H);
    std::vector<float> shared_gs(2*F,.0003f),shared_ds(H,.0004f);
    for(size_t i=0;i<shared_gate.size();++i)shared_gate[i]=int8_t(int((i*11+i/19)%255)-127);
    for(size_t i=0;i<shared_down.size();++i)shared_down[i]=int8_t(int((i*13+i/17)%255)-127);
    rk_npu_moe_w8_expert shared{shared_gate.data(),shared_gs.data(),shared_down.data(),shared_ds.data()};
    auto mixed_experts=experts;mixed_experts.push_back(shared);
    auto mixed_cfg=cfg;mixed_cfg.weights.shared_expert=1;
    auto* mixed=rk_npu_moe_w4_weights_create(domain,&mixed_cfg.weights,mixed_experts.data());
    auto* mixed_slots=rk_npu_moe_w4_slots_create(domain,&mixed_cfg.weights,&shared);
    require(mixed&&mixed_slots,"create mixed resident/slot weights");
    require(rk_npu_moe_quant_expert_blob_bytes(mixed_slots)==bytes,"mixed W4 cache record layout");
    require(!rk_npu_moe_w4_slots_create(domain,&cfg.weights,&shared),"mixed slots require shared_expert=1");
    require(!rk_npu_moe_w4_slots_create(domain,&mixed_cfg.weights,nullptr),"mixed slots require shared weights");
    require(rk_npu_moe_quant_import_expert(mixed_slots,E,blob.data(),blob.size())==RK_NPU_ERR_PARAM,"shared W8 is not an import slot");
    require(rk_npu_moe_quant_export_expert(mixed_slots,E,blob.data(),blob.size())==RK_NPU_ERR_PARAM,"shared W8 is not a W4 cache record");
    require(rk_npu_moe_quant_import_expert(mixed_slots,0,blob.data(),bytes-1)==RK_NPU_ERR_PARAM,"reject short mixed record");
    for(int cores:{1,2,3}) {
        mixed_cfg.npu_cores=cores;
        auto* mixed_work=rk_npu_moe_w4_workspace_create(domain,&mixed_cfg);
        require(mixed_work,"create mixed workspace");
        int rotation=0;
        for(int rows:{1,5,128,1}) {
            for(int e=0;e<E;++e) {
                const int slot=(e+rotation)%E;
                require(!rk_npu_moe_quant_export_expert(weights,e,blob.data(),blob.size()),"export all-W4 routed record");
                require(!rk_npu_moe_quant_import_expert(mixed_slots,slot,blob.data(),blob.size()),"import mixed routed record");
                require(!rk_npu_moe_quant_export_expert(mixed_slots,slot,roundtrip.data(),roundtrip.size()),"export mixed routed record");
                require(blob==roundtrip,"mixed routed byte roundtrip");
            }
            std::vector<float> input(rows*H),actual(rows*H),expected(rows*H),routes(rows*K);
            std::vector<int32_t> ids(rows*K),physical(rows*K);
            for(size_t i=0;i<input.size();++i)input[i]=std::sin(float(i%97)*.13f)*.3f;
            for(int r=0;r<rows;++r)for(int k=0;k<K;++k) {
                // Include duplicate routes and non-unit sums; shared stays coefficient 1.
                ids[r*K+k]=(r+(r%2?k:0))%E;
                physical[r*K+k]=(ids[r*K+k]+rotation)%E;
                routes[r*K+k]=k?.375f:.875f;
            }
            const int rc=rk_npu_moe_w8_run_f32(mixed_work,mixed,rows,input.data(),ids.data(),routes.data(),expected.data(),nullptr);
            if(rc)std::fprintf(stderr,"mixed resident cores=%d rows=%d rc=%d\n",cores,rows,rc);
            require(!rc,"run independently packed mixed weights");
            require(!rk_npu_moe_w8_run_f32(mixed_work,mixed_slots,rows,input.data(),physical.data(),routes.data(),actual.data(),nullptr),"run mixed slots");
            require(actual==expected,"mixed slot replacement changed routed/shared output");
            for(int row:{0,rows/2,rows-1}) {
                std::vector<float> serial(H);
                require(!rk_npu_moe_w8_run_f32(mixed_work,mixed,1,input.data()+row*H,ids.data()+row*K,routes.data()+row*K,serial.data(),nullptr),"run mixed single-row control");
                require(std::equal(serial.begin(),serial.end(),expected.begin()+row*H),"mixed batch gather changed single-row output");
            }
            ++rotation;
        }
        rk_npu_moe_w8_workspace_free(mixed_work);
    }
    rk_npu_moe_w8_weights_free(mixed_slots);rk_npu_moe_w8_weights_free(mixed);
    std::puts("Mixed W4/W8 cache import and shared-expert checks passed");
    rk_npu_moe_w8_workspace_free(slot_work);rk_npu_moe_w8_weights_free(slots);
    rk_npu_moe_w8_workspace_free(work);rk_npu_moe_w8_weights_free(weights);
    rk_npu_iommu_domain_free(domain);rk_npu_close(ctx);
    std::puts("Quantized MoE checks passed");return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}
