/* Complete integer W4A4 call benchmark and bounded strategy search.
 * Static B packing/plan creation excluded. A packing, synchronization, all
 * K waves and exact INT32 reduction included. Stop on first error/mismatch. */
#include "rk_npu_matmul_i4.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using W=std::unique_ptr<rk_npu_i4i4i32_weights,decltype(&rk_npu_i4i4i32_weights_free)>;
using P=std::unique_ptr<rk_npu_i4_workspace,decltype(&rk_npu_i4_workspace_free)>;
static void check(int rc) { if(rc) throw std::runtime_error(std::string("NPU run stopped: ")+rk_npu_strerror(rc)); }
static double median(std::vector<double> x) { std::sort(x.begin(),x.end()); return (x[(x.size()-1)/2]+x[x.size()/2])/2; }
static std::string freq(const char* path) { std::ifstream f(path);std::string s;f>>s;return s; }

int main(int argc,char** argv) {
    rk_npu_i4_config base; rk_npu_i4_config_init(&base,1,4096,4096);
    int loops=10,warmup=2;bool sweep=false;std::string csv;
    try {
        for (int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            auto value=[&]() { if(i+1>=argc) throw std::runtime_error("missing option value");return std::stoi(argv[++i]); };
            if(arg=="--shape") { base.M=value();base.K=value();base.N=value(); }
            else if(arg=="--m-tile") base.m_tile=value();
            else if(arg=="--k-tile") base.k_tile=value();
            else if(arg=="--n-tile") base.n_tile=value();
            else if(arg=="--mask") base.npu_core_mask=value();
            else if(arg=="--cpu-threads") base.cpu_threads=value();
            else if(arg=="--loops") loops=value();
            else if(arg=="--warmup") warmup=value();
            else if(arg=="--serial") base.pipeline=0;
            else if(arg=="--sweep") sweep=true;
            else if(arg=="--csv" && i+1<argc) csv=argv[++i];
            else if(arg=="--help") {
                std::puts("bench_i4 [--shape M K N] [--mask 1|2|4|3|7] [--m-tile M] [--k-tile K] [--n-tile N]\n"
                          "         [--cpu-threads 1..4] [--serial] [--sweep] [--loops 10] [--warmup 2] [--csv FILE]\n"
                          "Sweep: K128/256/480, M64/128, N512/2048, serial/pipeline; fixed core mask.\n"
                          "M tails deduplicated. Every candidate has sampled exact validation.");
                return 0;
            } else throw std::runtime_error("unknown option: "+arg);
        }
        rk_npu_i4_memory_info info{};check(rk_npu_i4_memory_query(&base,&info));
        if(loops<1||warmup<0) throw std::runtime_error("invalid loop count");
        std::ofstream output;
        if(!csv.empty()) {output.open(csv);if(!output) throw std::runtime_error("cannot open CSV");}
        const std::string header="M,K,N,mt,kt,nt,mask,pipeline,cpu_threads,waves,tasks_per_wave,weight_bytes,pack_us,sync_us,submit_us,reduce_us,total_us,GOPS,sampled_exact";
        if(output) output<<header<<'\n';
        std::printf("integer W4A4 complete calls; NPU=%s Hz DDR=%s Hz; warmup=%d loops=%d\n",
            freq("/sys/class/devfreq/fdab0000.npu/cur_freq").c_str(),
            freq("/sys/class/devfreq/dmc/cur_freq").c_str(),warmup,loops);
        std::unique_ptr<rk_npu_ctx,decltype(&rk_npu_close)> ctx(rk_npu_open(nullptr),rk_npu_close);
        if(!ctx) throw std::runtime_error("cannot open NPU");
        std::unique_ptr<rk_npu_iommu_domain,decltype(&rk_npu_iommu_domain_free)> domain(
            rk_npu_iommu_domain_create(ctx.get(),0),rk_npu_iommu_domain_free);
        if(!domain) throw std::runtime_error("cannot create domain");
        std::mt19937 rng(20260920);
        std::vector<int8_t> a(size_t(base.M)*base.K),b(size_t(base.K)*base.N);
        std::vector<int32_t> c(size_t(base.M)*base.N);
        for(auto& v:a)v=int(rng()%16)-8;
        for(auto& v:b)v=int(rng()%16)-8;
        std::set<int> column_set;
        for(int i=0;i<17;++i)column_set.insert(int(int64_t(i)*(base.N-1)/16));
        std::vector<int> columns(column_set.begin(),column_set.end());
        std::vector<int32_t> expected(size_t(base.M)*columns.size());
        for(int m=0;m<base.M;++m)for(size_t j=0;j<columns.size();++j){
            int64_t sum=0;
            for(int k=0;k<base.K;++k)sum+=int(a[size_t(m)*base.K+k])*int(b[size_t(k)*base.N+columns[j]]);
            expected[size_t(m)*columns.size()+j]=int32_t(sum);
        }
        std::vector<rk_npu_i4_config> configs;
        std::set<std::tuple<int,int,int,int>> seen;
        auto add=[&](rk_npu_i4_config cfg) {
            cfg.m_tile=std::min(cfg.m_tile,cfg.M);
            if(seen.insert({cfg.m_tile,cfg.k_tile,cfg.n_tile,cfg.pipeline}).second)configs.push_back(cfg);
        };
        add(base);
        if(sweep)for(int kt:{128,256,480})for(int mt:{64,128})for(int nt:{512,2048})for(int pipe:{0,1}){
            auto cfg=base;cfg.k_tile=kt;cfg.m_tile=mt;cfg.n_tile=nt;cfg.pipeline=pipe;add(cfg);
        }
        // Reuse each packed K partition through every M/N/pipeline candidate.
        std::map<int,W> weights;
        double best=1e300;rk_npu_i4_config winner{};
        for(const auto& cfg:configs){
            check(rk_npu_i4_memory_query(&cfg,&info));
            if(!weights.count(cfg.k_tile)){
                rk_npu_i4_weight_config wc{cfg.K,cfg.N,cfg.k_tile};
                W w(rk_npu_i4i4i32_weights_create(domain.get(),&wc,b.data()),rk_npu_i4i4i32_weights_free);
                if(!w)throw std::runtime_error("weight creation failed");
                weights.emplace(cfg.k_tile,std::move(w));
            }
            P plan(rk_npu_i4_workspace_create(domain.get(),&cfg),rk_npu_i4_workspace_free);
            if(!plan)throw std::runtime_error("workspace creation failed");
            auto* w=weights.at(cfg.k_tile).get();
            for(int i=0;i<warmup;++i)check(rk_npu_i4i4i32_run(plan.get(),w,a.data(),c.data(),nullptr));
            std::vector<double> pack,sync,submit,reduce,total;
            for(int i=0;i<loops;++i){
                rk_npu_i4_timings t{};check(rk_npu_i4i4i32_run(plan.get(),w,a.data(),c.data(),&t));
                pack.push_back(t.pack_us);sync.push_back(t.sync_us);submit.push_back(t.submit_us);
                reduce.push_back(t.reduce_us);total.push_back(t.total_us);
            }
            for(int m=0;m<base.M;++m)for(size_t j=0;j<columns.size();++j)
                if(c[size_t(m)*base.N+columns[j]]!=expected[size_t(m)*columns.size()+j])
                    throw std::runtime_error("sampled integer mismatch; stop sweep");
            const double p=median(pack),s=median(sync),u=median(submit),r=median(reduce),t=median(total);
            const double gops=2.0*cfg.M*cfg.K*cfg.N/(t*1000);
            std::printf("M=%d K=%d N=%d tile=%d/%d/%d mask=%u pipe=%d cpu=%d total=%.1fus pack=%.1f sync=%.1f submit=%.1f reduce=%.1f %.2fGOPS exact\n",
                cfg.M,cfg.K,cfg.N,cfg.m_tile,cfg.k_tile,cfg.n_tile,cfg.npu_core_mask,cfg.pipeline,cfg.cpu_threads,t,p,s,u,r,gops);
            std::fflush(stdout);
            if(output){
                output<<cfg.M<<','<<cfg.K<<','<<cfg.N<<','<<cfg.m_tile<<','<<cfg.k_tile<<','<<cfg.n_tile<<','
                      <<cfg.npu_core_mask<<','<<cfg.pipeline<<','<<cfg.cpu_threads<<','<<info.wave_count<<','<<info.tasks_per_wave<<','
                      <<info.packed_weight_bytes<<','<<p<<','<<s<<','<<u<<','<<r<<','<<t<<','<<gops<<",1\n";
                output.flush();
            }
            if(t<best){best=t;winner=cfg;}
        }
        std::printf("BEST complete-call median %.1fus: M/K/N tiles=%d/%d/%d mask=%u pipeline=%d; %zu candidates\n",
            best,winner.m_tile,winner.k_tile,winner.n_tile,winner.npu_core_mask,winner.pipeline,configs.size());
    }catch(const std::exception& e){std::fprintf(stderr,"ERROR %s\n",e.what());return 1;}
}
