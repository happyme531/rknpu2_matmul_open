#include "rk_npu_mm.h"
#include "rk_npu_matmul_f16.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(x) do { if (!(x)) throw std::runtime_error(std::string(#x)+" line "+std::to_string(__LINE__)); } while(0)
using Plan = std::unique_ptr<rk_npu_mm_plan, decltype(&rk_npu_mm_plan_free)>;
using Work = std::unique_ptr<rk_npu_mm_workspace, decltype(&rk_npu_mm_workspace_free)>;
using Weight = std::unique_ptr<rk_npu_mm_packed_b, decltype(&rk_npu_mm_packed_b_free)>;
float truncated(float x) { uint32_t b; std::memcpy(&b,&x,4); b&=0xffffe000u;std::memcpy(&x,&b,4);return x; }
int align32(int n) { return (n+31)/32*32; }
rk_npu_mem memory(void* p, uint64_t size) { rk_npu_mem m{}; m.vaddr=p;m.size=size;m.capacity=size;return m; }

struct Case {
    Plan plan{nullptr,rk_npu_mm_plan_free};
    rk_npu_mm_info info{};
    std::vector<float> a,b,c;
    Case(int m,int n,int k,int batch,int trans,int broadcast,int layouts,int nt,int split,uint32_t mask=7) {
        rk_npu_mm_desc d;rk_npu_mm_desc_init(&d,m,n,k,batch);
        d.a_type=d.b_type=RK_NPU_MM_TF32;d.c_type=RK_NPU_MM_F32;
        d.trans_a=trans&1;d.trans_b=(trans>>1)&1;
        d.lda=(d.trans_a?m:k)+3;d.ldb=(d.trans_b?k:n)+5;d.ldc=n+7;
        d.batch_stride_a=broadcast&1?0:(d.trans_a?k:m)*d.lda+11;
        d.batch_stride_b=broadcast&2?0:(d.trans_b?n:k)*d.ldb+13;
        d.batch_stride_c=m*d.ldc+17;
        rk_npu_mm_options o;rk_npu_mm_options_init(&o);
        o.a_layout=layouts&1?RK_NPU_MM_LAYOUT_NATIVE:RK_NPU_MM_LAYOUT_NORMAL;
        o.c_layout=layouts&2?RK_NPU_MM_LAYOUT_NATIVE:RK_NPU_MM_LAYOUT_NORMAL;
        o.n_tile=nt;o.split_k=split;o.allowed_npu_core_mask=mask;
        rk_npu_mm_plan* p=nullptr;CHECK(rk_npu_mm_plan_create(&d,&o,&p)==0);plan.reset(p);
        CHECK(rk_npu_mm_plan_get_info(p,&info)==0);
        a.resize(info.host_a_bytes/4+16,12345);b.resize(info.host_b_bytes/4+16,12345);c.resize(info.host_c_bytes/4+16,12345);
        for(int z=0;z<batch;++z) {
            for(int r=0;r<m;++r)for(int q=0;q<k;++q)a[ai(z,r,q)]=float((r*7+q*3+z*5)%23-11)/37;
            for(int q=0;q<k;++q)for(int s=0;s<n;++s)b[bi(z,q,s)]=float((q*5+s*11+z*3)%19-9)/43;
        }
    }
    uint64_t ai(int z,int m,int k)const {const auto& d=info.desc;return z*d.batch_stride_a+(d.trans_a?k*d.lda+m:m*d.lda+k);}
    uint64_t bi(int z,int k,int n)const {const auto& d=info.desc;return z*d.batch_stride_b+(d.trans_b?n*d.ldb+k:k*d.ldb+n);}
    uint64_t ci(int z,int m,int n)const {const auto& d=info.desc;return z*d.batch_stride_c+m*d.ldc+n;}
    void verify()const {
        const auto& d=info.desc;std::vector<bool> written(c.size(),false);double err=0,norm=0;
        for(int z=0;z<d.batch_count;++z)for(int m=0;m<d.M;++m)for(int n=0;n<d.N;++n) {
            double ref=0,absolute=0;
            for(int k=0;k<d.K;++k) {double v=double(truncated(a[ai(z,m,k)]))*truncated(b[bi(z,k,n)]);ref+=v;absolute+=std::fabs(v);}
            const float got=c[ci(z,m,n)];CHECK(std::isfinite(got));
            CHECK(std::fabs(got-ref)<=2e-6*absolute+1e-30);
            err+=(got-ref)*(got-ref);norm+=ref*ref;written[ci(z,m,n)]=true;
        }
        for(size_t j=0;j<c.size();++j)if(!written[j])CHECK(c[j]==12345);
        const double rel=std::sqrt(err/std::max(norm,1e-300));CHECK(rel<3e-6);
        std::printf(" rel_l2=%.3g",rel);
    }
};

void cpu_layout(Case& t) {
    const auto& i=t.info;const auto& d=i.desc;const int ak=(i.part_k+15)/16*16,an=align32(d.N);
    const int items=d.batch_count*i.split_count;
    std::vector<float> ap(i.device.input_bytes/4+16,12345),bp(i.device.weight_bytes/4+16,12345);
    auto am=memory(ap.data(),i.device.input_bytes),bm=memory(bp.data(),i.device.weight_bytes);
    CHECK(rk_npu_mm_pack_a(t.plan.get(),t.a.data(),i.host_a_bytes,&am)==0);
    CHECK(rk_npu_mm_pack_b(t.plan.get(),t.b.data(),i.host_b_bytes,&bm)==0);
    // Independent byte-level oracle: TF32 K16 words are FP16 K32 halfwords.
    // The old packers only move uint16_t bit patterns; they do no FP16 math.
    rk_npu_matmul_f16_config cfg;rk_npu_matmul_f16_config_init(&cfg,d.M,d.N,2*ak,RK_NPU_FUSE_NONE);
    cfg.a_layout=i.options.a_layout==RK_NPU_MM_LAYOUT_NATIVE?RK_NPU_F16_A_LAYOUT_NATIVE_K8_M8:RK_NPU_F16_A_LAYOUT_NORMAL;
    rk_npu_matmul_sizes one{};CHECK(rk_npu_matmul_f16_query(&cfg,&one)==0);
    std::vector<uint16_t> ah(d.M*2*ak),bh(2*ak*d.N),ao(one.input_bytes/2),bo(one.weight_bytes/2);
    auto aom=memory(ao.data(),one.input_bytes),bom=memory(bo.data(),one.weight_bytes);
    for(int item=0;item<items;++item) {
        std::fill(ah.begin(),ah.end(),0);std::fill(bh.begin(),bh.end(),0);
        const int z=item/i.split_count,k0=(item%i.split_count)*i.part_k;
        for(int k=0;k<i.part_k && k0+k<d.K;++k) {
            for(int m=0;m<d.M;++m)std::memcpy(&ah[m*2*ak+2*k],&t.a[t.ai(z,m,k0+k)],4);
            for(int n=0;n<d.N;++n) {uint16_t halves[2];std::memcpy(halves,&t.b[t.bi(z,k0+k,n)],4);
                bh[(2*k)*d.N+n]=halves[0];bh[(2*k+1)*d.N+n]=halves[1];}
        }
        CHECK(rk_npu_matmul_f16_pack_a(&cfg,ah.data(),&aom)==0);
        CHECK(rk_npu_matmul_f16_pack_b(&cfg,bh.data(),&bom)==0);
        CHECK(std::memcmp(ao.data(),reinterpret_cast<const char*>(ap.data())+item*one.input_bytes,one.input_bytes)==0);
        CHECK(std::memcmp(bo.data(),reinterpret_cast<const char*>(bp.data())+item*one.weight_bytes,one.weight_bytes)==0);
    }
    CHECK(ap.back()==12345 && bp.back()==12345);
    // Reference de-tiling: double N into halfwords, then reinterpret pairs.
    std::vector<float> out(i.device.output_bytes/4),expected(size_t(d.batch_count)*d.M*d.N,0);
    for(size_t j=0;j<out.size();++j)out[j]=float(int(j%31)-15)/16;
    auto om=memory(out.data(),i.device.output_bytes);
    CHECK(rk_npu_mm_unpack_c(t.plan.get(),&om,t.c.data(),i.host_c_bytes)==0);
    cfg.N=2*an;cfg.n_tile=2*i.options.n_tile;
    cfg.d_layout=i.options.c_layout==RK_NPU_MM_LAYOUT_NATIVE?RK_NPU_F16_D_LAYOUT_NATIVE_N8_M8:RK_NPU_F16_D_LAYOUT_NORMAL_PADDED;
    CHECK(rk_npu_matmul_f16_query(&cfg,&one)==0);
    std::vector<uint16_t> raw(one.output_bytes/2),dense(d.M*2*an);std::vector<float> vals(d.M*an);
    for(int item=0;item<items;++item) {
        std::memcpy(raw.data(),reinterpret_cast<const char*>(out.data())+item*one.output_bytes,one.output_bytes);
        auto rm=memory(raw.data(),one.output_bytes);
        CHECK(rk_npu_matmul_f16_unpack_d(&cfg,&rm,dense.data())==0);
        std::memcpy(vals.data(),dense.data(),vals.size()*4);
        for(int m=0;m<d.M;++m)for(int n=0;n<d.N;++n)
            expected[(size_t(item/i.split_count)*d.M+m)*d.N+n]+=vals[m*an+n];
    }
    std::vector<bool> written(t.c.size(),false);
    for(int z=0;z<d.batch_count;++z)for(int m=0;m<d.M;++m)for(int n=0;n<d.N;++n) {
        CHECK(t.c[t.ci(z,m,n)]==expected[(size_t(z)*d.M+m)*d.N+n]);written[t.ci(z,m,n)]=true;
    }
    for(size_t j=0;j<t.c.size();++j)if(!written[j])CHECK(t.c[j]==12345);
    CHECK(rk_npu_mm_pack_a(t.plan.get(),t.a.data(),i.host_a_bytes-1,&am)==RK_NPU_ERR_NOMEM);
    CHECK(rk_npu_mm_pack_b(t.plan.get(),t.b.data(),i.host_b_bytes-1,&bm)==RK_NPU_ERR_NOMEM);
    CHECK(rk_npu_mm_unpack_c(t.plan.get(),&om,t.c.data(),i.host_c_bytes-1)==RK_NPU_ERR_NOMEM);
}

void cpu_tests() {
    for(int ac=0;ac<4;++ac)for(int tr=0;tr<4;++tr)for(int split:{0,2,3}) {
        Case t(5,65,97,2,tr,tr,ac,32,split);cpu_layout(t);
    }
    for(int k:{1,16,33,48,64}) {Case t(1,1,k,1,0,0,3,0,0);cpu_layout(t);}
    rk_npu_mm_desc d;rk_npu_mm_desc_init(&d,1,32,48,1);d.a_type=d.b_type=RK_NPU_MM_TF32;d.c_type=RK_NPU_MM_F32;
    rk_npu_mm_plan* p=nullptr;CHECK(rk_npu_mm_plan_create(&d,nullptr,&p)==0);rk_npu_mm_plan_free(p);
    d.c_type=RK_NPU_MM_TF32;CHECK(rk_npu_mm_plan_create(&d,nullptr,&p)==RK_NPU_MM_ERR_UNSUPPORTED && !p);
    d.c_type=RK_NPU_MM_F32;d.a_type=RK_NPU_MM_F32;CHECK(rk_npu_mm_plan_create(&d,nullptr,&p)==RK_NPU_MM_ERR_UNSUPPORTED);
    d.a_type=d.b_type=RK_NPU_MM_TF32;d.struct_size+=8;CHECK(rk_npu_mm_plan_create(&d,nullptr,&p)==RK_NPU_ERR_PARAM);
    rk_npu_mm_desc_init(&d,1,32,6000,1);d.a_type=d.b_type=RK_NPU_MM_TF32;d.c_type=RK_NPU_MM_F32;
    rk_npu_mm_options o;rk_npu_mm_options_init(&o);o.split_k=0;
    CHECK(rk_npu_mm_plan_create(&d,&o,&p)==RK_NPU_MM_ERR_UNSUPPORTED && !p);
    o.split_k=3;CHECK(rk_npu_mm_plan_create(&d,&o,&p)==0);rk_npu_mm_plan_free(p);
    std::printf("TF32 CPU PASS: 53 layout cases, raw FP32 packing, shape/type/ABI/CBUF guards\n");
}

Work workspace(rk_npu_iommu_domain* d,Case& t,rk_npu_mm_workspace_mode mode) {
    rk_npu_mm_workspace* w=nullptr;CHECK(rk_npu_mm_workspace_create(d,t.plan.get(),mode,&w)==0);return Work(w,rk_npu_mm_workspace_free);
}
Weight weights(rk_npu_iommu_domain* d,Case& t) {
    rk_npu_mm_packed_b* b=nullptr;CHECK(rk_npu_mm_packed_b_create(d,t.plan.get(),t.b.data(),t.info.host_b_bytes,&b)==0);return Weight(b,rk_npu_mm_packed_b_free);
}
void run_case(rk_npu_iommu_domain* d,Case& t,const char* name) {
    auto w=workspace(d,t,RK_NPU_MM_HOST_DYNAMIC);const auto& i=t.info;
    std::printf("TF32 %s M=%d K=%d N=%d batch=%d split=%d mask=%u",name,i.desc.M,i.desc.K,i.desc.N,i.desc.batch_count,i.split_count,i.actual_npu_core_mask);std::fflush(stdout);
    const int rc=rk_npu_mm_run(w.get(),t.a.data(),i.host_a_bytes,t.b.data(),i.host_b_bytes,t.c.data(),i.host_c_bytes);
    if(rc)std::fprintf(stderr," run rc=%d (%s)\n",rc,rk_npu_strerror(rc));
    CHECK(rc==0);t.verify();
    const auto expected=t.c;auto b=weights(d,t);auto wp=workspace(d,t,RK_NPU_MM_HOST_PACKED_B);
    CHECK(rk_npu_mm_run_packed_b(wp.get(),t.a.data(),i.host_a_bytes,b.get(),t.c.data(),i.host_c_bytes)==0);CHECK(t.c==expected);
    t.a[t.ai(0,0,0)]=1.0017f;
    CHECK(rk_npu_mm_run(w.get(),t.a.data(),i.host_a_bytes,t.b.data(),i.host_b_bytes,t.c.data(),i.host_c_bytes)==0);t.verify();
    std::printf(" PASS\n");
}
void board_tests() {
    std::unique_ptr<rk_npu_ctx,decltype(&rk_npu_close)> ctx(rk_npu_open(nullptr),rk_npu_close);CHECK(ctx);
    std::unique_ptr<rk_npu_iommu_domain,decltype(&rk_npu_iommu_domain_free)> domain(rk_npu_iommu_domain_create(ctx.get(),0),rk_npu_iommu_domain_free);CHECK(domain);
    for(int ac=0;ac<4;++ac) {Case t(7,80,33,2,3,2,ac,32,0);run_case(domain.get(),t,"layouts_transpose_broadcast");}
    Case gemv(1,33,48,1,0,0,3,32,0,4);run_case(domain.get(),gemv,"gemv_k48_core2");
    Case mn(123,96,1280,2,1,1,3,64,0,3);run_case(domain.get(),mn,"mn_tiled_bmm");
    Case normal(16,64,2048,1,0,0,0,32,0);run_case(domain.get(),normal,"normal_k2048");
    Case k4096(16,64,4096,1,0,0,3,0,0);run_case(domain.get(),k4096,"native_k4096");
    Case k5120(26,64,5120,1,0,0,3,0,-1);run_case(domain.get(),k5120,"k5120_auto_split");
    Case split(99,65,2049,1,3,0,3,32,-1);CHECK(split.info.split_count==3);run_case(domain.get(),split,"auto_split3");
    Case split2(5,80,3841,2,2,2,0,32,2,3);run_case(domain.get(),split2,"split2_bmm");
    Case ffn(4,96,11008,1,0,0,3,32,-1);run_case(domain.get(),ffn,"ffn_auto_split");
    setenv("RK_NPU_MM_SPLIT_K","0",1);
    Case disabled(3,65,2049,1,0,0,3,32,-1);CHECK(disabled.info.split_count==1);run_case(domain.get(),disabled,"env_disabled");
    unsetenv("RK_NPU_MM_SPLIT_K");
    auto b=weights(domain.get(),split);Case reuse(3,65,2049,1,3,0,0,0,3);auto w=workspace(domain.get(),reuse,RK_NPU_MM_HOST_PACKED_B);
    CHECK(rk_npu_mm_run_packed_b(w.get(),reuse.a.data(),reuse.info.host_a_bytes,b.get(),reuse.c.data(),reuse.info.host_c_bytes)==0);reuse.verify();
    // Device entry includes explicit ownership sync and host FP32 reduction.
    auto wd=workspace(domain.get(),split2,RK_NPU_MM_DEVICE_ONLY);const auto& i=split2.info;
    rk_npu_mem a{},weight{},c{};
    CHECK(rk_npu_mem_alloc(domain.get(),i.device.input_bytes,RK_NPU_MEM_DATA_DEFAULT,&a)==0);
    CHECK(rk_npu_mem_alloc(domain.get(),i.device.weight_bytes,RK_NPU_MEM_DATA_DEFAULT,&weight)==0);
    CHECK(rk_npu_mem_alloc(domain.get(),i.device.output_bytes,RK_NPU_MEM_DATA_DEFAULT,&c)==0);
    CHECK(rk_npu_mm_pack_a(split2.plan.get(),split2.a.data(),i.host_a_bytes,&a)==0);
    CHECK(rk_npu_mm_pack_b(split2.plan.get(),split2.b.data(),i.host_b_bytes,&weight)==0);
    CHECK(rk_npu_mem_sync(ctx.get(),&a,RK_NPU_SYNC_TO_DEVICE)==0);CHECK(rk_npu_mem_sync(ctx.get(),&weight,RK_NPU_SYNC_TO_DEVICE)==0);
    CHECK(rk_npu_mm_run_device(wd.get(),&a,&weight,&c)==0);CHECK(rk_npu_mem_sync(ctx.get(),&c,RK_NPU_SYNC_FROM_DEVICE)==0);
    CHECK(rk_npu_mm_unpack_c(split2.plan.get(),&c,split2.c.data(),i.host_c_bytes)==0);split2.verify();
    rk_npu_mem_free(ctx.get(),&a);rk_npu_mem_free(ctx.get(),&weight);rk_npu_mem_free(ctx.get(),&c);
    // Public API rounding discriminator: halfway and near-halfway values on
    // both sides of zero must truncate, not silently receive host-side RNE.
    Case rounding(16,32,1,1,3,0,3,0,0);
    const uint32_t raw[]={0x3f800fff,0x3f801000,0x3f801001,0x3f801fff,0x3f802000,0x3f802fff,0x3f803000,0x3f803fff,
                          0xbf800fff,0xbf801000,0xbf801001,0xbf801fff,0xbf802000,0xbf802fff,0xbf803000,0xbf803fff};
    for(int m=0;m<16;++m)std::memcpy(&rounding.a[rounding.ai(0,m,0)],&raw[m],4);
    for(int n=0;n<32;++n)rounding.b[rounding.bi(0,0,n)]=1;
    auto wr=workspace(domain.get(),rounding,RK_NPU_MM_HOST_DYNAMIC);
    CHECK(rk_npu_mm_run(wr.get(),rounding.a.data(),rounding.info.host_a_bytes,rounding.b.data(),rounding.info.host_b_bytes,rounding.c.data(),rounding.info.host_c_bytes)==0);
    for(int m=0;m<16;++m)for(int n=0;n<32;++n)CHECK(rounding.c[rounding.ci(0,m,n)]==truncated(rounding.a[rounding.ai(0,m,0)]));
    for(int sk:{0,3}) {
        Case range(2,32,96,1,3,0,3,32,sk);std::fill(range.a.begin(),range.a.end(),0);std::fill(range.b.begin(),range.b.end(),0);
        range.a[range.ai(0,0,0)]=0x1p80f;range.a[range.ai(0,1,64)]=0x1p-30f;
        for(int n=0;n<32;++n){range.b[range.bi(0,0,n)]=0x1p20f;range.b[range.bi(0,64,n)]=0x1p-30f;}
        auto rw=workspace(domain.get(),range,RK_NPU_MM_HOST_DYNAMIC);
        CHECK(rk_npu_mm_run(rw.get(),range.a.data(),range.info.host_a_bytes,range.b.data(),range.info.host_b_bytes,range.c.data(),range.info.host_c_bytes)==0);
        for(int n=0;n<32;++n){CHECK(range.c[range.ci(0,0,n)]==0x1p100f);CHECK(range.c[range.ci(0,1,n)]==0x1p-60f);}
    }
    // Same K/N/split geometry is insufficient to reuse a BF16 packed B.
    Case typed(4,32,96,1,0,0,0,0,3);auto tw=workspace(domain.get(),typed,RK_NPU_MM_HOST_PACKED_B);
    rk_npu_mm_desc bd=typed.info.desc;bd.a_type=bd.b_type=bd.c_type=RK_NPU_MM_BF16;
    rk_npu_mm_options bo;rk_npu_mm_options_init(&bo);bo.split_k=3;
    rk_npu_mm_plan* raw_plan=nullptr;CHECK(rk_npu_mm_plan_create(&bd,&bo,&raw_plan)==0);Plan bp(raw_plan,rk_npu_mm_plan_free);
    std::vector<uint16_t> bf16(typed.info.host_b_bytes/2,0x3f80);rk_npu_mm_packed_b* raw_b=nullptr;
    CHECK(rk_npu_mm_packed_b_create(domain.get(),bp.get(),bf16.data(),bf16.size()*2,&raw_b)==0);Weight bw(raw_b,rk_npu_mm_packed_b_free);
    const auto before=typed.c;
    CHECK(rk_npu_mm_run_packed_b(tw.get(),typed.a.data(),typed.info.host_a_bytes,bw.get(),typed.c.data(),typed.info.host_c_bytes)==RK_NPU_ERR_PARAM);
    CHECK(typed.c==before);
    std::printf("\nTF32 rounding, range, mixed packed-B guards PASS\n");
    std::printf("\nTF32 BOARD ALL PASS\n");
}
int main(int argc,char** argv) {
    try {
        if(argc==2 && !std::strcmp(argv[1],"--cpu"))cpu_tests();
        else if(argc==2 && !std::strcmp(argv[1],"--board"))board_tests();
        else {std::fprintf(stderr,"usage: test_mm_tf32 --cpu|--board\n");return 2;}
    }catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
    return 0;
}
