#include "rk_npu_matmul_i4.h"
#include "../src/rk_npu_i4_cpu.h"
#include "../src/rk_npu_i4_regs.h"
#include <algorithm>
#include <climits>
#include <cstdio>
#include <stdexcept>
#include <vector>
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while(0)
static int code(const uint8_t* p, size_t i) {
    const int q=(p[i/2] >> ((i&1)?0:4))&15;
    return q>=8?q-16:q;
}
int main() {
    try {
        for (int k:{1,31,32,33,127,479,480,991,992,1472,1984,2048}) {
            const int rows=3,pk=(k+31)/32*32,lda=k+7;
            std::vector<int8_t> a(5*lda);
            for (size_t i=0;i<a.size();++i) a[i]=int(i*7%16)-8;
            std::vector<uint8_t> packed(rows*pk/2+16,0x5a);
            rk_npu_i4_input_tile t{1,rows,2,k,pk,packed.data(),uint64_t(rows*pk/2)};
            CHECK(!rk_npu_i4_pack_a_tile(&t,a.data(),lda));
            for (int m=0;m<rows;++m) for (int j=0;j<pk;++j)
                CHECK(code(packed.data(),(j/32*rows+m)*32+j%32)==(j<k?a[(m+1)*lda+j+2]:0));
            CHECK(std::all_of(packed.end()-16,packed.end(),[](uint8_t x){return x==0x5a;}));
            a[lda+2]=8; CHECK(rk_npu_i4_pack_a_tile(&t,a.data(),lda)==RK_NPU_ERR_PARAM);
        }
        const int K=511,N=67,pk=32,pn=128;
        std::vector<int8_t> b(K*N);
        for (int i=0;i<K*N;++i) b[i]=int(i*11%16)-8;
        std::vector<uint8_t> packed(pk*pn/2);
        CHECK(!rknpu2_matmul_open::detail::pack_i4_weights(K,N,480,31,pk,pn,b.data(),packed.data()));
        for (int k=0;k<pk;++k) for (int n=0;n<pn;++n) {
            const size_t index=(((n/64)*(pk/32)+k/32)*64+n%64)*32+k%32;
            CHECK(code(packed.data(),index)==(k<31&&n<N?b[(480+k)*N+n]:0));
        }
        std::vector<int16_t> c(3*72);
        std::vector<int32_t> out(3*69,100000);
        for (int n=0;n<72;++n) for (int m=0;m<3;++m) c[(n/8*3+m)*8+n%8]=int16_t(-30000+101*m+17*n);
        rknpu2_matmul_open::detail::reduce_i4_partial(3,67,72,c.data(),out.data(),69,true);
        for (int m=0;m<3;++m) for (int n=0;n<69;++n)
            CHECK(out[m*69+n]==(n<67?70000+101*m+17*n:100000));
        rk_npu_i4_config cfg; rk_npu_i4_config_init(&cfg,129,65,513);
        cfg.m_tile=64; cfg.n_tile=64;
        rk_npu_i4_memory_info info{};
        CHECK(!rk_npu_i4_memory_query(&cfg,&info));
        CHECK(info.wave_count==2 && info.tasks_per_wave==6 && info.packed_weight_bytes==34816);
        cfg.k_tile=512; CHECK(!rk_npu_i4_memory_query(&cfg,&info));
        cfg.k_tile=2080; CHECK(rk_npu_i4_memory_query(&cfg,&info)==RK_NPU_ERR_PARAM);
        cfg.k_tile=480; cfg.npu_core_mask=5; CHECK(rk_npu_i4_memory_query(&cfg,&info)==RK_NPU_ERR_PARAM);
        cfg.npu_core_mask=1; cfg.K=INT_MAX; CHECK(rk_npu_i4_memory_query(&cfg,&info)==RK_NPU_ERR_PARAM);
        std::vector<uint64_t> regs;
        CHECK(!rknpu2_matmul_open::detail::make_i4_regs(2,480,64,0x1000,0x2000,0x3000,regs));
        CHECK(regs.size()==108 && (regs[34]&65535)==0x1110 && ((regs[34]>>16)&UINT32_MAX)==0x2000);
        CHECK(!rknpu2_matmul_open::detail::make_i4_regs(2,2048,64,0,0,0,regs));
        CHECK(rknpu2_matmul_open::detail::make_i4_regs(2,2080,64,0,0,0,regs)==RK_NPU_ERR_PARAM);
        std::puts("PASS i4 CPU: signed codes, padding, guard, widening, bounds, register layout");
    } catch (const std::exception& e) { std::fprintf(stderr,"FAIL %s\n",e.what()); return 1; }
}
