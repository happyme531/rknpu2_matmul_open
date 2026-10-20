#include "rk_npu_attention_f16_internal.h"
#include "rk_npu_cpu_kernels.h"
#include "rk_npu_trace.h"
#include <cmath>
#include <cstdlib>
#include <charconv>
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#include <arm_neon.h>
#endif

namespace rknpu2_matmul_open::attention {
DirtySyncPolicy experimental_dirty_sync_policy()noexcept{
    DirtySyncPolicy policy;
    const char* threshold=std::getenv("RK_NPU_ATTN_DIRTY_SYNC_AFTER");
    if(threshold){
        // Strict nonnegative decimal tokens. Invalid/overflow disables the
        // experiment rather than accidentally removing the length gate.
        if(threshold[0]<'0' || threshold[0]>'9')return {};
        const char* end=threshold+std::strlen(threshold);
        auto parsed=std::from_chars(threshold,end,policy.after);
        if(parsed.ec!=std::errc{} || parsed.ptr!=end)return {};
    }
    const char* value=std::getenv("RK_NPU_ATTN_DIRTY_SYNC_GROUPS");
    if(!value)policy.groups=threshold?1:0;
    else if(!std::strcmp(value,"span"))policy.groups=32;
    else if(value[0] && !value[1] && (value[0]=='1' || value[0]=='2' || value[0]=='4'))policy.groups=value[0]-'0';
    return policy;
}
int validate(const rk_npu_attention_f16_config& c){
    if(c.kv_heads<1 || c.kv_heads>32 || c.query_heads<c.kv_heads ||
       c.query_heads%c.kv_heads || c.head_dim<32 || c.head_dim%32 ||
       !std::isfinite(c.softmax_scale) || c.softmax_scale<0 ||
       c.max_query_rows<1 || c.max_query_rows>128 || c.initial_capacity<32 ||
       c.initial_capacity%32 || c.max_capacity<c.initial_capacity ||
       c.max_capacity%32 || (c.kv_tile!=256 && c.kv_tile!=512 && c.kv_tile!=1024 &&
       c.kv_tile!=2048 && c.kv_tile!=4096) ||
       (c.core_mask!=0 && c.core_mask!=1 && c.core_mask!=2 && c.core_mask!=3 && c.core_mask!=4 && c.core_mask!=7) ||
       (c.flags!=RK_NPU_ATTENTION_F16_FIXED_SHIFT_EXPERIMENTAL && c.flags!=RK_NPU_ATTENTION_F16_STABLE_SOFTMAX) || !std::isfinite(c.exp_shift) ||
       (c.mask_mode!=RK_NPU_ATTENTION_F16_CAUSAL && c.mask_mode!=RK_NPU_ATTENTION_F16_NO_MASK &&
        c.mask_mode!=RK_NPU_ATTENTION_F16_BOOLEAN_MASK))
        return RK_NPU_ERR_PARAM;
    return RK_NPU_OK;
}
Shape gqa_shape(const rk_npu_attention_f16_config& c){
    return {c.head_dim,c.head_dim,c.query_heads/c.kv_heads,false,
        c.softmax_scale?c.softmax_scale:1/std::sqrt(float(c.head_dim))};
}

static void ones(Buffer& b,int heads,int blocks,int n,int tile,int d,int length){
    for(int hi=0;hi<heads;++hi)for(int pos=0;pos<length;++pos){
        const int block=pos/tile,inner=pos%tile;
        b.data()[size_t(hi*blocks+block)*n*tile+((d/16)*(tile/32)+inner/32)*512+inner%32]=half(1);
    }
}
static uint16_t value_bound(const uint16_t* source,size_t count,uint16_t previous=0){
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    auto maximum=vdupq_n_u16(previous);
    for(;count>=8;source+=8,count-=8)
        maximum=vmaxq_u16(maximum,vandq_u16(vld1q_u16(source),vdupq_n_u16(0x7fff)));
    previous=vmaxvq_u16(maximum);
#endif
    for(size_t i=0;i<count;++i)previous=std::max(previous,uint16_t(source[i]&0x7fff));
    return previous;
}
void Cache::reserve(int requested){
    RK_NPU_TRACE_EVENT("rknpu.detail", "attention.cache_reserve", "requested", requested,
        "capacity", capacity, "kv_length", length);
    if(requested<0)throw Error{RK_NPU_ERR_PARAM};
    if(requested>cfg.max_capacity)throw Error{RK_NPU_ERR_NOMEM};
    if(requested<=capacity)return;
    const int next=align_up(requested,32),nb=ceil_div(next,cfg.kv_tile),d=shape.key_dim;
    // Large, non-power-of-two KV buffers otherwise request power-of-two IOVA
    // alignment. Growing several caches can exhaust aligned gaps in the 32-bit
    // domain while substantial space remains free. Page alignment is sufficient
    // for native K/V DMA; retain the old policy for older/non-IOMMU drivers.
    uint32_t alloc_flags=RK_NPU_MEM_DATA_DEFAULT;
    if(lease.domain->ctx->iommu_enabled && lease.domain->ctx->driver_version>=908)
        alloc_flags|=RK_NPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT;
    Buffer nk,nv;nk.alloc(lease.domain,size_t(h())*next*d*2,alloc_flags);
    nv.alloc(lease.domain,size_t(h())*nb*n()*cfg.kv_tile*2,alloc_flags);
    nk.zero();nv.zero();
    if(length){
        const size_t kcount=size_t(align_up(length,16))*d;
        const size_t vcount=size_t(ceil_div(length,cfg.kv_tile))*n()*cfg.kv_tile;
        for(int hi=0;hi<h();++hi){
            std::memcpy(nk.data()+size_t(hi)*next*d,key.data()+size_t(hi)*capacity*d,kcount*2);
            std::memcpy(nv.data()+size_t(hi)*nb*n()*cfg.kv_tile,value.data()+size_t(hi)*blocks*n()*cfg.kv_tile,vcount*2);
        }
    }
    nk.sync(RK_NPU_SYNC_TO_DEVICE);nv.sync(RK_NPU_SYNC_TO_DEVICE);
    key.swap(nk);value.swap(nv);capacity=next;blocks=nb;++generation;
    if(lut_exp){value_total.resize(size_t(h())*shape.value_dim);value_prefix.resize(size_t(capacity/32+1)*h()*shape.value_dim);}
    value_absmax_prefix.resize(size_t(capacity/32+1)*h());
    tail_epoch=~uint64_t(0);
}

void Cache::write(int position,const uint16_t* k,const uint16_t* v,int stride,int src_position){
    const int d=shape.key_dim,tile=cfg.kv_tile,block=position/tile,inner=position%tile;
    for(int hi=0;hi<h();++hi){
        for(int c=0;c<d;++c)
            key.data()[size_t(hi)*capacity*d+((size_t(position/16)*(d/32)+c/32)*16+position%16)*32+c%32]=k[(size_t(hi)*stride+src_position)*d+c];
        for(int c=0;c<shape.value_dim;++c)
            value.data()[voffset(hi,block)+((size_t(c/16)*(tile/32)+inner/32)*16+c%16)*32+inner%32]=v[(size_t(hi)*stride+src_position)*shape.value_dim+c];
    }
    for(int hi=0;hi<h();++hi)
        value.data()[voffset(hi,block)+((shape.value_dim/16)*(tile/32)+inner/32)*512+inner%32]=half(1);
}

void Cache::load(const uint16_t* k,const uint16_t* v,int count){
    if(count<0 || count>cfg.max_capacity || (count && (!k || !v)))throw Error{RK_NPU_ERR_PARAM};
    reserve(count);
    poisoned=true;length=0;key.zero();value.zero();ones(value,h(),blocks,n(),cfg.kv_tile,shape.value_dim,count);
    if(lut_exp){std::fill(value_total.begin(),value_total.end(),0);add_value_sum(v,count,0);}
    std::fill(value_absmax_prefix.begin(),value_absmax_prefix.end(),uint16_t(0));
    add_value_bound(v,count,0);
    const int d=shape.key_dim,tile=cfg.kv_tile;
    for(int hi=0;hi<h();++hi)for(int pos=0;pos<count;++pos)for(int kb=0;kb<d/32;++kb)
        std::memcpy(key.data()+size_t(hi)*capacity*d+((size_t(pos/16)*(d/32)+kb)*16+pos%16)*32,
                    k+(size_t(hi)*count+pos)*d+kb*32,64);
    for(int hi=0;hi<h();++hi)for(int block=0;block<ceil_div(count,tile);++block){
        const int valid=std::min(tile,count-block*tile);
        auto* dst=value.data()+voffset(hi,block);
        if(valid==tile)cpu::f16_pack_b_native_n16_k32(tile,shape.value_dim,tile,shape.value_dim,v+(size_t(hi)*count+block*tile)*shape.value_dim,dst);
        else for(int c=0;c<shape.value_dim;++c)for(int pos=0;pos<valid;++pos)
            dst[((size_t(c/16)*(tile/32)+pos/32)*16+c%16)*32+pos%32]=v[(size_t(hi)*count+block*tile+pos)*shape.value_dim+c];
    }
    key.sync(RK_NPU_SYNC_TO_DEVICE);value.sync(RK_NPU_SYNC_TO_DEVICE);
    length=count;poisoned=false;++epoch;dirty_first=0;dirty_count=count;tail_epoch=~uint64_t(0);
}

void Cache::append(const uint16_t* k,const uint16_t* v,int count,double* reserve_time){
    RK_NPU_TRACE_EVENT("rknpu", "attention.cache_append", "rows", count,
        "kv_length_before", length, "kv_heads", cfg.kv_heads, "head_dim", cfg.head_dim);
    if(poisoned)throw Error{RK_NPU_ERR_SUBMIT};
    if(count<0 || (count && (!k || !v)))throw Error{RK_NPU_ERR_PARAM};
    if(count>cfg.max_capacity-length)throw Error{RK_NPU_ERR_NOMEM};
    if(!count)return;
    auto before=Clock::now();
    if(length+count>capacity)reserve(std::min(cfg.max_capacity,std::max(length+count,capacity*2)));
    if(reserve_time)*reserve_time=elapsed(before,Clock::now());
    const int first=length,last=first+count,d=shape.key_dim,tile=cfg.kv_tile;
    // Match tail staging: compare the logical post-append length, not capacity
    // or align32 span. A multi-token append uses one policy for the whole call.
    const int dirty_sync_groups=dirty_sync.for_length(last);
    try {
        { RK_NPU_TRACE_EVENT("rknpu.detail", "attention.cache_pack", "rows", count);
        add_value_bound(v,count,first);
        // Single-token decode has no measured packing benefit. Keep its
        // original scatter; block transposes target multi-token prefill.
        if(count==1)write(first,k,v,1,0);
        else {
            // K^T's native B layout contains contiguous K32 channel pieces.
            // V uses N16/K32 transposes, including partial stripe updates.
            // Only new rows are written; old tokens and padding survive.
            for(int hi=0;hi<h();++hi){
                for(int i=0;i<count;++i)for(int kb=0;kb<d/32;++kb){
                    const int pos=first+i;
                    std::memcpy(key.data()+size_t(hi)*capacity*d+
                        ((size_t(pos/16)*(d/32)+kb)*16+pos%16)*32,
                        k+(size_t(hi)*count+i)*d+kb*32,64);
                }
                for(int b=first/tile;b<ceil_div(last,tile);++b){
                    const int lo=std::max(first,b*tile)-b*tile;
                    const int end=std::min(last,(b+1)*tile)-b*tile;
                    auto* dst=value.data()+voffset(hi,b);
                    cpu::f16_pack_b_native_n16_k32_update(end-lo,shape.value_dim,tile,lo,
                        v+(size_t(hi)*count+b*tile+lo-first)*shape.value_dim,dst);
                    for(int pos=lo;pos<end;){
                        const int valid=std::min(end-pos,32-pos%32);
                        std::fill_n(dst+((size_t(shape.value_dim/16)*(tile/32)+pos/32)*512)+pos%32,
                            valid,half(1));
                        pos+=valid;
                    }
                }
            }
        }
        if(lut_exp)add_value_sum(v,count,first);
        }
        { RK_NPU_TRACE_EVENT("rknpu.detail", "attention.cache_sync", "rows", count);
        for(int hi=0;hi<h();++hi){
            const int kfirst=first/16*16,kend=align_up(last,16);
            key.range((size_t(hi)*capacity+kfirst)*d*2,size_t(kend-kfirst)*d*2);
            for(int b=first/tile;b<ceil_div(last,tile);++b){
                const int lo=std::max(first,b*tile)-b*tile,hi_pos=std::min(last,(b+1)*tile)-b*tile;
                if(dirty_sync_groups)
                    dirty_value_ranges(voffset(hi,b)*2,shape.value_dim,tile,lo,hi_pos,dirty_sync_groups,
                        [&](uint64_t offset,uint64_t bytes){value.range(offset,bytes);});
                else {
                    value.range(voffset(hi,b)*2,size_t(shape.value_dim)*tile*2);
                    for(int kb=lo/32;kb<ceil_div(hi_pos,32);++kb)
                        value.range((voffset(hi,b)+((shape.value_dim/16)*(tile/32)+kb)*512)*2,64);
                }
            }
        }
        }
    }catch(...){poisoned=true;throw;}
    length=last;++epoch;dirty_first=first;dirty_count=count;
}

void Cache::truncate(int count){
    if(poisoned)throw Error{RK_NPU_ERR_SUBMIT};
    if(cfg.flags!=RK_NPU_ATTENTION_F16_STABLE_SOFTMAX || count<0 || count>length)
        throw Error{RK_NPU_ERR_PARAM};
    if(count==length)return;
    if(lut_exp){
        const int first=count/32*32,tile=cfg.kv_tile;
        for(int hi=0;hi<h();++hi){
            auto* total=value_total.data()+size_t(hi)*shape.value_dim;
            std::copy_n(value_prefix.data()+(size_t(count/32)*h()+hi)*shape.value_dim,shape.value_dim,total);
            for(int pos=first;pos<count;++pos)for(int c=0;c<shape.value_dim;++c){
                const int inner=pos%tile;
                total[c]+=widen(value.data()[voffset(hi,pos/tile)+((size_t(c/16)*(tile/32)+inner/32)*16+c%16)*32+inner%32]);
            }
            // LUT masking leaves a positive carrier on every physical key.
            // Clear discarded V/denominator rows so correction uses live KV.
            for(int b=count/tile;b<ceil_div(length,tile);++b){
                const int lo=std::max(count,b*tile)-b*tile,end=std::min(length,(b+1)*tile)-b*tile;
                auto* dst=value.data()+voffset(hi,b);
                for(int c=0;c<=shape.value_dim;++c)for(int pos=lo;pos<end;){
                    const int n=std::min(end-pos,32-pos%32);
                    std::fill_n(dst+((size_t(c/16)*(tile/32)+pos/32)*16+c%16)*32+pos%32,n,uint16_t(0));pos+=n;
                }
                value.range(voffset(hi,b)*2,size_t(n())*tile*2);
            }
        }
    }
    // Restore the bound of exactly the retained prefix. A discarded high V
    // must not change LUT/CPU softmax selection after transaction rollback.
    for(int hi=0;hi<h();++hi){
        uint16_t maximum=value_absmax_prefix[size_t(count/32)*h()+hi];
        for(int pos=count/32*32;pos<count;++pos){
            const int inner=pos%cfg.kv_tile;
            const auto* src=value.data()+voffset(hi,pos/cfg.kv_tile);
            for(int c=0;c<shape.value_dim;++c)
                maximum=std::max(maximum,uint16_t(src[((size_t(c/16)*(cfg.kv_tile/32)+inner/32)*16+c%16)*32+inner%32]&0x7fff));
        }
        value_absmax[hi]=maximum;
    }
    length=count;++epoch;dirty_first=count;dirty_count=0;tail_epoch=~uint64_t(0);
    // CPU softmax masks retained native rows beyond length.
}

void Cache::add_value_bound(const uint16_t* v,int count,int first){
    for(int hi=0;hi<h();++hi){
        uint16_t maximum=first?value_absmax[hi]:0;
        const auto* src=v?v+size_t(hi)*count*shape.value_dim:nullptr;
        for(int pos=0;pos<count;++pos){
            maximum=value_bound(src?src+size_t(pos)*shape.value_dim:nullptr,shape.value_dim,maximum);
            if((first+pos+1)%32==0)
                value_absmax_prefix[size_t((first+pos+1)/32)*h()+hi]=maximum;
        }
        value_absmax[hi]=maximum;
    }
}

void Cache::softmax_bounds(int count,uint16_t* out)const{
    if(count<0 || count>length)throw Error{RK_NPU_ERR_PARAM};
    for(int hi=0;hi<h();++hi){
        out[hi]=value_absmax[hi];
        if(count==length || widen(out[hi])<=lut_value_limit(cfg.kv_tile))continue;
        out[hi]=value_absmax_prefix[size_t(count/32)*h()+hi];
    }
    extend_softmax_bounds(count/32*32,count%32,out);
}

void Cache::extend_softmax_bounds(int first,int count,uint16_t* out)const{
    for(int hi=0;hi<h();++hi){
        // A full-cache bound already below every switching threshold can be
        // reused for every prefix. Otherwise continue an exact prefix maximum.
        if(out[hi]==value_absmax[hi])continue;
        for(int pos=first;pos<first+count;++pos){
            const int inner=pos%cfg.kv_tile;
            const auto* src=value.data()+voffset(hi,pos/cfg.kv_tile);
            for(int c=0;c<shape.value_dim;++c)
                out[hi]=std::max(out[hi],uint16_t(src[((size_t(c/16)*(cfg.kv_tile/32)+inner/32)*16+c%16)*32+inner%32]&0x7fff));
        }
    }
}

void Cache::add_value_sum(const uint16_t* v,int count,int first){
    const int d=shape.value_dim;
    for(int hi=0;hi<h();++hi){
        auto* total=value_total.data()+size_t(hi)*d;
        for(int pos=0;pos<count;++pos){
            const auto* src=v+(size_t(hi)*count+pos)*d;
            int c=0;
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
            for(;c+8<=d;c+=8){
                const auto x=vreinterpretq_f16_u16(vld1q_u16(src+c));
                vst1q_f32(total+c,vaddq_f32(vld1q_f32(total+c),vcvt_f32_f16(vget_low_f16(x))));
                vst1q_f32(total+c+4,vaddq_f32(vld1q_f32(total+c+4),vcvt_f32_f16(vget_high_f16(x))));
            }
#endif
            for(;c<d;++c)total[c]+=widen(src[c]);
            if((first+pos+1)%32==0)
                std::copy_n(total,d,value_prefix.data()+(size_t((first+pos+1)/32)*h()+hi)*d);
        }
    }
}

void Cache::stage_tail(int span){
    RK_NPU_TRACE_EVENT("rknpu.detail", "attention.tail_stage", "compute_length", span,
        "kv_tile", cfg.kv_tile, "kv_length", length);
    const int size=span%cfg.kv_tile,block=span/cfg.kv_tile,tile=cfg.kv_tile;
    if(!size)return;
    if(poisoned)throw Error{RK_NPU_ERR_SUBMIT};
    const uint64_t needed=size_t(h())*n()*size*2;
    const bool resize=tail.mem.size<needed;
    if(resize){tail.ensure(lease.domain,std::max(needed,std::min(uint64_t(h())*n()*tile*2,tail.mem.size*2)));++tail_generation;}
    const bool same=!resize && tail_block==block && tail_length==size;
    if(same && tail_epoch==epoch)return;
    const bool incremental=same && tail_epoch+1==epoch && dirty_first>=block*tile && dirty_first+dirty_count<=block*tile+size;
    const int dirty_sync_groups=dirty_sync.for_length(length);
    for(int hi=0;hi<h();++hi){
        auto* dst=tail.data()+size_t(hi)*n()*size;
        const auto* src=value.data()+voffset(hi,block);
        if(incremental){
            for(int c=0;c<=shape.value_dim;++c)for(int pos=dirty_first-block*tile;pos<dirty_first-block*tile+dirty_count;++pos)
                dst[((size_t(c/16)*(size/32)+pos/32)*16+c%16)*32+pos%32]=
                    src[((size_t(c/16)*(tile/32)+pos/32)*16+c%16)*32+pos%32];
        }else for(int group=0;group<n()/16;++group)
            std::memcpy(dst+size_t(group)*size*16,src+size_t(group)*tile*16,size_t(size)*16*2);
        if(incremental && dirty_sync_groups)
            dirty_value_ranges(size_t(hi)*n()*size*2,shape.value_dim,size,
                dirty_first-block*tile,dirty_first-block*tile+dirty_count,dirty_sync_groups,
                [&](uint64_t offset,uint64_t bytes){tail.range(offset,bytes);},dirty_sync_groups==32);
        else tail.range(size_t(hi)*n()*size*2,size_t(incremental?shape.value_dim+16:n())*size*2);
    }
    tail_block=block;tail_length=size;tail_epoch=epoch;
}
} // namespace
