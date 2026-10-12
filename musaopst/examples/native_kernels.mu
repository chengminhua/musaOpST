// SPDX-License-Identifier: MulanPSL-2.0
// Simple bounded demonstration kernels, not optimized production implementations.
#include "native_host.hpp"
#include <cmath>
using namespace opst;
__device__ int64_t position(Layout layout,int64_t index) {
    int64_t result=layout.offset;
    for(int d=layout.rank-1;d>=0;--d) { result+=(index%layout.shape[d])*layout.strides[d]; index/=layout.shape[d]; }
    return result;
}
__device__ float from_bits(uint32_t bits) { union { uint32_t bits; float value; } v; v.bits=bits; return v.value; }
__device__ uint32_t to_bits(float value) { union { uint32_t bits; float value; } v; v.value=value; return v.bits; }
__device__ float half_read(uint16_t h) {
    uint32_t sign=uint32_t(h&0x8000)<<16, exp=(h>>10)&31, frac=h&1023;
    if(exp==0) { if(!frac) return from_bits(sign); int shift=0; while(!(frac&1024)) { frac<<=1; ++shift; } return from_bits(sign | uint32_t(127-14-shift)<<23 | (frac&1023)<<13); }
    return from_bits(sign | (exp==31?255:exp+112)<<23 | frac<<13);
}
__device__ uint16_t half_write(float value) {
    uint32_t bits=to_bits(value), sign=(bits>>16)&0x8000, fraction=bits&0x7fffff; int exponent=int((bits>>23)&255)-127+15;
    if(((bits>>23)&255)==255) return uint16_t(sign|0x7c00|(fraction?0x0200:0));
    if(exponent>=31) return uint16_t(sign|0x7c00);
    if(exponent<=0) { if(exponent < -10) return uint16_t(sign); uint32_t mantissa=fraction|0x800000; unsigned shift=unsigned(14-exponent); uint32_t rounded=(mantissa+((1u<<(shift-1))-1)+((mantissa>>shift)&1))>>shift; return uint16_t(sign|rounded); }
    uint32_t rounded=fraction+0xfff+((fraction>>13)&1); if(rounded&0x800000) { rounded=0; ++exponent; }
    return uint16_t(sign|(uint32_t(exponent)<<10)|(rounded>>13));
}
__device__ float read_float(const void* data,int64_t i,int dtype) {
    if(dtype==0) return static_cast<const float*>(data)[i];
    uint16_t bits=static_cast<const uint16_t*>(data)[i]; return dtype==1?half_read(bits):from_bits(uint32_t(bits)<<16);
}
__device__ void write_float(void* data,int64_t i,int dtype,float value) {
    if(dtype==0) static_cast<float*>(data)[i]=value;
    else if(dtype==1) static_cast<uint16_t*>(data)[i]=half_write(value);
    else { uint32_t bits=to_bits(value); uint16_t rounded;
        if((bits&0x7f800000)==0x7f800000) rounded=uint16_t((bits>>16)|((bits&0x007fffff)?0x0040:0));
        else rounded=uint16_t((bits+0x7fff+((bits>>16)&1))>>16);
        static_cast<uint16_t*>(data)[i]=rounded; }
}
int floating_dtype(const std::string& dtype) {
    if(dtype=="float32") return 0; if(dtype=="float16") return 1; if(dtype=="bfloat16") return 2;
    throw Unsupported("Native numerical examples support float32/float16/bfloat16; identity supports all protocol dtypes");
}
__global__ void copy_kernel(const uint8_t* a,uint8_t* out,Layout layout,int bytes) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x; if(i>=layout.n) return;
    int64_t from=position(layout,i); for(int b=0;b<bytes;++b) out[i*bytes+b]=a[from*bytes+b];
}
__global__ void axpby_kernel(const void* a,const void* b,void* out,Layout la,Layout lb,int dtype,float alpha,float beta) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x; if(i<la.n) write_float(out,i,dtype,alpha*read_float(a,position(la,i),dtype)+beta*read_float(b,position(lb,i),dtype));
}
__global__ void inplace_kernel(void* a,Layout layout,int dtype,float value) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x; if(i<layout.n) { int64_t offset=position(layout,i); write_float(a,offset,dtype,read_float(a,offset,dtype)+value); }
}
__global__ void fill_kernel(void* out,int64_t n,int dtype,float value) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x; if(i<n) write_float(out,i,dtype,value);
}
__global__ void stats_kernel(const void* a,void* sum,void* max,Layout layout,int dtype) {
    if(blockIdx.x||threadIdx.x) return; float s=0,m=layout.n?-INFINITY:0;
    for(int64_t i=0;i<layout.n;++i) { float value=read_float(a,position(layout,i),dtype); s+=value; m=fmaxf(m,value); }
    write_float(sum,0,dtype,s); write_float(max,0,dtype,m);
}
__global__ void mm_kernel(const void* a,const void* b,void* out,Layout la,Layout lb,int dtype,int64_t m,int64_t k,int64_t n) {
    int64_t col=int64_t(blockIdx.x)*blockDim.x+threadIdx.x,row=int64_t(blockIdx.y)*blockDim.y+threadIdx.y;
    if(row<m && col<n) { float value=0; for(int64_t q=0;q<k;++q) value+=read_float(a,la.offset+row*la.strides[0]+q*la.strides[1],dtype)*read_float(b,lb.offset+q*lb.strides[0]+col*lb.strides[1],dtype); write_float(out,row*n+col,dtype,value); }
}
__global__ void softmax_kernel(const void* a,void* out,Layout layout,int dtype) {
    int64_t row=int64_t(blockIdx.x)*blockDim.x+threadIdx.x; if(row>=layout.shape[0]) return; int64_t n=layout.shape[1]; if(!n) return;
    float max=-INFINITY,denom=0; for(int64_t i=0;i<n;++i) max=fmaxf(max,read_float(a,layout.offset+row*layout.strides[0]+i*layout.strides[1],dtype));
    for(int64_t i=0;i<n;++i) denom+=expf(read_float(a,layout.offset+row*layout.strides[0]+i*layout.strides[1],dtype)-max);
    for(int64_t i=0;i<n;++i) write_float(out,row*n+i,dtype,expf(read_float(a,layout.offset+row*layout.strides[0]+i*layout.strides[1],dtype)-max)/denom);
}
void prepare(Context& c) {
    auto op=c.request.at("operator").str(); auto& in=c.inputs;
    if(op=="fill") { if(!in.empty()) throw Unsupported("fill expects no inputs"); auto& contract=c.request.at("output_contracts").items.at(0);
        auto dtype=c.attrs().has("dtype")?c.attrs().at("dtype").str():contract.at("dtype").str();
        auto shape=integers(c.attrs().has("shape")?c.attrs().at("shape"):contract.at("shape"));
        floating_dtype(dtype); c.output(shape,dtype); return; }
    if(in.empty()) throw Unsupported("Example expects at least one tensor input"); auto& a=*in[0]; auto dtype=a.dtype;
    if(op=="identity") { if(in.size()!=1) throw Unsupported("identity requires one tensor"); c.output(a.shape,dtype); return; }
    floating_dtype(dtype);
    if(op=="axpby" || op=="matmul") { if(in.size()!=2 || in[1]->dtype!=dtype) throw Unsupported("Example needs two matching dtypes");
        if(op=="axpby") { if(a.shape!=in[1]->shape) throw Unsupported("axpby requires same shapes"); c.output(a.shape,dtype); }
        else { if(a.shape.size()!=2 || in[1]->shape.size()!=2 || a.shape[1]!=in[1]->shape[0]) throw Unsupported("matmul requires compatible rank-2 shapes"); c.output({a.shape[0],in[1]->shape[1]},dtype); }
    } else if(op=="sum_max") { if(in.size()!=1 || a.count()>65536) throw Unsupported("sum_max example supports one tensor <=65536 elements"); c.output({},dtype); c.output({},dtype); }
    else if(op=="softmax") { if(in.size()!=1 || a.shape.size()!=2 || a.shape[1]>65536) throw Unsupported("softmax example requires rank2 and <=65536 columns"); if(c.attrs().integer("dim",-1)!=-1 && c.attrs().integer("dim",-1)!=1) throw Unsupported("softmax supports last axis"); c.output(a.shape,dtype); }
    else if(op=="inplace_add") { if(in.size()!=1) throw Unsupported("inplace_add needs one input"); for(size_t i=0;i<a.shape.size();++i) if(a.shape[i]>1 && a.strides[i]==0) throw Unsupported("Cannot mutate overlapping input"); c.alias_output(0); }
    else throw Unsupported("Unknown native example operator "+op);
}
void launch(Context& c) {
    auto op=c.request.at("operator").str(); auto& in=c.inputs; auto& out=c.outputs; auto stream=c.stream;
    if(op=="fill") { auto n=out[0]->count(); if(n) fill_kernel<<<(n+255)/256,256,0,stream>>>(out[0]->device,n,floating_dtype(out[0]->dtype),float(c.attrs().has("value")?c.attrs().at("value").number():0)); return; }
    auto& a=*in[0]; auto layout=a.layout(); int64_t n=a.count();
    if(op=="identity") { if(n) copy_kernel<<<(n+255)/256,256,0,stream>>>(static_cast<uint8_t*>(a.device),static_cast<uint8_t*>(out[0]->device),layout,int(itemsize(a.dtype))); return; }
    int dtype=floating_dtype(a.dtype);
    if(op=="axpby") { if(n) axpby_kernel<<<(n+255)/256,256,0,stream>>>(a.device,in[1]->device,out[0]->device,layout,in[1]->layout(),dtype,float(c.attrs().has("alpha")?c.attrs().at("alpha").number():1),float(c.attrs().has("beta")?c.attrs().at("beta").number():1)); }
    else if(op=="sum_max") stats_kernel<<<1,1,0,stream>>>(a.device,out[0]->device,out[1]->device,layout,dtype);
    else if(op=="softmax") { int64_t rows=a.shape[0]; if(rows && a.shape[1]) softmax_kernel<<<(rows+127)/128,128,0,stream>>>(a.device,out[0]->device,layout,dtype); }
    else if(op=="matmul") { int64_t m=a.shape[0],k=a.shape[1],n=in[1]->shape[1]; if(m&&n) mm_kernel<<<dim3((n+15)/16,(m+15)/16),dim3(16,16),0,stream>>>(a.device,in[1]->device,out[0]->device,layout,in[1]->layout(),dtype,m,k,n); }
    else if(op=="inplace_add") { if(n) inplace_kernel<<<(n+255)/256,256,0,stream>>>(a.device,layout,dtype,float(c.attrs().has("value")?c.attrs().at("value").number():1)); }
}
#ifndef MUSA_OPST_KERNELS_ONLY
int main(int argc,char** argv) { return run_native(argc,argv,prepare,launch); }
#endif
