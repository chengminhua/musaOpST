// SPDX-License-Identifier: MulanPSL-2.0
#include <torch/extension.h>
#include <torch_musa/csrc/core/MUSAStream.h>
#define MUSA_OPST_KERNELS_ONLY
#include "native_kernels.mu"
static Layout tensor_layout(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.dim()<=8,"rank >8 unsupported"); Layout value{}; value.rank=int(tensor.dim()); value.n=tensor.numel(); value.offset=0;
    for(int i=0;i<value.rank;++i) { value.shape[i]=tensor.size(i); value.strides[i]=tensor.stride(i); }
    return value;
}
static int dtype_id(const torch::Tensor& tensor) {
    if(tensor.scalar_type()==at::kFloat) return 0; if(tensor.scalar_type()==at::kHalf) return 1; if(tensor.scalar_type()==at::kBFloat16) return 2;
    TORCH_CHECK(false,"numerical example dtype unsupported"); return -1;
}
void advanced_run_out(const std::string& op, std::vector<torch::Tensor> inputs, std::vector<torch::Tensor> outputs, double first, double second) {
    TORCH_CHECK(!outputs.empty(),"outputs required"); auto device=outputs[0].device();
    TORCH_CHECK(device.type()==c10::DeviceType::PrivateUse1,"MUSA output required");
    for(const auto& tensor:inputs) TORCH_CHECK(tensor.device()==device,"input device mismatch");
    for(const auto& tensor:outputs) TORCH_CHECK(tensor.device()==device,"output device mismatch");
    musaStream_t stream=c10::musa::getCurrentMUSAStream(device.index());
    if(op=="fill") { auto n=outputs[0].numel(); if(n) fill_kernel<<<(n+255)/256,256,0,stream>>>(outputs[0].data_ptr(),n,dtype_id(outputs[0]),float(first)); }
    else {
        TORCH_CHECK(!inputs.empty(),"input required"); const auto& a=inputs[0]; auto la=tensor_layout(a); auto n=a.numel();
        if(op=="identity") { if(n) copy_kernel<<<(n+255)/256,256,0,stream>>>(reinterpret_cast<const uint8_t*>(a.data_ptr()),reinterpret_cast<uint8_t*>(outputs[0].data_ptr()),la,a.element_size()); }
        else if(op=="axpby") { if(n) axpby_kernel<<<(n+255)/256,256,0,stream>>>(a.data_ptr(),inputs[1].data_ptr(),outputs[0].data_ptr(),la,tensor_layout(inputs[1]),dtype_id(a),float(first),float(second)); }
        else if(op=="sum_max") stats_kernel<<<1,1,0,stream>>>(a.data_ptr(),outputs[0].data_ptr(),outputs[1].data_ptr(),la,dtype_id(a));
        else if(op=="inplace_add") { if(n) inplace_kernel<<<(n+255)/256,256,0,stream>>>(a.data_ptr(),la,dtype_id(a),float(first)); }
        else if(op=="matmul") { int64_t m=a.size(0),k=a.size(1),n=inputs[1].size(1); if(m&&n) mm_kernel<<<dim3((n+15)/16,(m+15)/16),dim3(16,16),0,stream>>>(a.data_ptr(),inputs[1].data_ptr(),outputs[0].data_ptr(),la,tensor_layout(inputs[1]),dtype_id(a),m,k,n); }
        else if(op=="softmax") { int64_t rows=a.size(0); if(rows && a.size(1)) softmax_kernel<<<(rows+127)/128,128,0,stream>>>(a.data_ptr(),outputs[0].data_ptr(),la,dtype_id(a)); }
        else TORCH_CHECK(false,"unknown example operation");
    }
    auto error=musaGetLastError(); TORCH_CHECK(error==musaSuccess,"advanced kernel launch failed: ",musaGetErrorString(error));
}
