// SPDX-License-Identifier: MulanPSL-2.0
// Original musaopst example; see project LICENSE.
#include <torch/extension.h>
#include <torch_musa/csrc/core/MUSAStream.h>
#include <musa_runtime.h>

// Accumulate fp16/bfloat16 inputs in float, then round once to output dtype.
template <typename scalar_t>
__global__ void add_kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, int64_t n) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < n) out[index] = scalar_t(float(a[index]) + float(b[index]));
}

void vector_add_out(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
    TORCH_CHECK(a.device().type() == c10::DeviceType::PrivateUse1, "a must be a MUSA tensor");
    TORCH_CHECK(b.device() == a.device() && out.device() == a.device(), "device mismatch");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && out.is_contiguous(), "contiguous tensors required");
    TORCH_CHECK(a.dim() == 1 && a.sizes() == b.sizes() && a.sizes() == out.sizes(), "shape mismatch");
    TORCH_CHECK(a.scalar_type() == b.scalar_type() && a.scalar_type() == out.scalar_type(), "dtype mismatch");
    const int64_t n = a.numel();
    if (n == 0) return;
    TORCH_CHECK(n <= 2147483647LL * 256, "grid dimension too large");
    const musaStream_t stream = c10::musa::getCurrentMUSAStream(a.get_device());
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                  a.scalar_type(), "musaopst_vector_add", [&] {
        add_kernel<scalar_t><<<(n + 255) / 256, 256, 0, stream>>>(
            a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n);
    });
    const musaError_t status = musaGetLastError();
    TORCH_CHECK(status == musaSuccess, "vector_add launch failed: ", musaGetErrorString(status));
}
