// SPDX-License-Identifier: MulanPSL-2.0
// Original musaopst example; see project LICENSE.
#include <torch/extension.h>
void vector_add_out(torch::Tensor a, torch::Tensor b, torch::Tensor out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("vector_add_out", &vector_add_out, "MUSA vector add into explicit output");
}
