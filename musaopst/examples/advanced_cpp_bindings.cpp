// SPDX-License-Identifier: MulanPSL-2.0
#include <torch/extension.h>
void advanced_run_out(const std::string&,std::vector<torch::Tensor>,std::vector<torch::Tensor>,double,double);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("run_out",&advanced_run_out); }
