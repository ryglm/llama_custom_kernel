// rmsnorm_binding.cpp
#include <torch/extension.h>

torch::Tensor rmsnorm_forward(torch::Tensor x, torch::Tensor w, double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_forward", &rmsnorm_forward, "RMSNorm forward (CUDA)");
}

