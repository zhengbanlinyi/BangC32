#pragma once

#include <bang.h>
#include <math.h>
#include <stdint.h>

#ifndef BANG_DEVICE_ONLY
#include <torch/extension.h>
#include <cnrt.h>
#include <framework/core/MLUStream.h>

namespace bangc {

inline void check_device(const torch::Tensor &tensor) {
#ifndef BANGC_CPU_TEST
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, "expected an MLU tensor");
#endif
}

inline void check_floating(const torch::Tensor &tensor) {
    check_device(tensor);
    TORCH_CHECK(tensor.scalar_type() == at::kHalf || tensor.scalar_type() == at::kFloat,
                "expected float16 or float32 input");
}

inline void check_like(const torch::Tensor &tensor, const torch::Tensor &reference) {
    TORCH_CHECK(tensor.device() == reference.device(), "tensor devices must match");
    TORCH_CHECK(tensor.scalar_type() == reference.scalar_type(), "tensor dtypes must match");
}

inline int64_t axis(const torch::Tensor &tensor, int64_t dim) {
    TORCH_CHECK(dim >= -tensor.dim() && dim < tensor.dim(), "dimension out of range");
    return dim < 0 ? dim + tensor.dim() : dim;
}

inline void check_indices(const torch::Tensor &index, int64_t size, bool allow_negative = false) {
    TORCH_CHECK(index.scalar_type() == at::kLong, "indices must be int64");
    auto host = index.to(torch::kCPU).contiguous();
    const auto *values = host.data_ptr<int64_t>();
    for (int64_t i = 0; i < host.numel(); ++i) {
        TORCH_CHECK(values[i] >= (allow_negative ? -size : 0) && values[i] < size,
                    "index out of bounds at position ", i);
    }
}

inline torch::TensorOptions default_options() {
#ifdef BANGC_CPU_TEST
    return torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU);
#else
    int device = 0;
    TORCH_CHECK(cnrtGetDevice(&device) == cnrtSuccess, "cannot obtain the current MLU device");
    return torch::TensorOptions().dtype(torch::kFloat)
        .device(c10::Device(c10::DeviceType::PrivateUse1, device));
#endif
}

}  // namespace bangc

#define BANGC_DISPATCH(TENSOR, ...)                                      \
    do {                                                               \
        bangc::check_floating(TENSOR);                                  \
        if ((TENSOR).scalar_type() == at::kHalf) {                       \
            using scalar_t = half;                                     \
            __VA_ARGS__                                                \
        } else {                                                       \
            using scalar_t = float;                                    \
            __VA_ARGS__                                                \
        }                                                              \
    } while (false)
#endif
