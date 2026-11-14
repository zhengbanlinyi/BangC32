#pragma once
#include "bangc_common.h"

template <typename T>
__mlu_entry__ void linear_kernel(const T *x, const T *weight, const T *bias, T *out,
                                int64_t rows, int64_t inputs, int64_t outputs) {
    for (int64_t i = taskId; i < rows * outputs; i += taskDim) {
        int64_t row = i / outputs;
        int64_t column = i % outputs;
        float sum = bias == nullptr ? 0.0f : static_cast<float>(bias[column]);
        for (int64_t k = 0; k < inputs; ++k) {
            sum += static_cast<float>(x[row * inputs + k]) * static_cast<float>(weight[column * inputs + k]);
        }
        out[i] = static_cast<T>(sum);
    }
}

#ifndef BANG_DEVICE_ONLY
inline torch::Tensor run_linear(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(x.dim() >= 1 && weight.dim() == 2 && x.size(-1) == weight.size(1),
                "expected x [..., in_features] and weight [out_features, in_features]");
    bangc::check_like(weight, x);
    bool has_bias = bias.defined() && bias.numel() != 0;
    if (has_bias) {
        bangc::check_like(bias, x);
        TORCH_CHECK(bias.dim() == 1 && bias.numel() == weight.size(0), "bias shape mismatch");
    }
    auto input = x.contiguous();
    auto weights = weight.contiguous();
    auto biases = has_bias ? bias.contiguous() : torch::Tensor();
    auto shape = x.sizes().vec();
    shape.back() = weight.size(0);
    auto out = torch::empty(shape, x.options());
    int64_t rows = 1;
    for (int64_t i = 0; i < x.dim() - 1; ++i) rows *= x.size(i);
    cnrtDim3_t grid = {64, 1, 1};
    auto stream = torch_mlu::getCurMLUStream();
    BANGC_DISPATCH(x,
        linear_kernel<scalar_t><<<grid, cnrtFuncTypeBlock, stream>>>(
            reinterpret_cast<const scalar_t *>(input.data_ptr()), reinterpret_cast<const scalar_t *>(weights.data_ptr()),
            has_bias ? reinterpret_cast<const scalar_t *>(biases.data_ptr()) : nullptr,
            reinterpret_cast<scalar_t *>(out.data_ptr()), rows, weight.size(1), weight.size(0));
    );
    return out;
}

#endif
