#pragma once
#include "bangc_common.h"

template <typename T>
__mlu_entry__ void masked_softmax_kernel(const T *x, const bool *keep, T *out,
                                         int64_t rows, int64_t cols, float scale) {
    for (int64_t row = taskId; row < rows; row += taskDim) {
        float maximum = -INFINITY;
        for (int64_t col = 0; col < cols; ++col) {
            int64_t i = row * cols + col;
            if (keep == nullptr || keep[i]) {
                float value = static_cast<float>(static_cast<T>(static_cast<float>(x[i]) * scale));
                if (isnan(value)) { maximum = value; break; }
                if (value > maximum) maximum = value;
            }
        }
        float sum = 0.0f;
        for (int64_t col = 0; col < cols; ++col) {
            int64_t i = row * cols + col;
            if (keep == nullptr || keep[i]) {
                float value = static_cast<float>(static_cast<T>(static_cast<float>(x[i]) * scale));
                sum += expf(value - maximum);
            }
        }
        if (maximum == -INFINITY) sum = NAN;
        for (int64_t col = 0; col < cols; ++col) {
            int64_t i = row * cols + col;
            float value = static_cast<float>(static_cast<T>(static_cast<float>(x[i]) * scale));
            float numerator = keep == nullptr || keep[i] ? expf(value - maximum) : 0.0f;
            out[i] = static_cast<T>(numerator / sum);
        }
    }
}

#ifndef BANG_DEVICE_ONLY
inline torch::Tensor run_masked_softmax(torch::Tensor x, torch::Tensor mask, double scale) {
    TORCH_CHECK(x.dim() >= 1, "expected at least one dimension");
    auto input = x.contiguous();
    auto out = torch::empty(x.sizes(), x.options());
    if (x.numel() == 0) return out;
    torch::Tensor expanded;
    const bool *keep = nullptr;
    if (mask.defined()) {
        TORCH_CHECK(mask.scalar_type() == at::kBool && mask.device() == x.device(),
                    "mask must be boolean and on the input device");
        expanded = mask.expand(x.sizes()).contiguous();
        keep = expanded.data_ptr<bool>();
    }
    int64_t cols = x.size(-1);
    cnrtDim3_t grid = {64, 1, 1};
    auto stream = torch_mlu::getCurMLUStream();
    BANGC_DISPATCH(x,
        masked_softmax_kernel<scalar_t><<<grid, cnrtFuncTypeBlock, stream>>>(
            reinterpret_cast<const scalar_t *>(input.data_ptr()), keep,
            reinterpret_cast<scalar_t *>(out.data_ptr()), x.numel() / cols, cols, static_cast<float>(scale));
    );
    return out;
}
#endif
