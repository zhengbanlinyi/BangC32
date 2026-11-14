#pragma once
#include "bangc_common.h"

template <typename T>
__mlu_entry__ void scan_kernel(const T *x, const bool *mask, float *out,
                              int64_t outer, int64_t length, int64_t inner,
                              bool exclusive, bool reverse) {
    for (int64_t line = taskId; line < outer * inner; line += taskDim) {
        int64_t base = (line / inner) * length * inner + line % inner;
        float sum = 0.0f;
        for (int64_t step = 0; step < length; ++step) {
            int64_t index = base + (reverse ? length - 1 - step : step) * inner;
            float value = static_cast<float>(x[index]);
            if (mask != nullptr) {
                value *= mask[index] ? 1.0f : 0.0f;
            }
            if (exclusive) {
                out[index] = sum;
                sum += value;
            } else {
                sum += value;
                out[index] = sum;
            }
        }
    }
}

#ifndef BANG_DEVICE_ONLY
inline torch::Tensor run_scan(torch::Tensor x, int dim, torch::Tensor mask,
                              bool exclusive, bool reverse) {
    bangc::check_floating(x);
    int64_t axis = bangc::axis(x, dim);
    auto input = x.contiguous();
    torch::Tensor expanded_mask;
    const bool *mask_ptr = nullptr;
    if (mask.defined()) {
        TORCH_CHECK(mask.scalar_type() == at::kBool && mask.device() == x.device(),
                    "mask must be boolean and on the input device");
        expanded_mask = mask.expand(x.sizes()).contiguous();
        mask_ptr = expanded_mask.data_ptr<bool>();
    }
    auto out = torch::empty(x.sizes(), x.options().dtype(torch::kFloat));
    int64_t outer = 1;
    int64_t inner = 1;
    for (int64_t i = 0; i < axis; ++i) outer *= x.size(i);
    for (int64_t i = axis + 1; i < x.dim(); ++i) inner *= x.size(i);
    cnrtDim3_t grid = {64, 1, 1};
    auto stream = torch_mlu::getCurMLUStream();
    BANGC_DISPATCH(x,
        scan_kernel<scalar_t><<<grid, cnrtFuncTypeBlock, stream>>>(
            reinterpret_cast<const scalar_t *>(input.data_ptr()), mask_ptr, out.data_ptr<float>(),
            outer, x.size(axis), inner, exclusive, reverse);
    );
    return out;
}
#endif
