#pragma once
#include "bangc_common.h"
#include "conv_scalar_kernels.h"

#ifndef BANG_DEVICE_ONLY
inline torch::Tensor run_conv2d(torch::Tensor x, torch::Tensor weight, int stride,
                                int padding, int dilation, int groups) {
    TORCH_CHECK(x.dim() == 4 && weight.dim() == 4, "expected NCHW input and OIHW kernel");
    TORCH_CHECK(stride > 0 && padding >= 0 && dilation > 0 && groups > 0, "invalid convolution parameters");
    TORCH_CHECK(x.size(1) > 0 && x.size(1) % groups == 0 && weight.size(0) % groups == 0 &&
                weight.size(1) == x.size(1) / groups && weight.size(2) == weight.size(3) && weight.size(2) > 0,
                "kernel shape or groups mismatch");
    bangc::check_like(weight, x);
    bangc::check_floating(x);
    int64_t kernel = weight.size(2);
    int64_t numerator_h = x.size(2) + 2LL * padding - dilation * (kernel - 1) - 1;
    int64_t numerator_w = x.size(3) + 2LL * padding - dilation * (kernel - 1) - 1;
    TORCH_CHECK(numerator_h >= 0 && numerator_w >= 0, "kernel is larger than the padded input");
    int64_t out_h = numerator_h / stride + 1;
    int64_t out_w = numerator_w / stride + 1;
    auto input = x.contiguous();
    auto weights = weight.contiguous();
    auto out = torch::empty({x.size(0), weight.size(0), out_h, out_w}, x.options());
    TORCH_CHECK(x.numel() <= INT32_MAX && weight.numel() <= INT32_MAX && out.numel() <= INT32_MAX,
                "convolution exceeds the scalar kernel's int32 indexing range");
    cnrtDim3_t grid = {64, 1, 1};
    auto stream = torch_mlu::getCurMLUStream();
    if (x.scalar_type() == at::kHalf) {
        conv2d_half_kernel<<<grid, cnrtFuncTypeBlock, stream>>>(
            reinterpret_cast<const half *>(input.data_ptr()), reinterpret_cast<const half *>(weights.data_ptr()),
            reinterpret_cast<half *>(out.data_ptr()), out.numel(), x.size(0), x.size(1), weight.size(0),
            x.size(2), x.size(3), out_h, out_w, kernel, stride, padding, dilation, groups);
    } else {
        conv2d_float_kernel<<<grid, cnrtFuncTypeBlock, stream>>>(
            input.data_ptr<float>(), weights.data_ptr<float>(), out.data_ptr<float>(),
            out.numel(), x.size(0), x.size(1), weight.size(0), x.size(2), x.size(3), out_h, out_w,
            kernel, stride, padding, dilation, groups);
    }
    return out;
}
#endif
