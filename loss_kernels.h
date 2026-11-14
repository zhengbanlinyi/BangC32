#pragma once
#include "bangc_common.h"

template <typename T>
__mlu_entry__ void pointwise_loss_kernel(const T *prediction, const T *target, float *partials,
                                        int64_t count, int parts, bool huber, float delta) {
    for (int part = taskId; part < parts; part += taskDim) {
        float sum = 0.0f;
        for (int64_t i = part; i < count; i += parts) {
            float error = static_cast<float>(prediction[i]) - static_cast<float>(target[i]);
            float magnitude = fabsf(error);
            sum += huber ? (magnitude < delta ? 0.5f * error * error : delta * (magnitude - 0.5f * delta))
                         : error * error;
        }
        partials[part] = sum;
    }
}

template <typename T>
__mlu_entry__ void loss_mean_kernel(const float *partials, T *out, int64_t count, int parts) {
    if (taskId != 0) return;
    float sum = 0.0f;
    for (int i = 0; i < parts; ++i) sum += partials[i];
    out[0] = static_cast<T>(count == 0 ? NAN : sum / count);
}

#ifndef BANG_DEVICE_ONLY
inline torch::Tensor run_pointwise_loss(torch::Tensor predictions, torch::Tensor targets, bool huber, double delta) {
    bangc::check_like(targets, predictions);
    TORCH_CHECK(!huber || delta > 0.0, "delta must be positive");
    auto expanded = torch::broadcast_tensors({predictions, targets});
    auto p = expanded[0].contiguous();
    auto t = expanded[1].contiguous();
    constexpr int parts = 64;
    auto partials = torch::empty({parts}, predictions.options().dtype(torch::kFloat));
    auto out = torch::empty({}, predictions.options());
    cnrtDim3_t grid = {parts, 1, 1};
    cnrtDim3_t one = {1, 1, 1};
    auto stream = torch_mlu::getCurMLUStream();
    BANGC_DISPATCH(predictions,
        pointwise_loss_kernel<scalar_t><<<grid, cnrtFuncTypeBlock, stream>>>(
            reinterpret_cast<const scalar_t *>(p.data_ptr()), reinterpret_cast<const scalar_t *>(t.data_ptr()),
            partials.data_ptr<float>(), p.numel(), parts, huber, static_cast<float>(delta));
        loss_mean_kernel<scalar_t><<<one, cnrtFuncTypeBlock, stream>>>(partials.data_ptr<float>(),
            reinterpret_cast<scalar_t *>(out.data_ptr()), p.numel(), parts);
    );
    return out;
}
#endif
