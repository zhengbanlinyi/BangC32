#pragma once
#include "bangc_common.h"

template <typename T>
__mlu_func__ float attention_dot(const T *q, const T *k, int64_t width, float scale) {
    float dot = 0.0f;
    for (int64_t i = 0; i < width; ++i) dot += static_cast<float>(q[i]) * static_cast<float>(k[i]);
    return dot * scale;
}

template <typename T>
__mlu_entry__ void attention_kernel(const T *q, const T *k, const T *v, const T *bias, T *out,
                                    int64_t matrices, int64_t queries, int64_t keys,
                                    int64_t width, int64_t value_width,
                                    float scale, int64_t window, bool causal) {
    int64_t count = matrices * queries * value_width;
    for (int64_t i = taskId; i < count; i += taskDim) {
        int64_t matrix = i / (queries * value_width);
        int64_t row = i / value_width % queries;
        int64_t value_col = i % value_width;
        int64_t begin = window >= 0 && row > window ? row - window : 0;
        int64_t end = keys;
        if (window >= 0 && row + window + 1 < end) end = row + window + 1;
        if (causal && row + 1 < end) end = row + 1;
        const T *query = q + (matrix * queries + row) * width;
        float maximum = -INFINITY;
        for (int64_t col = begin; col < end; ++col) {
            float score = attention_dot(query, k + (matrix * keys + col) * width, width, scale);
            if (bias != nullptr) score += static_cast<float>(bias[(matrix * queries + row) * keys + col]);
            if (isnan(score)) { maximum = score; break; }
            if (score > maximum) maximum = score;
        }
        float denominator = 0.0f;
        float numerator = 0.0f;
        for (int64_t col = begin; col < end; ++col) {
            float score = attention_dot(query, k + (matrix * keys + col) * width, width, scale);
            if (bias != nullptr) score += static_cast<float>(bias[(matrix * queries + row) * keys + col]);
            float probability = expf(score - maximum);
            denominator += probability;
            numerator += probability * static_cast<float>(v[(matrix * keys + col) * value_width + value_col]);
        }
        out[i] = static_cast<T>(numerator / denominator);
    }
}

#ifndef BANG_DEVICE_ONLY
inline torch::Tensor run_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                   torch::Tensor bias, double scale, int64_t window, bool causal) {
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "expected [batch, heads, sequence, width]");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(1) == k.size(1) && q.size(3) == k.size(3),
                "query and key dimensions do not match");
    TORCH_CHECK(v.size(0) == k.size(0) && v.size(1) == k.size(1) && v.size(2) == k.size(2),
                "key and value dimensions do not match");
    TORCH_CHECK(q.size(3) > 0 && k.size(2) > 0, "attention needs nonempty key and feature dimensions");
    bangc::check_like(k, q);
    bangc::check_like(v, q);
    auto query = q.contiguous();
    auto key = k.contiguous();
    auto value = v.contiguous();
    torch::Tensor expanded_bias;
    if (bias.defined()) {
        bangc::check_like(bias, q);
        expanded_bias = bias.expand({q.size(0), q.size(1), q.size(2), k.size(2)}).contiguous();
    }
    auto out = torch::empty({q.size(0), q.size(1), q.size(2), v.size(3)}, q.options());
    cnrtDim3_t grid = {64, 1, 1};
    auto stream = torch_mlu::getCurMLUStream();
    BANGC_DISPATCH(q,
        attention_kernel<scalar_t><<<grid, cnrtFuncTypeBlock, stream>>>(
            reinterpret_cast<const scalar_t *>(query.data_ptr()), reinterpret_cast<const scalar_t *>(key.data_ptr()),
            reinterpret_cast<const scalar_t *>(value.data_ptr()),
            bias.defined() ? reinterpret_cast<const scalar_t *>(expanded_bias.data_ptr()) : nullptr,
            reinterpret_cast<scalar_t *>(out.data_ptr()), q.size(0) * q.size(1), q.size(2), k.size(2),
            q.size(3), v.size(3), static_cast<float>(scale), window, causal);
    );
    return out;
}
#endif
