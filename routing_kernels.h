#pragma once
#include "bangc_common.h"

#ifndef BANG_DEVICE_ONLY
inline std::pair<torch::Tensor, torch::Tensor> prepare_routes(torch::Tensor indices, torch::Tensor bins,
                                                             torch::Device device, int64_t route_slots) {
    TORCH_CHECK(indices.dim() == 1 && bins.dim() == 1, "routing metadata must be one-dimensional");
    TORCH_CHECK((indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong) &&
                (bins.scalar_type() == at::kInt || bins.scalar_type() == at::kLong), "routing metadata must be integral");
    auto routes = indices.to(device, torch::kLong).contiguous();
    auto boundaries = bins.to(device, torch::kLong).contiguous();
    bangc::check_indices(routes, route_slots);
    auto host = boundaries.to(torch::kCPU).contiguous();
    int64_t previous = 0;
    for (int64_t i = 0; i < host.numel(); ++i) {
        int64_t end = host.data_ptr<int64_t>()[i];
        TORCH_CHECK(end >= previous && end <= indices.numel(), "bins must be nondecreasing route counts");
        previous = end;
    }
    TORCH_CHECK(previous == indices.numel(), "last bin must equal the route count");
    return {routes, boundaries};
}
#endif

template <typename T>
__mlu_entry__ void routed_gather_kernel(const T *x, const int64_t *indices, const T *weights, T *out,
                                       int64_t routes, int64_t width, int top_k) {
    for (int64_t i = taskId; i < routes * width; i += taskDim) {
        int64_t route = indices[i / width];
        out[i] = static_cast<T>(static_cast<float>(x[route / top_k * width + i % width])
                                * static_cast<float>(weights[route]));
    }
}

template <typename T>
__mlu_entry__ void binned_gather_kernel(const T *x, const int64_t *indices, const T *weights,
                                       const int64_t *bins, T *out, int64_t experts,
                                       int64_t capacity, int64_t width, int top_k) {
    for (int64_t i = taskId; i < experts * capacity * width; i += taskDim) {
        int64_t expert = i / (capacity * width);
        int64_t entry = i / width % capacity;
        int64_t start = expert == 0 ? 0 : bins[expert - 1];
        if (start + entry < bins[expert]) {
            int64_t route = indices[start + entry];
            out[i] = static_cast<T>(static_cast<float>(x[route / top_k * width + i % width])
                                    * static_cast<float>(weights[route]));
        } else {
            out[i] = static_cast<T>(0.0f);
        }
    }
}

template <typename T>
__mlu_entry__ void binned_scatter_kernel(const T *x, const int64_t *indices, const T *weights,
                                        const int64_t *bins, float *out, int64_t tokens,
                                        int64_t experts, int64_t capacity, int64_t width, int top_k) {
    for (int64_t i = taskId; i < tokens * width; i += taskDim) {
        float sum = 0.0f;
        int64_t start = 0;
        for (int64_t expert = 0; expert < experts; ++expert) {
            int64_t end = bins[expert];
            int64_t accepted = end - start < capacity ? end - start : capacity;
            for (int64_t entry = 0; entry < accepted; ++entry) {
                int64_t route = indices[start + entry];
                if (route / top_k == i / width) {
                    sum += static_cast<float>(x[(expert * capacity + entry) * width + i % width])
                         * static_cast<float>(weights[route]);
                }
            }
            start = end;
        }
        out[i] = sum;
    }
}
