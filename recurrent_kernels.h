#pragma once
#include "bangc_common.h"
#include <vector>

template <typename T>
__mlu_entry__ void recurrent_affine_kernel(const T *input, const T *hidden,
                                           const T *weight_ih, const T *weight_hh,
                                           const T *bias_ih, const T *bias_hh, float *gates,
                                           int64_t batch, int64_t sequence, int64_t step,
                                           int64_t input_size, int64_t hidden_size, int gate_count) {
    int64_t width = gate_count * hidden_size;
    for (int64_t i = taskId; i < batch * width; i += taskDim) {
        int64_t sample = i / width;
        int64_t gate = i % width;
        float from_input = bias_ih == nullptr ? 0.0f : static_cast<float>(bias_ih[gate]);
        float from_hidden = bias_hh == nullptr ? 0.0f : static_cast<float>(bias_hh[gate]);
        for (int64_t k = 0; k < input_size; ++k) {
            from_input += static_cast<float>(input[(sample * sequence + step) * input_size + k])
                        * static_cast<float>(weight_ih[gate * input_size + k]);
        }
        for (int64_t k = 0; k < hidden_size; ++k) {
            from_hidden += static_cast<float>(hidden[sample * hidden_size + k])
                         * static_cast<float>(weight_hh[gate * hidden_size + k]);
        }
        gates[sample * 2 * width + gate] = from_input;
        gates[sample * 2 * width + width + gate] = from_hidden;
    }
}

__mlu_func__ inline float recurrent_sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

template <typename T>
__mlu_entry__ void recurrent_update_kernel(const float *gates, T *hidden, T *cell, T *out,
                                           int64_t batch, int64_t sequence, int64_t step,
                                           int64_t width, bool lstm) {
    int gate_count = lstm ? 4 : 3;
    for (int64_t i = taskId; i < batch * width; i += taskDim) {
        int64_t sample = i / width;
        int64_t column = i % width;
        const float *from_input = gates + sample * 2 * gate_count * width + column;
        const float *from_hidden = from_input + gate_count * width;
        float value;
        if (lstm) {
            float input_gate = recurrent_sigmoid(from_input[0] + from_hidden[0]);
            float forget_gate = recurrent_sigmoid(from_input[width] + from_hidden[width]);
            float candidate = tanhf(from_input[2 * width] + from_hidden[2 * width]);
            float output_gate = recurrent_sigmoid(from_input[3 * width] + from_hidden[3 * width]);
            float next_cell = forget_gate * static_cast<float>(cell[i]) + input_gate * candidate;
            cell[i] = static_cast<T>(next_cell);
            value = output_gate * tanhf(next_cell);
        } else {
            float reset = recurrent_sigmoid(from_input[0] + from_hidden[0]);
            float update = recurrent_sigmoid(from_input[width] + from_hidden[width]);
            float candidate = tanhf(from_input[2 * width] + reset * from_hidden[2 * width]);
            value = (1.0f - update) * candidate + update * static_cast<float>(hidden[i]);
        }
        hidden[i] = static_cast<T>(value);
        out[(sample * sequence + step) * width + column] = static_cast<T>(value);
    }
}

#ifndef BANG_DEVICE_ONLY
inline torch::Tensor run_recurrent(torch::Tensor x, std::vector<torch::Tensor> parameters,
                                   int input_size, int hidden_size, int num_layers, bool lstm) {
    TORCH_CHECK(x.dim() == 3 && x.size(2) == input_size && x.size(1) > 0,
                "expected input [batch, nonempty_sequence, input_size]");
    TORCH_CHECK(input_size > 0 && hidden_size > 0 && num_layers > 0, "invalid recurrent dimensions");
    bool has_bias = parameters.size() == static_cast<size_t>(4 * num_layers);
    TORCH_CHECK(has_bias || parameters.size() == static_cast<size_t>(2 * num_layers),
                "provide [weight_ih, weight_hh, bias_ih, bias_hh] per layer, or two weights per bias-free layer");
    int gate_count = lstm ? 4 : 3;
    int parameter_stride = has_bias ? 4 : 2;
    auto input = x.contiguous();
    cnrtDim3_t grid = {64, 1, 1};
    auto stream = torch_mlu::getCurMLUStream();
    for (int layer = 0; layer < num_layers; ++layer) {
        int64_t features = layer == 0 ? input_size : hidden_size;
        auto weight_ih = parameters[layer * parameter_stride].contiguous();
        auto weight_hh = parameters[layer * parameter_stride + 1].contiguous();
        auto bias_ih = has_bias ? parameters[layer * parameter_stride + 2].contiguous() : torch::Tensor();
        auto bias_hh = has_bias ? parameters[layer * parameter_stride + 3].contiguous() : torch::Tensor();
        bangc::check_like(weight_ih, x);
        bangc::check_like(weight_hh, x);
        TORCH_CHECK(weight_ih.dim() == 2 && weight_ih.size(0) == gate_count * hidden_size &&
                    weight_ih.size(1) == features, "weight_ih shape mismatch");
        TORCH_CHECK(weight_hh.dim() == 2 && weight_hh.size(0) == gate_count * hidden_size &&
                    weight_hh.size(1) == hidden_size, "weight_hh shape mismatch");
        if (has_bias) {
            bangc::check_like(bias_ih, x);
            bangc::check_like(bias_hh, x);
            TORCH_CHECK(bias_ih.numel() == gate_count * hidden_size && bias_hh.numel() == gate_count * hidden_size,
                        "recurrent bias shape mismatch");
        }
        auto hidden = torch::zeros({x.size(0), hidden_size}, x.options());
        auto cell = torch::zeros({x.size(0), hidden_size}, x.options());
        auto gates = torch::empty({x.size(0), 2 * gate_count * hidden_size}, x.options().dtype(torch::kFloat));
        auto output = torch::empty({x.size(0), x.size(1), hidden_size}, x.options());
        for (int64_t step = 0; step < x.size(1); ++step) {
            BANGC_DISPATCH(x,
                recurrent_affine_kernel<scalar_t><<<grid, cnrtFuncTypeBlock, stream>>>(
                    reinterpret_cast<const scalar_t *>(input.data_ptr()), reinterpret_cast<const scalar_t *>(hidden.data_ptr()),
                    reinterpret_cast<const scalar_t *>(weight_ih.data_ptr()), reinterpret_cast<const scalar_t *>(weight_hh.data_ptr()),
                    has_bias ? reinterpret_cast<const scalar_t *>(bias_ih.data_ptr()) : nullptr,
                    has_bias ? reinterpret_cast<const scalar_t *>(bias_hh.data_ptr()) : nullptr,
                    gates.data_ptr<float>(), x.size(0), x.size(1), step, features, hidden_size, gate_count);
                recurrent_update_kernel<scalar_t><<<grid, cnrtFuncTypeBlock, stream>>>(
                    gates.data_ptr<float>(), reinterpret_cast<scalar_t *>(hidden.data_ptr()),
                    reinterpret_cast<scalar_t *>(cell.data_ptr()), reinterpret_cast<scalar_t *>(output.data_ptr()),
                    x.size(0), x.size(1), step, hidden_size, lstm);
            );
        }
        input = output;
    }
    return input;
}
#endif
