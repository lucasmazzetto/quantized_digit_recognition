#include "nn.h"

static int64_t round_shift_right_symmetric_i64(int64_t value, uint32_t shift)
{
    uint64_t abs_value, rounded_abs, rounding_offset;

    // Signed symmetric rounded right shift using integer-only arithmetic
    // This avoids bias for negative numbers compared to always adding +0.5 LSB
    if (shift == 0U) {
        return value;
    }

    // Compute absolute value in unsigned domain to avoid INT64_MIN overflow
    if (value < 0LL) {
        abs_value = (uint64_t)(-(value + 1LL)) + 1ULL;
    } else {
        abs_value = (uint64_t)value;
    }

    rounding_offset = (uint64_t)1U << (shift - 1U);
    // Round in magnitude domain then restore the original sign
    rounded_abs = (abs_value + rounding_offset) >> shift;

    return (value < 0LL) ? -(int64_t)rounded_abs : (int64_t)rounded_abs;
}

static int8_t saturate_to_int8(int64_t value)
{
    // Saturate signed integer to int8 range used by quantized weights and activations
    if (value > (int64_t)INT8_MAX_VALUE) {
        return (int8_t)INT8_MAX_VALUE;
    }
    if (value < -(int64_t)INT8_MAX_VALUE) {
        return -(int8_t)INT8_MAX_VALUE;
    }
    return (int8_t)value;
}

void mat_mult(const int8_t *left_matrix, const int8_t *right_matrix, int *output_matrix,
              const unsigned int batch_size, const unsigned int input_features,
              const unsigned int output_features)
{
    unsigned int n, k, m;
    unsigned int row, col;
    int accumulator;

    for (m = 0; m < output_features; m++) {
        for (n = 0; n < batch_size; n++) {
            // Row-major flattening for one sample row
            row = n * input_features;
            accumulator = 0;
            for (k = 0; k < input_features; k++) {
                // Right matrix is stored as [input_features][output_features]
                col = k * output_features;
                accumulator += left_matrix[row + k] * right_matrix[col + m];
            }
            output_matrix[n * output_features + m] = accumulator;
        }
    }
}

int get_output_dim(int input_dim, int kernel_size, int stride)
{
    // Valid convolution/pooling with no padding
    int output_dim = (input_dim - (kernel_size - 1) - 1) / stride;
    return output_dim + 1;
}

void conv2d(const int8_t *input, const int8_t *weights, int *output, int batch_size,
            int input_channels, int output_channels, int input_height, int input_width,
            int output_height, int output_width, int kernel_height,
            int kernel_width, int stride_height, int stride_width)
{
    // Iterators for batch index and channel index
    int batch_idx, out_channel_idx, in_channel_idx;
    // Iterators over kernel coordinates
    int kernel_row, kernel_col;
    // Iterators over output map coordinates
    int out_row, out_col;

    for (batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        // Base offsets for the current sample in flattened tensors
        int batch_idx_output = batch_idx * output_channels * output_height * output_width;
        int batch_idx_input = batch_idx * input_channels * input_height * input_width;

        for (out_channel_idx = 0; out_channel_idx < output_channels; out_channel_idx++) {
            // Offset of output channel map and filter bank
            int out_channel_idx_output = out_channel_idx * output_height * output_width;
            int out_channel_idx_kernel = out_channel_idx * input_channels * kernel_height
                                         * kernel_width;

            for (out_row = 0; out_row < output_height; out_row++) {
                for (out_col = 0; out_col < output_width; out_col++) {
                    // Flattened output index for (out_row, out_col)
                    int output_idx = out_row * output_width + out_col;
                    // Input origin of the receptive field after stride
                    int input_idx = out_row * stride_height * input_width + out_col * stride_width;
                    int sum = 0;

                    for (in_channel_idx = 0; in_channel_idx < input_channels; in_channel_idx++) {
                        // Offsets for one input channel plane and its kernel slice
                        int in_channel_idx_input = in_channel_idx * input_height * input_width;
                        int in_channel_idx_kernel = in_channel_idx * kernel_height * kernel_width;

                        for (kernel_row = 0; kernel_row < kernel_height; kernel_row++) {
                            for (kernel_col = 0; kernel_col < kernel_width; kernel_col++) {
                                // Local offsets inside kernel and input channel plane
                                int kernel_idx = kernel_row * kernel_width + kernel_col;
                                int kernel_idx_input = kernel_row * input_width + kernel_col;
                                int input_value = (int)input[batch_idx_input + in_channel_idx_input
                                                              + kernel_idx_input + input_idx];
                                int weight_value = (int)weights[out_channel_idx_kernel
                                                                + in_channel_idx_kernel
                                                                + kernel_idx];
                                // Accumulate dot-product in integer domain
                                sum += input_value * weight_value;
                            }
                        }
                    }

                    output[batch_idx_output + out_channel_idx_output + output_idx] = sum;
                }
            }
        }
    }
}

void pooling2d(int *input, int *output, int batch_size, int output_channels, int input_height,
               int input_width, int output_height, int output_width, int kernel_height,
               int kernel_width, int stride_height, int stride_width)
{
    // Iterators for batch and output channel
    int batch_idx, out_channel_idx;
    // Iterators over pooling kernel coordinates
    int kernel_row, kernel_col;
    // Iterators over pooled output coordinates
    int out_row, out_col;

    for (batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        // Base offsets for the current sample in flattened tensors
        int batch_idx_output = batch_idx * output_channels * output_height * output_width;
        int batch_idx_input = batch_idx * output_channels * input_height * input_width;

        for (out_channel_idx = 0; out_channel_idx < output_channels; out_channel_idx++) {
            int out_channel_idx_output = out_channel_idx * output_height * output_width;
            int out_channel_idx_input = out_channel_idx * input_height * input_width;

            for (out_row = 0; out_row < output_height; out_row++) {
                for (out_col = 0; out_col < output_width; out_col++) {
                    int output_idx = out_row * output_width + out_col;
                    // Top-left corner of the pooling window
                    int input_idx = out_row * stride_height * input_width + out_col * stride_width;
                    // Start max with the first value in the current window
                    int max = input[batch_idx_input + out_channel_idx_input + input_idx];

                    for (kernel_row = 0; kernel_row < kernel_height; kernel_row++) {
                        for (kernel_col = 0; kernel_col < kernel_width; kernel_col++) {
                            int kernel_idx = kernel_row * input_width + kernel_col;
                            int value = input[batch_idx_input + out_channel_idx_input
                                              + kernel_idx + input_idx];
                            if (value > max) {
                                max = value;
                            }
                        }
                    }

                    output[batch_idx_output + out_channel_idx_output + output_idx] = max;
                }
            }
        }
    }
}

void relu(int *tensor_inout, const unsigned int size)
{
    unsigned int i;
    for (i = 0; i < size; i++) {
        tensor_inout[i] = MAX(tensor_inout[i], 0);
    }
}

void quantize(const int *tensor_inout, int8_t *tensor_q, const int scale_factor,
              const unsigned int size)
{
    unsigned int i;

    for (i = 0; i < size; i++) {
        // Input and scale are Q16 so product is Q32 before shifting back
        int64_t scaled_q32 = (int64_t)tensor_inout[i] * (int64_t)scale_factor;
        int64_t quantized_value = round_shift_right_symmetric_i64(
            scaled_q32, (uint32_t)(2 * FXP_VALUE));

        // Saturate before narrowing to int8 to prevent wraparound
        tensor_q[i] = saturate_to_int8(quantized_value);
    }
}

void dequantize_per_row(int *matrix_inout, const int *scale_factor_w_inv,
                        const int scale_factor_x_inv, const unsigned int batch_size,
                        const unsigned int output_features)
{
    unsigned int k, n;
    int64_t out_value_q32;

    for (n = 0; n < batch_size; n++) {
        for (k = 0; k < output_features; k++) {
            // scale_factor_w_inv and scale_factor_x_inv are Q16 values
            // Their product is Q32 and after one shift by 16 the result is Q16
            int64_t scaled_value_q16;
            // One inverse scale per output feature for linear layers
            out_value_q32 = (int64_t)scale_factor_w_inv[k] * (int64_t)scale_factor_x_inv;
            scaled_value_q16 = round_shift_right_symmetric_i64(
                out_value_q32 * (int64_t)matrix_inout[n * output_features + k],
                (uint32_t)FXP_VALUE);
            matrix_inout[n * output_features + k] = (int)scaled_value_q16;
        }
    }
}

void dequantize_per_channel(int *tensor_inout, const int *weight_scale_inv,
                            const int input_scale_inv, const unsigned int batch_size,
                            const unsigned int output_channels,
                            const unsigned int input_features)
{
    unsigned int k, n, c;
    int64_t out_value_q32;

    for (n = 0; n < batch_size; n++) {
        for (c = 0; c < output_channels; c++) {
            // This scale is constant across all features of the same output channel
            out_value_q32 = (int64_t)weight_scale_inv[c] * (int64_t)input_scale_inv;

            for (k = 0; k < input_features; k++) {
                unsigned int tensor_idx;
                int64_t scaled_value_q32;
                int64_t scaled_value_q16;

                // Flatten (n, c, k) into one index for the contiguous tensor buffer
                tensor_idx = n * output_channels * input_features + c * input_features + k;

                scaled_value_q32 = out_value_q32 * (int64_t)tensor_inout[tensor_idx];

                scaled_value_q16 = round_shift_right_symmetric_i64(
                    scaled_value_q32, (uint32_t)FXP_VALUE);

                tensor_inout[tensor_idx] = (int)scaled_value_q16;
            }
        }
    }
}

void argmax_over_cols(const int *matrix_in, unsigned int *indices,
                      const unsigned int batch_size, const unsigned int output_features)
{
    unsigned int n, m, max_idx;
    int row_max, value;

    for (n = 0; n < batch_size; n++) {
        // Initialize with the first element then scan the whole row
        row_max = matrix_in[n * output_features];
        max_idx = 0;

        for (m = 0; m < output_features; m++) {
            value = matrix_in[n * output_features + m];

            if (value > row_max) {
                row_max = value;
                max_idx = m;
            }
        }

        indices[n] = max_idx;
    }
}

void linear_layer(const int *input, const int8_t *weights, int *output, const int input_scale,
                  const int *weight_scale_inv, const int input_scale_inv,
                  const unsigned int batch_size, const unsigned int input_features,
                  const unsigned int output_features, const unsigned int apply_relu)
{
    int8_t input_quantized[batch_size * input_features];

    // Quantize Q16 activations to int8 domain expected by integer GEMM
    quantize(input, input_quantized, input_scale, batch_size * input_features);
    
    // Integer matmul gives accumulators in int domain
    mat_mult(input_quantized, weights, output, batch_size, input_features, output_features);

    // Bring accumulators back to Q16 for the next layer
    dequantize_per_row(output, weight_scale_inv, input_scale_inv, batch_size, output_features);

    if (apply_relu) {
        relu(output, batch_size * output_features);
    }
}

void conv2d_layer(const int *input, const int8_t *weights, int *output, const int input_scale,
                  const int *weight_scale_inv, const int input_scale_inv,
                  const unsigned int batch_size, const unsigned int input_channels,
                  const unsigned int output_channels, const int input_height,
                  const int input_width, const int output_height, const int output_width,
                  const int kernel_height, const int kernel_width, const int stride_height,
                  const int stride_width)
{
    int8_t input_quantized[batch_size * input_channels * input_height * input_width];

    // Quantize Q16 activations to int8 domain expected by convolution
    quantize(input, input_quantized, input_scale,
             batch_size * input_channels * input_height * input_width);

    // Integer convolution on quantized activations and weights
    conv2d(input_quantized, weights, output, batch_size, input_channels, output_channels,
           input_height, input_width, output_height, output_width,
           kernel_height, kernel_width, stride_height, stride_width);

    // Bring accumulators back to Q16 then apply non-linearity
    dequantize_per_channel(output, weight_scale_inv, input_scale_inv,
                           batch_size, output_channels, output_height * output_width);

    relu(output, batch_size * output_channels * output_height * output_width);
}
