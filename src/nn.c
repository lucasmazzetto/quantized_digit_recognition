#include "nn.h"

/**
 * @brief Applies symmetric rounded right-shift on a signed 64-bit integer.
 *
 * Rounding is performed in the magnitude domain and the original sign is restored afterwards,
 * which keeps positive and negative values symmetric.
 *
 * @param value Signed integer value to be shifted.
 * @param shift Number of bits to shift right.
 * @return Rounded and shifted signed value.
 */
static int64_t round_shift_right_symmetric(int64_t value, uint32_t shift)
{
    // Signed symmetric rounded right shift using integer-only arithmetic
    // This avoids bias for negative numbers compared to always adding +0.5 LSB
    if (shift == 0U) {
        return value;
    }

    // Compute absolute value in unsigned domain to avoid INT64_MIN overflow
    const uint64_t abs_value = (value < 0LL) ? (uint64_t)(-(value + 1LL)) + 1ULL : (uint64_t)value;
    const uint64_t rounding_offset = (uint64_t)1U << (shift - 1U);

    // Round in magnitude domain then restore the original sign
    const uint64_t rounded_abs = (abs_value + rounding_offset) >> shift;

    return (value < 0LL) ? -(int64_t)rounded_abs : (int64_t)rounded_abs;
}

/**
 * @brief Clamps a signed 64-bit value to the representable int8 range.
 *
 * The clamp range is symmetric around zero and matches the quantized domain used in this module.
 *
 * @param value Input value in wider integer format.
 * @return Saturated int8 value.
 */
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

/**
 * @brief Computes integer matrix multiplication for one batched linear layer.
 *
 * This helper computes: output_matrix = left_matrix x right_matrix where `left_matrix` is
 * [batch_size, input_features] and `right_matrix` is [input_features, output_features], both
 * flattened in row-major order.
 *
 * @param left_matrix Quantized activations matrix.
 * @param right_matrix Quantized weights matrix.
 * @param output_matrix Integer accumulation output matrix.
 * @param batch_size Number of rows in the left matrix.
 * @param input_features Shared dimension between left and right matrices.
 * @param output_features Number of columns in the right matrix.
 */
static void mat_mult(const int8_t *left_matrix, const int8_t *right_matrix, int *output_matrix,
                     const unsigned int batch_size, const unsigned int input_features,
                     const unsigned int output_features)
{
    unsigned int out_feat_idx, batch_idx, in_feat_idx;
    int accumulator;

    for (out_feat_idx = 0; out_feat_idx < output_features; out_feat_idx++) {
        for (batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            // Row-major flattening for one sample row
            const unsigned int in_row_base = batch_idx * input_features;
            accumulator = 0;

            for (in_feat_idx = 0; in_feat_idx < input_features; in_feat_idx++) {
                // Right matrix is stored as [input_features][output_features]
                const unsigned int weight_row_base = in_feat_idx * output_features;
                accumulator += left_matrix[in_row_base + in_feat_idx] *
                               right_matrix[weight_row_base + out_feat_idx];
            }

            output_matrix[batch_idx * output_features + out_feat_idx] = accumulator;
        }
    }
}

/**
 * @brief Performs low-level integer 2D convolution over flattened tensors.
 *
 * All tensors are expected in contiguous row-major layout: input [N, C_in, H_in, W_in], weights
 * [C_out, C_in, K_h, K_w], output [N, C_out, H_out, W_out].
 *
 * @param input Quantized input tensor.
 * @param weights Quantized convolution filters.
 * @param output Output accumulator tensor.
 * @param batch_size Number of input samples.
 * @param input_channels Number of input channels.
 * @param output_channels Number of output channels.
 * @param input_height Input tensor height.
 * @param input_width Input tensor width.
 * @param output_height Output tensor height.
 * @param output_width Output tensor width.
 * @param kernel_height Filter kernel height.
 * @param kernel_width Filter kernel width.
 * @param stride_height Convolution stride in height dimension.
 * @param stride_width Convolution stride in width dimension.
 */
static void conv2d(const int8_t *input, const int8_t *weights, int *output, int batch_size,
                   int input_channels, int output_channels, int input_height, int input_width,
                   int output_height, int output_width, int kernel_height, int kernel_width,
                   int stride_height, int stride_width)
{
    const int input_channel_area = input_height * input_width;
    const int output_channel_area = output_height * output_width;
    const int kernel_size = kernel_height * kernel_width;

    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        // Base offsets for the current sample in flattened tensors
        const int output_batch_offset = batch_idx * output_channels * output_channel_area;
        const int input_batch_offset = batch_idx * input_channels * input_channel_area;

        for (int out_chan_idx = 0; out_chan_idx < output_channels; out_chan_idx++) {
            // Offsets for the current output channel and its filter bank
            const int output_channel_offset = out_chan_idx * output_channel_area;
            const int kernel_output_channel_offset = out_chan_idx * input_channels * kernel_size;

            for (int output_row = 0; output_row < output_height; output_row++) {
                for (int output_col = 0; output_col < output_width; output_col++) {
                    const int output_channel_position = output_row * output_width + output_col;

                    // Input origin of the receptive field after stride
                    const int input_window_start_offset =
                        output_row * stride_height * input_width + output_col * stride_width;

                    int sum = 0;

                    for (int in_chan_idx = 0; in_chan_idx < input_channels; in_chan_idx++) {
                        // Offsets for the current input channel and its kernel slice
                        const int input_channel_offset = in_chan_idx * input_channel_area;
                        const int kernel_input_channel_offset = in_chan_idx * kernel_size;

                        for (int kernel_row = 0; kernel_row < kernel_height; kernel_row++) {
                            for (int kernel_col = 0; kernel_col < kernel_width; kernel_col++) {
                                const int kernel_position = kernel_row * kernel_width + kernel_col;

                                const int input_kernel_offset =
                                    kernel_row * input_width + kernel_col;

                                const int input_index = input_batch_offset + input_channel_offset +
                                                        input_window_start_offset +
                                                        input_kernel_offset;

                                const int weight_index = kernel_output_channel_offset +
                                                         kernel_input_channel_offset +
                                                         kernel_position;

                                const int input_value = (int)input[input_index];
                                const int weight_value = (int)weights[weight_index];

                                // Accumulate dot-product in integer domain
                                sum += input_value * weight_value;
                            }
                        }
                    }

                    const int output_index =
                        output_batch_offset + output_channel_offset + output_channel_position;

                    output[output_index] = sum;
                }
            }
        }
    }
}

void pooling2d(const int *input, int *output, int batch_size, int channels, int input_height,
               int input_width, int output_height, int output_width, int kernel_height,
               int kernel_width, int stride_height, int stride_width)
{
    const int input_channel_area = input_height * input_width;
    const int output_channel_area = output_height * output_width;

    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        // Base offsets for the current sample in flattened tensors
        const int output_batch_offset = batch_idx * channels * output_channel_area;
        const int input_batch_offset = batch_idx * channels * input_channel_area;

        for (int channel_index = 0; channel_index < channels; channel_index++) {
            const int output_channel_offset = channel_index * output_channel_area;
            const int input_channel_offset = channel_index * input_channel_area;

            for (int output_row = 0; output_row < output_height; output_row++) {
                for (int output_col = 0; output_col < output_width; output_col++) {
                    const int output_channel_position = output_row * output_width + output_col;

                    // Top-left corner of the pooling window
                    const int input_window_start_offset =
                        output_row * stride_height * input_width + output_col * stride_width;

                    // Start max with the first value in the current window
                    const int first_input_index =
                        input_batch_offset + input_channel_offset + input_window_start_offset;

                    int max_value = input[first_input_index];

                    for (int kernel_row = 0; kernel_row < kernel_height; kernel_row++) {
                        for (int kernel_col = 0; kernel_col < kernel_width; kernel_col++) {
                            const int input_kernel_offset = kernel_row * input_width + kernel_col;

                            const int input_index = input_batch_offset + input_channel_offset +
                                                    input_window_start_offset + input_kernel_offset;

                            const int value = input[input_index];

                            if (value > max_value) {
                                max_value = value;
                            }
                        }
                    }

                    const int output_index =
                        output_batch_offset + output_channel_offset + output_channel_position;
                        
                    output[output_index] = max_value;
                }
            }
        }
    }
}

/**
 * @brief Applies ReLU in-place on a flattened integer tensor.
 *
 * @param data Input/output tensor buffer.
 * @param size Number of elements in the flattened tensor.
 */
static void relu(int *data, const unsigned int size)
{
    unsigned int i;

    for (i = 0; i < size; i++) {
        data[i] = MAX(data[i], 0);
    }
}

/**
 * @brief Quantizes Q(FRAC_BITS) activations into int8 values with symmetric rounding.
 *
 * @param input Input tensor in Q(FRAC_BITS) domain.
 * @param output Output tensor in int8 quantized domain.
 * @param input_scale Q(FRAC_BITS) scale factor used for quantization.
 * @param size Number of flattened tensor elements.
 */
static void quantize(const int *input, int8_t *output, const int input_scale,
                     const unsigned int size)
{
    unsigned int i;

    for (i = 0; i < size; i++) {
        // Input and scale are Q(FRAC_BITS) so product is Q(2 * FRAC_BITS)
        const int64_t scaled_product = (int64_t)input[i] * (int64_t)input_scale;

        const int64_t quantized_value =
            round_shift_right_symmetric(scaled_product, (uint32_t)(2 * FRAC_BITS));

        // Saturate before narrowing to int8 to prevent wraparound
        output[i] = saturate_to_int8(quantized_value);
    }
}

/**
 * @brief Dequantizes linear-layer accumulators using per-output-feature scales.
 *
 * This helper is used after fully-connected matmul, where one inverse weight scale is applied per
 * output feature (column).
 *
 * @param data Matrix to dequantize in-place.
 * @param weight_scale_inv Inverse weight scales, one per output feature.
 * @param input_scale_inv Inverse input scale.
 * @param batch_size Number of rows in the output matrix.
 * @param output_features Number of output features (columns).
 */
static void dequantize_per_col(int *data, const int *weight_scale_inv, const int input_scale_inv,
                               const unsigned int batch_size, const unsigned int output_features)
{
    unsigned int batch_idx, out_feat_idx;

    for (batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        for (out_feat_idx = 0; out_feat_idx < output_features; out_feat_idx++) {
            // Build the per-output inverse scale and reproject the accumulator back to Q
            const int64_t combined_inv_scale =
                (int64_t)weight_scale_inv[out_feat_idx] * (int64_t)input_scale_inv;

            const unsigned int out_idx = batch_idx * output_features + out_feat_idx;

            const int64_t dequantized_value = round_shift_right_symmetric(
                combined_inv_scale * (int64_t)data[out_idx], (uint32_t)FRAC_BITS);

            data[out_idx] = (int)dequantized_value;
        }
    }
}

/**
 * @brief Dequantizes convolution accumulators using per-output-channel scales.
 *
 * @param data Tensor to dequantize in-place, flattened as [N, C_out, ...].
 * @param weight_scale_inv Inverse weight scales, one per output channel.
 * @param input_scale_inv Inverse input activation scale.
 * @param batch_size Number of input samples.
 * @param output_channels Number of output channels.
 * @param output_size Flattened output spatial size per channel.
 */
static void dequantize_per_channel(int *data, const int *weight_scale_inv,
                                   const int input_scale_inv, const unsigned int batch_size,
                                   const unsigned int output_channels,
                                   const unsigned int output_size)
{
    unsigned int batch_idx, out_chan_idx, out_pos_idx;

    for (batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        for (out_chan_idx = 0; out_chan_idx < output_channels; out_chan_idx++) {
            // This scale is constant across all features of the same output channel
            const int64_t combined_inv_scale =
                (int64_t)weight_scale_inv[out_chan_idx] * (int64_t)input_scale_inv;

            for (out_pos_idx = 0; out_pos_idx < output_size; out_pos_idx++) {
                // Flatten (n, c, k) into one index for the contiguous tensor buffer
                const unsigned int out_idx = batch_idx * output_channels * output_size +
                                             out_chan_idx * output_size + out_pos_idx;

                const int64_t scaled_accumulator = combined_inv_scale * (int64_t)data[out_idx];

                const int64_t dequantized_value =
                    round_shift_right_symmetric(scaled_accumulator, (uint32_t)FRAC_BITS);

                data[out_idx] = (int)dequantized_value;
            }
        }
    }
}

void argmax_per_row(const int *input, unsigned int *indices, const unsigned int batch_size,
                    const unsigned int output_features)
{
    unsigned int batch_idx, out_feat_idx, max_out_feat_idx;
    int row_max;

    for (batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        // Initialize with the first element then scan the whole row
        const unsigned int row_base = batch_idx * output_features;
        row_max = input[row_base];
        max_out_feat_idx = 0;

        for (out_feat_idx = 0; out_feat_idx < output_features; out_feat_idx++) {
            const int value = input[row_base + out_feat_idx];

            if (value > row_max) {
                row_max = value;
                max_out_feat_idx = out_feat_idx;
            }
        }

        indices[batch_idx] = max_out_feat_idx;
    }
}

void linear_layer(const int *input, const int8_t *weights, int *output, const int input_scale,
                  const int *weight_scale_inv, const int input_scale_inv,
                  const unsigned int batch_size, const unsigned int input_features,
                  const unsigned int output_features, const unsigned int apply_relu)
{
    int8_t input_quantized[batch_size * input_features];

    // Quantize Q(FRAC_BITS) activations to int8 domain expected by integer GEMM
    quantize(input, input_quantized, input_scale, batch_size * input_features);

    // Integer matmul gives accumulators in int domain
    mat_mult(input_quantized, weights, output, batch_size, input_features, output_features);

    // Bring accumulators back to Q(FRAC_BITS) for the next layer
    dequantize_per_col(output, weight_scale_inv, input_scale_inv, batch_size, output_features);

    if (apply_relu) {
        relu(output, batch_size * output_features);
    }
}

void conv2d_layer(const int *input, const int8_t *weights, int *output, const int input_scale,
                  const int *weight_scale_inv, const int input_scale_inv,
                  const unsigned int batch_size, const unsigned int input_channels,
                  const unsigned int output_channels, const int input_height, const int input_width,
                  const int output_height, const int output_width, const int kernel_height,
                  const int kernel_width, const int stride_height, const int stride_width)
{
    int8_t input_quantized[batch_size * input_channels * input_height * input_width];

    // Quantize Q(FRAC_BITS) activations to int8 domain expected by convolution
    quantize(input, input_quantized, input_scale,
             batch_size * input_channels * input_height * input_width);

    // Integer convolution on quantized activations and weights
    conv2d(input_quantized, weights, output, batch_size, input_channels, output_channels,
           input_height, input_width, output_height, output_width, kernel_height, kernel_width,
           stride_height, stride_width);

    // Bring accumulators back to Q(FRAC_BITS) then apply non-linearity
    dequantize_per_channel(output, weight_scale_inv, input_scale_inv, batch_size, output_channels,
                           output_height * output_width);

    relu(output, batch_size * output_channels * output_height * output_width);
}
