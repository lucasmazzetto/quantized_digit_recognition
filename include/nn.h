#ifndef NN_H
#define NN_H

#include <stdint.h>

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))

#define NUM_BITS 8
#define INT8_MAX_VALUE 127
#define FXP_VALUE 16
#define ROUND_CONST (1 << (FXP_VALUE - 1)) // = 0.5 before right shifting to improve rounding

/**
 * @brief Performs 2D max pooling.
 *
 * @param input The input tensor.
 * @param output The output tensor.
 * @param batch_size The number of samples.
 * @param channels The number of channels (pooling preserves channel count).
 * @param input_height The input height.
 * @param input_width The input width.
 * @param output_height The output height.
 * @param output_width The output width.
 * @param kernel_height The kernel height.
 * @param kernel_width The kernel width.
 * @param stride_height The height stride.
 * @param stride_width The width stride.
 */
void pooling2d(int *input, int *output, int batch_size, int channels, int input_height,
               int input_width, int output_height, int output_width, int kernel_height,
               int kernel_width, int stride_height, int stride_width);

/**
 * @brief Computes argmax along columns for each row of a flattened matrix.
 *
 * Interprets `matrix_in` as row-major [batch_size, output_features]. For each
 * row `n`, stores in `indices[n]` the column index `m` with the maximum value.
 * If multiple columns share the maximum, the first one is kept.
 *
 * @param matrix_in Input matrix flattened in row-major order.
 * @param indices Output array of length `batch_size` with argmax column indices.
 * @param batch_size Number of matrix rows.
 * @param output_features Number of columns per row. Must be greater than zero.
 */
void argmax_per_row(const int *matrix_in, unsigned int *indices,
                    const unsigned int batch_size, const unsigned int output_features);

/**
 * @brief Runs a linear neural-network layer without bias.
 *
 * Input is quantized before multiplication with weights and then dequantized
 * per-column before optional activation.
 *
 * @param input The [batch_size x input_features] input matrix.
 * @param weights The [input_features x output_features] layer weight matrix.
 * @param output The [batch_size x output_features] output matrix.
 * @param input_scale The scale factor for input quantization.
 * @param weight_scale_inv The [1 x output_features] inverse scale vector for weights.
 * @param input_scale_inv The inverse input scale factor.
 * @param batch_size The batch size.
 * @param input_features The input feature size.
 * @param output_features The output feature size.
 * @param apply_relu Non-zero if ReLU must be applied.
 */
void linear_layer(const int *input, const int8_t *weights, int *output, const int input_scale,
                  const int *weight_scale_inv, const int input_scale_inv,
                  const unsigned int batch_size, const unsigned int input_features,
                  const unsigned int output_features, const unsigned int apply_relu);

/**
 * @brief Runs a quantized 2D convolutional layer with ReLU activation.
 *
 * Input is quantized before convolution and then dequantized per-channel
 * before activation.
 *
 * @param input The (batch_size, input_channels, input_height, input_width) input tensor.
 * @param weights The (output_channels, input_channels, kernel_height, kernel_width) weight tensor.
 * @param output The (batch_size, output_channels, output_height, output_width) output tensor.
 * @param input_scale The scale factor for input quantization.
 * @param weight_scale_inv The inverse scale factors per output channel.
 * @param input_scale_inv The inverse input scale factor.
 * @param batch_size The batch size.
 * @param input_channels The input channels.
 * @param output_channels The output channels.
 * @param input_height The input height.
 * @param input_width The input width.
 * @param output_height The output height.
 * @param output_width The output width.
 * @param kernel_height The kernel height.
 * @param kernel_width The kernel width.
 * @param stride_height The stride height.
 * @param stride_width The stride width.
 */
void conv2d_layer(const int *input, const int8_t *weights, int *output, const int input_scale,
                  const int *weight_scale_inv, const int input_scale_inv,
                  const unsigned int batch_size, const unsigned int input_channels,
                  const unsigned int output_channels, const int input_height,
                  const int input_width, const int output_height, const int output_width,
                  const int kernel_height, const int kernel_width, const int stride_height,
                  const int stride_width);

#endif // NN_H
