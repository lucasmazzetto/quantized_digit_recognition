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
 * @brief Calculates matrix multiplication as: Y = XW.
 *
 * @param left_matrix The left matrix (X), size batch_sizexinput_features.
 * @param right_matrix The right matrix (W), size input_featuresxoutput_features.
 * @param output_matrix The output matrix (Y), size batch_sizexoutput_features.
 * @param batch_size The number of rows in X.
 * @param input_features The number of columns/rows in X/W.
 * @param output_features The number of columns in W.
 */
void mat_mult(const int8_t *left_matrix, const int8_t *right_matrix, int *output_matrix,
              const unsigned int batch_size, const unsigned int input_features,
              const unsigned int output_features);

/**
 * @brief Computes output dimension for valid convolution/pooling.
 *
 * @param input_dim The input dimension.
 * @param kernel_size The kernel size.
 * @param stride The stride size.
 * @return The output dimension.
 */
int get_output_dim(int input_dim, int kernel_size, int stride);

/**
 * @brief Performs a low-level 2D convolution kernel.
 *
 * @param input The input tensor.
 * @param weights The convolution weights.
 * @param output The output tensor.
 * @param batch_size The number of samples.
 * @param input_channels The number of input channels.
 * @param output_channels The number of output channels.
 * @param input_height The input height.
 * @param input_width The input width.
 * @param output_height The output height.
 * @param output_width The output width.
 * @param kernel_height The kernel height.
 * @param kernel_width The kernel width.
 * @param stride_height The height stride.
 * @param stride_width The width stride.
 */
void conv2d(const int8_t *input, const int8_t *weights, int *output,
            int batch_size, int input_channels, int output_channels, int input_height,
            int input_width, int output_height, int output_width,
            int kernel_height, int kernel_width, int stride_height,
            int stride_width);

/**
 * @brief Performs 2D max pooling.
 *
 * @param input The input tensor.
 * @param output The output tensor.
 * @param batch_size The number of samples.
 * @param output_channels The number of channels.
 * @param input_height The input height.
 * @param input_width The input width.
 * @param output_height The output height.
 * @param output_width The output width.
 * @param kernel_height The kernel height.
 * @param kernel_width The kernel width.
 * @param stride_height The height stride.
 * @param stride_width The width stride.
 */
void pooling2d(int *input, int *output, int batch_size, int output_channels, int input_height,
               int input_width, int output_height, int output_width, int kernel_height,
               int kernel_width, int stride_height, int stride_width);

/**
 * @brief ReLU activation function.
 *
 * @param tensor_inout The input/output tensor.
 * @param size The flattened tensor size.
 */
void relu(int *tensor_inout, const unsigned int size);

/**
 * @brief Scale quantization of a tensor by a single amax value.
 *
 * @param tensor_inout The input tensor.
 * @param tensor_q The output quantized tensor.
 * @param scale_factor The 127 / amax value (fixed-point).
 * @param size The size of the flattened tensor.
 */
void quantize(const int *tensor_inout, int8_t *tensor_q, const int scale_factor,
              const unsigned int size);

/**
 * @brief Scale dequantization with per-row granularity.
 *
 * @param matrix_inout The batch_sizexoutput_features matrix to dequantize.
 * @param scale_factor_w_inv The 1xoutput_features weight inverse-scale values.
 * @param scale_factor_x_inv The input inverse scale factor.
 * @param batch_size The number of samples.
 * @param output_features The output feature size.
 */
void dequantize_per_row(int *matrix_inout, const int *scale_factor_w_inv,
                        const int scale_factor_x_inv, const unsigned int batch_size,
                        const unsigned int output_features);

/**
 * @brief Scale dequantization with per-channel granularity.
 *
 * @param tensor_inout The input tensor to dequantize (batch_size, C, ...).
 * @param weight_scale_inv The 1xC row vector of inverse amax values.
 * @param input_scale_inv The input inverse amax value.
 * @param batch_size The number of samples.
 * @param output_channels The number of channels.
 * @param input_features The number of remaining flattened dimensions.
 */
void dequantize_per_channel(int *tensor_inout, const int *weight_scale_inv,
                            const int input_scale_inv, const unsigned int batch_size,
                            const unsigned int output_channels,
                            const unsigned int input_features);

/**
 * @brief Calculates argmax per row of a batch_sizexoutput_features matrix.
 *
 * @param matrix_in The batch_sizexoutput_features input matrix.
 * @param indices The output indices.
 * @param batch_size The number of rows.
 * @param output_features The number of columns.
 */
void argmax_over_cols(const int *matrix_in, unsigned int *indices,
                      const unsigned int batch_size, const unsigned int output_features);

/**
 * @brief Runs a linear neural-network layer without bias.
 *
 * Input is quantized before multiplication with weights and then dequantized
 * per-row before optional activation.
 *
 * @param input The batch_sizexinput_features input matrix.
 * @param weights The input_featuresxoutput_features layer weight matrix.
 * @param output The batch_sizexoutput_features output matrix.
 * @param input_scale The scale factor for input quantization.
 * @param weight_scale_inv The 1xoutput_features inverse scale vector for weights.
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
