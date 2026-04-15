#ifndef CONVNET_H
#define CONVNET_H

#define BATCH_SIZE 1

#include "params.h"

/**
 * @brief Runs the pre-defined convolutional neural network.
 *
 * @param input The input tensor.
 * @param predictions The [batch_size x 1] vector storing predicted class indices.
 */
void run_convnet(const int *input, unsigned int *predictions);

/**
 * @brief Shared neural-network forward function with intermediate outputs.
 *
 * @param input The flattened fixed-point input tensor.
 * @param conv_1_output The first convolution output buffer.
 * @param pool_1_output The first max-pooling output buffer.
 * @param conv_2_output The second convolution output buffer.
 * @param pool_2_output The second max-pooling output buffer.
 * @param linear_1_output The first linear layer output buffer.
 * @param linear_2_output The second linear layer output buffer.
 * @param logits The final logits buffer.
 * @param predictions The [batch_size x 1] vector storing predicted class indices.
 */
void convnet_forward(const int *input, int *conv_1_output, int *pool_1_output,
                     int *conv_2_output, int *pool_2_output,
                     int *linear_1_output, int *linear_2_output, int *logits,
                     unsigned int *predictions);

#endif  // CONVNET_H
