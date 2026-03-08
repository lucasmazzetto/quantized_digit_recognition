#ifndef CONVNET_H
#define CONVNET_H

#define BATCH_SIZE 1

#include "params.h"

/**
 * @brief Runs the pre-defined convolutional neural network.
 *
 * @param input The input tensor.
 * @param class_indices The batch_sizex1 vector storing predicted class indices.
 */
void run_convnet(const int *input, unsigned int *class_indices);

/**
 * @brief Shared neural-network forward function with intermediate outputs.
 *
 * @param input The flattened fixed-point input tensor.
 * @param conv1_out The first convolution output buffer.
 * @param pool1_out The first max-pooling output buffer.
 * @param conv2_out The second convolution output buffer.
 * @param pool2_out The second max-pooling output buffer.
 * @param linear1_out The first linear layer output buffer.
 * @param linear2_out The second linear layer output buffer.
 * @param output The final logits/output buffer.
 * @param class_indices The batch_sizex1 vector storing predicted class indices.
 */
void convnet_forward(const int *input, int *conv1_out, int *pool1_out,
                     int *conv2_out, int *pool2_out, int *linear1_out,
                     int *linear2_out, int *output,
                     unsigned int *class_indices);

#endif // CONVNET_H
