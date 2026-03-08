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

#endif // CONVNET_H
