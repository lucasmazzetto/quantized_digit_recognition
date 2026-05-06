#include "convnet.h"
#include "nn.h"

void convnet_forward(const int* input, int* conv_1_output, int* pool_1_output, int* conv_2_output,
                     int* pool_2_output, int* linear_1_output, int* linear_2_output, int* logits,
                     unsigned int* predictions)
{
    // conv1: input -> conv -> dequant -> ReLU
    conv2d_layer(input, conv_1_weight, conv_1_output, conv_1_input_scale, conv_1_weight_scale_inv,
                 conv_1_input_scale_inv, BATCH_SIZE, INPUT_CHANNELS, CONV_1_OUT_CHANNELS,
                 INPUT_HEIGHT, INPUT_WIDTH, CONV_1_OUT_HEIGHT, CONV_1_OUT_WIDTH, 5, 5, 1, 1);

    // pool1: downsample conv1 features
    pooling2d(conv_1_output, pool_1_output, BATCH_SIZE, CONV_1_OUT_CHANNELS, CONV_1_OUT_HEIGHT,
              CONV_1_OUT_WIDTH, POOL_1_OUT_HEIGHT, POOL_1_OUT_WIDTH, 2, 2, 2, 2);

    // conv2: pool1 -> conv -> dequant -> ReLU
    conv2d_layer(pool_1_output, conv_2_weight, conv_2_output, conv_2_input_scale,
                 conv_2_weight_scale_inv, conv_2_input_scale_inv, BATCH_SIZE, CONV_1_OUT_CHANNELS,
                 CONV_2_OUT_CHANNELS, POOL_1_OUT_HEIGHT, POOL_1_OUT_WIDTH, CONV_2_OUT_HEIGHT,
                 CONV_2_OUT_WIDTH, 5, 5, 1, 1);

    // pool2: downsample conv2 features
    pooling2d(conv_2_output, pool_2_output, BATCH_SIZE, CONV_2_OUT_CHANNELS, CONV_2_OUT_HEIGHT,
              CONV_2_OUT_WIDTH, POOL_2_OUT_HEIGHT, POOL_2_OUT_WIDTH, 2, 2, 2, 2);

    // linear stack: pool_2 -> linear_1 -> linear_2 -> linear_3
    linear_layer(pool_2_output, linear_1_weight, linear_1_output, linear_1_input_scale,
                 linear_1_weight_scale_inv, linear_1_input_scale_inv, BATCH_SIZE,
                 CONV_2_OUT_CHANNELS * POOL_2_OUT_HEIGHT * POOL_2_OUT_WIDTH, LINEAR_1_OUT_FEATURES,
                 1);

    linear_layer(linear_1_output, linear_2_weight, linear_2_output, linear_2_input_scale,
                 linear_2_weight_scale_inv, linear_2_input_scale_inv, BATCH_SIZE,
                 LINEAR_1_OUT_FEATURES, LINEAR_2_OUT_FEATURES, 1);

    linear_layer(linear_2_output, linear_3_weight, logits, linear_3_input_scale,
                 linear_3_weight_scale_inv, linear_3_input_scale_inv, BATCH_SIZE,
                 LINEAR_2_OUT_FEATURES, OUTPUT_DIM, 0);

    // final class index from logits
    argmax_per_row(logits, predictions, BATCH_SIZE, OUTPUT_DIM);
}

void convnet_run(const int* input, unsigned int* predictions)
{
    int conv_1_output[BATCH_SIZE * CONV_1_OUT_CHANNELS * CONV_1_OUT_HEIGHT * CONV_1_OUT_WIDTH];
    int pool_1_output[BATCH_SIZE * CONV_1_OUT_CHANNELS * POOL_1_OUT_HEIGHT * POOL_1_OUT_WIDTH];
    int conv_2_output[BATCH_SIZE * CONV_2_OUT_CHANNELS * CONV_2_OUT_HEIGHT * CONV_2_OUT_WIDTH];
    int pool_2_output[BATCH_SIZE * CONV_2_OUT_CHANNELS * POOL_2_OUT_HEIGHT * POOL_2_OUT_WIDTH];
    int linear_1_output[BATCH_SIZE * LINEAR_1_OUT_FEATURES];
    int linear_2_output[BATCH_SIZE * LINEAR_2_OUT_FEATURES];
    int logits[BATCH_SIZE * OUTPUT_DIM];

    convnet_forward(input, conv_1_output, pool_1_output, conv_2_output, pool_2_output,
                    linear_1_output, linear_2_output, logits, predictions);
}
