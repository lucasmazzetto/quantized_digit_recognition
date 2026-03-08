#include "convnet.h"
#include "nn.h"

void run_convnet(const int *input, unsigned int *class_indices)
{
    int out_conv1[BATCH_SIZE * CONV1_OUT_CHANNELS * CONV1_OUT_HEIGHT * CONV1_OUT_WIDTH];
    int out_pool1[BATCH_SIZE * CONV1_OUT_CHANNELS * POOL1_OUT_HEIGHT * POOL1_OUT_WIDTH];
    int out_conv2[BATCH_SIZE * CONV2_OUT_CHANNELS * CONV2_OUT_HEIGHT * CONV2_OUT_WIDTH];
    int out_pool2[BATCH_SIZE * CONV2_OUT_CHANNELS * POOL2_OUT_HEIGHT * POOL2_OUT_WIDTH];
    int out_linear1[BATCH_SIZE * LINEAR1_OUT_FEATURES];
    int out_linear2[BATCH_SIZE * LINEAR2_OUT_FEATURES];
    int output[BATCH_SIZE * OUTPUT_DIM];

    conv2d_layer(input, layer_1_weight, out_conv1, layer_1_input_scale, layer_1_weight_scale_inv,
                 layer_1_input_scale_inv, BATCH_SIZE, INPUT_CHANNELS, CONV1_OUT_CHANNELS,
                 INPUT_HEIGHT, INPUT_WIDTH, CONV1_OUT_HEIGHT, CONV1_OUT_WIDTH,
                 5, 5, 1, 1);

    pooling2d(out_conv1, out_pool1, BATCH_SIZE, CONV1_OUT_CHANNELS, CONV1_OUT_HEIGHT,
              CONV1_OUT_WIDTH, POOL1_OUT_HEIGHT, POOL1_OUT_WIDTH,
              2, 2, 2, 2);

    conv2d_layer(out_pool1, layer_2_weight, out_conv2, layer_2_input_scale, layer_2_weight_scale_inv,
                 layer_2_input_scale_inv, BATCH_SIZE, CONV1_OUT_CHANNELS, CONV2_OUT_CHANNELS,
                 POOL1_OUT_HEIGHT, POOL1_OUT_WIDTH, CONV2_OUT_HEIGHT, CONV2_OUT_WIDTH,
                 5, 5, 1, 1);

    pooling2d(out_conv2, out_pool2, BATCH_SIZE, CONV2_OUT_CHANNELS, CONV2_OUT_HEIGHT,
              CONV2_OUT_WIDTH, POOL2_OUT_HEIGHT, POOL2_OUT_WIDTH,
              2, 2, 2, 2);

    linear_layer(out_pool2, layer_3_weight, out_linear1, layer_3_input_scale, layer_3_weight_scale_inv,
                 layer_3_input_scale_inv, BATCH_SIZE,
                 CONV2_OUT_CHANNELS * POOL2_OUT_HEIGHT * POOL2_OUT_WIDTH,
                 LINEAR1_OUT_FEATURES, 1);

    linear_layer(out_linear1, layer_4_weight, out_linear2, layer_4_input_scale, layer_4_weight_scale_inv,
                 layer_4_input_scale_inv, BATCH_SIZE, LINEAR1_OUT_FEATURES,
                 LINEAR2_OUT_FEATURES, 1);

    linear_layer(out_linear2, layer_5_weight, output, layer_5_input_scale, layer_5_weight_scale_inv,
                 layer_5_input_scale_inv, BATCH_SIZE, LINEAR2_OUT_FEATURES, OUTPUT_DIM, 0);

    argmax_over_cols(output, class_indices, BATCH_SIZE, OUTPUT_DIM);
}
