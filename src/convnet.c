#include "convnet.h"
#include "nn.h"

void convnet_forward(const int *input, int *conv1_out, int *pool1_out,
                            int *conv2_out, int *pool2_out, int *linear1_out,
                            int *linear2_out, int *output,
                            unsigned int *class_indices)
{
    // conv1: input -> conv -> dequant -> ReLU
    conv2d_layer(input, layer_1_weight, conv1_out, layer_1_input_scale, layer_1_weight_scale_inv,
                 layer_1_input_scale_inv, BATCH_SIZE, INPUT_CHANNELS, CONV1_OUT_CHANNELS,
                 INPUT_HEIGHT, INPUT_WIDTH, CONV1_OUT_HEIGHT, CONV1_OUT_WIDTH,
                 5, 5, 1, 1);

    // pool1: downsample conv1 features
    pooling2d(conv1_out, pool1_out, BATCH_SIZE, CONV1_OUT_CHANNELS, CONV1_OUT_HEIGHT,
              CONV1_OUT_WIDTH, POOL1_OUT_HEIGHT, POOL1_OUT_WIDTH,
              2, 2, 2, 2);

    // conv2: pool1 -> conv -> dequant -> ReLU
    conv2d_layer(pool1_out, layer_2_weight, conv2_out, layer_2_input_scale, layer_2_weight_scale_inv,
                 layer_2_input_scale_inv, BATCH_SIZE, CONV1_OUT_CHANNELS, CONV2_OUT_CHANNELS,
                 POOL1_OUT_HEIGHT, POOL1_OUT_WIDTH, CONV2_OUT_HEIGHT, CONV2_OUT_WIDTH,
                 5, 5, 1, 1);

    // pool2: downsample conv2 features
    pooling2d(conv2_out, pool2_out, BATCH_SIZE, CONV2_OUT_CHANNELS, CONV2_OUT_HEIGHT,
              CONV2_OUT_WIDTH, POOL2_OUT_HEIGHT, POOL2_OUT_WIDTH,
              2, 2, 2, 2);

    // linear stack: pool2 -> fc1 -> fc2 -> fc3
    linear_layer(pool2_out, layer_3_weight, linear1_out, layer_3_input_scale, layer_3_weight_scale_inv,
                 layer_3_input_scale_inv, BATCH_SIZE,
                 CONV2_OUT_CHANNELS * POOL2_OUT_HEIGHT * POOL2_OUT_WIDTH,
                 LINEAR1_OUT_FEATURES, 1);

    linear_layer(linear1_out, layer_4_weight, linear2_out, layer_4_input_scale, layer_4_weight_scale_inv,
                 layer_4_input_scale_inv, BATCH_SIZE, LINEAR1_OUT_FEATURES,
                 LINEAR2_OUT_FEATURES, 1);

    linear_layer(linear2_out, layer_5_weight, output, layer_5_input_scale, layer_5_weight_scale_inv,
                 layer_5_input_scale_inv, BATCH_SIZE, LINEAR2_OUT_FEATURES, OUTPUT_DIM, 0);

    // final class index from logits
    argmax_over_cols(output, class_indices, BATCH_SIZE, OUTPUT_DIM);
}

void run_convnet(const int *input, unsigned int *class_indices)
{
    int conv1_out[BATCH_SIZE * CONV1_OUT_CHANNELS * CONV1_OUT_HEIGHT * CONV1_OUT_WIDTH];
    int pool1_out[BATCH_SIZE * CONV1_OUT_CHANNELS * POOL1_OUT_HEIGHT * POOL1_OUT_WIDTH];
    int conv2_out[BATCH_SIZE * CONV2_OUT_CHANNELS * CONV2_OUT_HEIGHT * CONV2_OUT_WIDTH];
    int pool2_out[BATCH_SIZE * CONV2_OUT_CHANNELS * POOL2_OUT_HEIGHT * POOL2_OUT_WIDTH];
    int linear1_out[BATCH_SIZE * LINEAR1_OUT_FEATURES];
    int linear2_out[BATCH_SIZE * LINEAR2_OUT_FEATURES];
    int output[BATCH_SIZE * OUTPUT_DIM];

    convnet_forward(input, conv1_out, pool1_out, conv2_out, pool2_out,
                           linear1_out, linear2_out, output, class_indices);
}
