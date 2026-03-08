#ifndef PARAMS
#define PARAMS

#define INPUT_FLAT_SIZE 784
#define INPUT_HEIGHT 28
#define INPUT_WIDTH 28
#define CONV1_OUT_HEIGHT 24
#define CONV1_OUT_WIDTH 24
#define POOL1_OUT_HEIGHT 12
#define POOL1_OUT_WIDTH 12
#define CONV2_OUT_HEIGHT 8
#define CONV2_OUT_WIDTH 8
#define POOL2_OUT_HEIGHT 4
#define POOL2_OUT_WIDTH 4
#define INPUT_CHANNELS 1
#define CONV1_OUT_CHANNELS 6
#define CONV2_OUT_CHANNELS 16
#define LINEAR1_OUT_FEATURES 120
#define LINEAR2_OUT_FEATURES 84
#define OUTPUT_DIM 10

#include <stdint.h>

// quantization/dequantization constants
extern const int layer_1_input_scale;
extern const int layer_1_input_scale_inv;
extern const int layer_1_weight_scale_inv[6];
extern const int layer_2_input_scale;
extern const int layer_2_input_scale_inv;
extern const int layer_2_weight_scale_inv[16];
extern const int layer_3_input_scale;
extern const int layer_3_input_scale_inv;
extern const int layer_3_weight_scale_inv[120];
extern const int layer_4_input_scale;
extern const int layer_4_input_scale_inv;
extern const int layer_4_weight_scale_inv[84];
extern const int layer_5_input_scale;
extern const int layer_5_input_scale_inv;
extern const int layer_5_weight_scale_inv[10];
// layer quantized parameters
extern const int8_t layer_1_weight[150];
extern const int8_t layer_2_weight[2400];
extern const int8_t layer_3_weight[30720];
extern const int8_t layer_4_weight[10080];
extern const int8_t layer_5_weight[840];

#endif // PARAMS
