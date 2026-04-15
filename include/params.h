#ifndef PARAMS
#define PARAMS

#define INPUT_FLAT_SIZE 784
#define INPUT_HEIGHT 28
#define INPUT_WIDTH 28
#define CONV_1_OUT_HEIGHT 24
#define CONV_1_OUT_WIDTH 24
#define POOL_1_OUT_HEIGHT 12
#define POOL_1_OUT_WIDTH 12
#define CONV_2_OUT_HEIGHT 8
#define CONV_2_OUT_WIDTH 8
#define POOL_2_OUT_HEIGHT 4
#define POOL_2_OUT_WIDTH 4
#define INPUT_CHANNELS 1
#define CONV_1_OUT_CHANNELS 6
#define CONV_2_OUT_CHANNELS 16
#define LINEAR_1_OUT_FEATURES 120
#define LINEAR_2_OUT_FEATURES 84
#define OUTPUT_DIM 10

#include <stdint.h>

// quantization/dequantization constants
extern const int conv_1_input_scale;
extern const int conv_1_input_scale_inv;
extern const int conv_1_weight_scale_inv[6];
extern const int conv_2_input_scale;
extern const int conv_2_input_scale_inv;
extern const int conv_2_weight_scale_inv[16];
extern const int linear_1_input_scale;
extern const int linear_1_input_scale_inv;
extern const int linear_1_weight_scale_inv[120];
extern const int linear_2_input_scale;
extern const int linear_2_input_scale_inv;
extern const int linear_2_weight_scale_inv[84];
extern const int linear_3_input_scale;
extern const int linear_3_input_scale_inv;
extern const int linear_3_weight_scale_inv[10];

// layer quantized parameters
extern const int8_t conv_1_weight[150];
extern const int8_t conv_2_weight[2400];
extern const int8_t linear_1_weight[30720];
extern const int8_t linear_2_weight[10080];
extern const int8_t linear_3_weight[840];

#endif // PARAMS
