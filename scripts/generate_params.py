import argparse
import re
from pathlib import Path

import numpy as np
import torch


def conv_output_dim(input_dim: int, kernel_size: int, stride: int):
    """
    @brief Calculate 1D output dimension for valid convolution/pooling.

    @param input_dim Input dimension.
    @param kernel_size Kernel size.
    @param stride Stride value.
    @return Output dimension.
    """
    return (input_dim - kernel_size) // stride + 1


def to_numpy(value):
    """
    @brief Convert tensors/arrays/scalars to numpy arrays.

    @param value Input value.
    @return NumPy representation of the input.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def scalar_to_fxp(value, frac_bits: int):
    """
    @brief Convert scalar value to fixed-point integer.

    @param value Scalar value.
    @param frac_bits Number of fractional bits.
    @return Fixed-point integer value.
    """
    return int(np.rint(float(np.asarray(value)) * (1 << frac_bits)))


def array_to_fxp(values, frac_bits: int):
    """
    @brief Convert array values to fixed-point integer array.

    @param values Array-like input.
    @param frac_bits Number of fractional bits.
    @return Fixed-point NumPy integer array.
    """
    arr = np.atleast_1d(to_numpy(values)).astype(np.float64)
    return np.rint(arr * (1 << frac_bits)).astype(np.int64)


def get_layer_indices(state_dict):
    """
    @brief Get sorted layer indices from quantized state dictionary keys.

    @param state_dict Quantized state dictionary.
    @return Sorted list of layer indices.
    """
    indices = set()
    pattern = re.compile(r'^layer_(\d+)_weight$')
    for key in state_dict:
        match = pattern.match(key)
        if match:
            indices.add(int(match.group(1)))
    return sorted(indices)


def infer_dimensions(state_dict, input_h: int, input_w: int, input_c: int):
    """
    @brief Infer all model dimensions from quantized weights and input shape.

    @param state_dict Quantized state dictionary.
    @param input_h Input height.
    @param input_w Input width.
    @param input_c Input channels.
    @return Dictionary with inferred dimensions and sizes.
    """
    w1 = to_numpy(state_dict['layer_1_weight'])
    w2 = to_numpy(state_dict['layer_2_weight'])
    w3 = to_numpy(state_dict['layer_3_weight'])
    w4 = to_numpy(state_dict['layer_4_weight'])
    w5 = to_numpy(state_dict['layer_5_weight'])

    c1 = int(w1.shape[0])
    c2 = int(w2.shape[0])
    k1 = int(w1.shape[2])
    k2 = int(w2.shape[2])

    h1_conv = conv_output_dim(input_h, k1, stride=1)
    w1_conv = conv_output_dim(input_w, k1, stride=1)
    h1_pool = conv_output_dim(h1_conv, kernel_size=2, stride=2)
    w1_pool = conv_output_dim(w1_conv, kernel_size=2, stride=2)

    h2_conv = conv_output_dim(h1_pool, k2, stride=1)
    w2_conv = conv_output_dim(w1_pool, k2, stride=1)
    h2_pool = conv_output_dim(h2_conv, kernel_size=2, stride=2)
    w2_pool = conv_output_dim(w2_conv, kernel_size=2, stride=2)

    linear_input = int(w3.shape[0])
    expected_linear_input = c2 * h2_pool * w2_pool
    if linear_input != expected_linear_input:
        raise ValueError(
            f'Mismatch in first linear input size: got {linear_input}, '
            f'expected {expected_linear_input} from conv dims.'
        )

    s1_linear = int(w3.shape[1])
    s2_linear = int(w4.shape[1])
    output_dim = int(w5.shape[1])

    return {'INPUT_HEIGHT': input_h,
            'INPUT_WIDTH': input_w,
            'INPUT_CHANNELS': input_c,
            'CONV1_OUT_HEIGHT': h1_conv,
            'CONV1_OUT_WIDTH': w1_conv,
            'POOL1_OUT_HEIGHT': h1_pool,
            'POOL1_OUT_WIDTH': w1_pool,
            'CONV2_OUT_HEIGHT': h2_conv,
            'CONV2_OUT_WIDTH': w2_conv,
            'POOL2_OUT_HEIGHT': h2_pool,
            'POOL2_OUT_WIDTH': w2_pool,
            'CONV1_OUT_CHANNELS': c1,
            'CONV2_OUT_CHANNELS': c2,
            'LINEAR1_OUT_FEATURES': s1_linear,
            'LINEAR2_OUT_FEATURES': s2_linear,
            'OUTPUT_DIM': output_dim}


def write_header_file(path: Path, state_dict, layer_indices, dims: dict):
    """
    @brief Write generated params.h file.

    @param path Output header path.
    @param state_dict Quantized state dictionary.
    @param layer_indices Sorted layer indices.
    @param dims Inferred dimensions dictionary.
    @return None.
    """
    with path.open('w', encoding='utf-8') as f:
        f.write('#ifndef PARAMS\n#define PARAMS\n\n')

        f.write(
            f"#define INPUT_FLAT_SIZE {dims['INPUT_HEIGHT'] * dims['INPUT_WIDTH'] * dims['INPUT_CHANNELS']}\n"
        )
        f.write(f"#define INPUT_HEIGHT {dims['INPUT_HEIGHT']}\n")
        f.write(f"#define INPUT_WIDTH {dims['INPUT_WIDTH']}\n")
        f.write(f"#define CONV1_OUT_HEIGHT {dims['CONV1_OUT_HEIGHT']}\n")
        f.write(f"#define CONV1_OUT_WIDTH {dims['CONV1_OUT_WIDTH']}\n")
        f.write(f"#define POOL1_OUT_HEIGHT {dims['POOL1_OUT_HEIGHT']}\n")
        f.write(f"#define POOL1_OUT_WIDTH {dims['POOL1_OUT_WIDTH']}\n")
        f.write(f"#define CONV2_OUT_HEIGHT {dims['CONV2_OUT_HEIGHT']}\n")
        f.write(f"#define CONV2_OUT_WIDTH {dims['CONV2_OUT_WIDTH']}\n")
        f.write(f"#define POOL2_OUT_HEIGHT {dims['POOL2_OUT_HEIGHT']}\n")
        f.write(f"#define POOL2_OUT_WIDTH {dims['POOL2_OUT_WIDTH']}\n")
        f.write(f"#define INPUT_CHANNELS {dims['INPUT_CHANNELS']}\n")
        f.write(f"#define CONV1_OUT_CHANNELS {dims['CONV1_OUT_CHANNELS']}\n")
        f.write(f"#define CONV2_OUT_CHANNELS {dims['CONV2_OUT_CHANNELS']}\n")
        f.write(f"#define LINEAR1_OUT_FEATURES {dims['LINEAR1_OUT_FEATURES']}\n")
        f.write(f"#define LINEAR2_OUT_FEATURES {dims['LINEAR2_OUT_FEATURES']}\n")
        f.write(f"#define OUTPUT_DIM {dims['OUTPUT_DIM']}\n\n")
        f.write('#include <stdint.h>\n\n')

        f.write('// quantization/dequantization constants\n')
        for layer_idx in layer_indices:
            f.write(f'extern const int layer_{layer_idx}_input_scale;\n')
            f.write(f'extern const int layer_{layer_idx}_input_scale_inv;\n')

            sw_inv = np.atleast_1d(to_numpy(state_dict[f'layer_{layer_idx}_s_w_inv']))
            f.write(f'extern const int layer_{layer_idx}_weight_scale_inv[{sw_inv.size}];\n')

        f.write('// layer quantized parameters\n')
        for layer_idx in layer_indices:
            weight = to_numpy(state_dict[f'layer_{layer_idx}_weight'])
            f.write(f'extern const int8_t layer_{layer_idx}_weight[{weight.size}];\n')

        f.write('\n#endif // PARAMS\n')


def write_source_file(path: Path, state_dict, layer_indices, frac_bits: int):
    """
    @brief Write generated params.c file.

    @param path Output source path.
    @param state_dict Quantized state dictionary.
    @param layer_indices Sorted layer indices.
    @param frac_bits Number of fractional bits for fixed-point conversion.
    @return None.
    """
    with path.open('w', encoding='utf-8') as f:
        f.write('#include "params.h"\n\n')

        for layer_idx in layer_indices:
            sx = scalar_to_fxp(state_dict[f'layer_{layer_idx}_s_x'], frac_bits)
            sx_inv = scalar_to_fxp(state_dict[f'layer_{layer_idx}_s_x_inv'], frac_bits)
            sw_inv = array_to_fxp(state_dict[f'layer_{layer_idx}_s_w_inv'], frac_bits)

            f.write(f'const int layer_{layer_idx}_input_scale = {sx};\n\n')
            f.write(f'const int layer_{layer_idx}_input_scale_inv = {sx_inv};\n\n')

            sw_values = ', '.join(str(int(v)) for v in sw_inv.flatten())
            f.write(
                f'const int layer_{layer_idx}_weight_scale_inv[{sw_inv.size}] = '
                f'{{{sw_values}}};\n\n'
            )

        for layer_idx in layer_indices:
            weights = to_numpy(state_dict[f'layer_{layer_idx}_weight']).flatten()
            values = ', '.join(str(int(v)) for v in weights)
            f.write(
                f'const int8_t layer_{layer_idx}_weight[{weights.size}] = '
                f'{{{values}}};\n\n'
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate params.h and params.c from a quantized model checkpoint.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=Path, default=Path('models/quantized.pt'),
                        help='Path to quantized checkpoint to convert (e.g., quantized.pt).')
    parser.add_argument('--output-dir', type=Path, default=Path('src'),
                        help='Directory where generated params.c is written.')
    parser.add_argument('--header-dir', type=Path, default=Path('include'),
                        help='Directory where generated params.h is written.')
    parser.add_argument('--input-h', type=int, default=28,
                        help='Input image height used to infer network dimensions.')
    parser.add_argument('--input-w', type=int, default=28,
                        help='Input image width used to infer network dimensions.')
    parser.add_argument('--input-c', type=int, default=1,
                        help='Input channel count used to infer network dimensions.')
    parser.add_argument('--fxp-frac-bits', type=int, default=16,
                        help='Number of fractional bits for fixed-point scale conversion.')
    args = parser.parse_args()

    checkpoint = torch.load(args.path, weights_only=False)
    if 'state_dict' not in checkpoint:
        raise KeyError(f"'state_dict' not found in checkpoint: {args.path}")
    state_dict = checkpoint['state_dict']

    layer_indices = get_layer_indices(state_dict)
    if layer_indices != [1, 2, 3, 4, 5]:
        raise ValueError(
            f'Expected quantized layers [1, 2, 3, 4, 5], got {layer_indices}.'
        )

    required_suffixes = ('_weight', '_s_x', '_s_x_inv', '_s_w_inv')
    for layer_idx in layer_indices:
        for suffix in required_suffixes:
            key = f'layer_{layer_idx}{suffix}'
            if key not in state_dict:
                raise KeyError(f'Missing key in quantized state_dict: {key}')

    dims = infer_dimensions(state_dict, args.input_h, args.input_w, args.input_c)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.header_dir.mkdir(parents=True, exist_ok=True)
    header_path = args.header_dir / 'params.h'
    source_path = args.output_dir / 'params.c'

    write_header_file(header_path, state_dict, layer_indices, dims)
    write_source_file(source_path, state_dict, layer_indices, args.fxp_frac_bits)

    print(f'Wrote {header_path}')
    print(f'Wrote {source_path}')
