import argparse
import ctypes
import re
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import ConvNet

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LAYER_ORDER = [("conv1", "Conv1 (ReLU)"),
               ("pool1", "Pool1"),
               ("conv2", "Conv2 (ReLU)"),
               ("pool2", "Pool2")]


def ensure_contiguous(array: np.ndarray) -> np.ndarray:
    """
    @brief Ensures an array is C-contiguous.

    @param array Input NumPy array.
    @return C-contiguous array.
    """
    return np.ascontiguousarray(array) if not array.flags["C_CONTIGUOUS"] else array


def load_c_lib(library_path: Path):
    """
    @brief Loads the compiled C shared library.

    @param library_path Path to the compiled shared object.
    @return ctypes library handle.
    """
    try:
        return ctypes.CDLL(str(library_path.resolve()))
    except OSError as exc:
        raise RuntimeError(f"Unable to load C library: {library_path}") from exc


def setup_c_signature(c_lib):
    """
    @brief Binds the convnet_forward function signature for ctypes.

    @param c_lib Loaded C shared library.
    """
    c_int_p = ctypes.POINTER(ctypes.c_int)
    c_uint_p = ctypes.POINTER(ctypes.c_uint)

    c_lib.convnet_forward.argtypes = (c_int_p, c_int_p, c_int_p, c_int_p,
                                      c_int_p, c_int_p, c_int_p, c_int_p,
                                      c_uint_p)
    c_lib.convnet_forward.restype = None


def load_params_dims(header_path: Path) -> Dict[str, int]:
    """
    @brief Parses tensor dimensions from generated params.h.

    @param header_path Path to the generated params header.
    @return Dictionary containing required tensor dimensions.
    """
    if not header_path.exists():
        raise FileNotFoundError(f"Params header not found: {header_path}")

    defines = {}
    pattern = re.compile(r"^#define\s+([A-Z0-9_]+)\s+([0-9]+)$")

    for line in header_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            defines[match.group(1)] = int(match.group(2))

    required = ["CONV_1_OUT_CHANNELS", "CONV_1_OUT_HEIGHT", "CONV_1_OUT_WIDTH",
                "POOL_1_OUT_HEIGHT", "POOL_1_OUT_WIDTH", "CONV_2_OUT_CHANNELS",
                "CONV_2_OUT_HEIGHT", "CONV_2_OUT_WIDTH", "POOL_2_OUT_HEIGHT",
                "POOL_2_OUT_WIDTH", "LINEAR_1_OUT_FEATURES", "LINEAR_2_OUT_FEATURES",
                "OUTPUT_DIM"]

    missing = [key for key in required if key not in defines]
    if missing:
        raise KeyError(f"Missing required defines in params header: {missing}")

    return {key: defines[key] for key in required}


def load_float_model(model_path: Path) -> ConvNet:
    """
    @brief Loads the original floating-point model checkpoint.

    @param model_path Path to float model checkpoint.
    @return Loaded ConvNet model in eval mode.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    saved_stats = torch.load(model_path, map_location="cpu", weights_only=False)

    if isinstance(saved_stats, dict) and "state_dict" in saved_stats:
        state_dict = saved_stats["state_dict"]
    else:
        state_dict = saved_stats

    model = ConvNet(h=28, w=28, inputs=1, outputs=10)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def find_representative_indices(dataset) -> Dict[int, int]:
    """
    @brief Finds one sample index for each label from 0 to 9.

    @param dataset MNIST dataset.
    @return Mapping label -> first index found in dataset.
    """
    representative_indices = {}

    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label = int(label)

        if label not in representative_indices:
            representative_indices[label] = idx

        if len(representative_indices) == 10:
            break

    return representative_indices


def extract_original_feature_maps(sample: torch.Tensor, model: ConvNet) -> Dict[str, np.ndarray]:
    """
    @brief Extracts conv/pool feature maps from the original Python model.

    @param sample One MNIST sample tensor with shape (1, 28, 28).
    @param model Floating-point model in eval mode.
    @return Dictionary with conv1, pool1, conv2 and pool2 feature maps.
    """
    x = sample.unsqueeze(0)

    # Match C conv2d_layer output semantics (conv + ReLU)
    x = model.convolutional_layers[0](x)
    x = model.convolutional_layers[1](x)
    conv1 = x.squeeze(0).cpu().numpy().astype(np.float32)

    x = model.convolutional_layers[2](x)
    pool1 = x.squeeze(0).cpu().numpy().astype(np.float32)

    x = model.convolutional_layers[3](x)
    x = model.convolutional_layers[4](x)
    conv2 = x.squeeze(0).cpu().numpy().astype(np.float32)

    x = model.convolutional_layers[5](x)
    pool2 = x.squeeze(0).cpu().numpy().astype(np.float32)

    return {"conv1": conv1, "pool1": pool1, "conv2": conv2, "pool2": pool2}


def run_c_convnet_forward(sample: torch.Tensor, c_lib, dims: Dict[str, int],
                          frac_bits: int) -> Tuple[int, Dict[str, np.ndarray]]:
    """
    @brief Runs C convnet_forward and returns dequantized conv/pool outputs.

    @param sample One MNIST sample tensor with shape (1, 28, 28).
    @param c_lib Loaded C shared library.
    @param dims Model dimensions parsed from params.h.
    @param frac_bits Fractional bits used across fixed-point C inference.
    @return Predicted class and dictionary with dequantized conv/pool feature maps.
    """
    # Use the same input preparation strategy as scripts/eval.py
    input_quantized = (sample * (1 << frac_bits)).round().to(torch.int32)
    input_flat = ensure_contiguous(input_quantized.flatten().cpu().numpy().astype(np.intc))

    conv_1_size = dims["CONV_1_OUT_CHANNELS"] * dims["CONV_1_OUT_HEIGHT"] * dims["CONV_1_OUT_WIDTH"]
    pool_1_size = dims["CONV_1_OUT_CHANNELS"] * dims["POOL_1_OUT_HEIGHT"] * dims["POOL_1_OUT_WIDTH"]
    conv_2_size = dims["CONV_2_OUT_CHANNELS"] * dims["CONV_2_OUT_HEIGHT"] * dims["CONV_2_OUT_WIDTH"]
    pool_2_size = dims["CONV_2_OUT_CHANNELS"] * dims["POOL_2_OUT_HEIGHT"] * dims["POOL_2_OUT_WIDTH"]

    conv_1_output = ensure_contiguous(np.zeros(conv_1_size, dtype=np.intc))
    pool_1_output = ensure_contiguous(np.zeros(pool_1_size, dtype=np.intc))
    conv_2_output = ensure_contiguous(np.zeros(conv_2_size, dtype=np.intc))
    pool_2_output = ensure_contiguous(np.zeros(pool_2_size, dtype=np.intc))

    # Required by C signature, not used in feature-map comparison plots
    linear_1_output = ensure_contiguous(np.zeros(dims["LINEAR_1_OUT_FEATURES"], dtype=np.intc))
    linear_2_output = ensure_contiguous(np.zeros(dims["LINEAR_2_OUT_FEATURES"], dtype=np.intc))
    output_out = ensure_contiguous(np.zeros(dims["OUTPUT_DIM"], dtype=np.intc))
    predictions = ensure_contiguous(np.zeros(1, dtype=np.uintc))

    c_int_p = ctypes.POINTER(ctypes.c_int)
    c_uint_p = ctypes.POINTER(ctypes.c_uint)

    c_convnet_forward = c_lib.convnet_forward
    c_convnet_forward(input_flat.ctypes.data_as(c_int_p),
                      conv_1_output.ctypes.data_as(c_int_p),
                      pool_1_output.ctypes.data_as(c_int_p),
                      conv_2_output.ctypes.data_as(c_int_p),
                      pool_2_output.ctypes.data_as(c_int_p),
                      linear_1_output.ctypes.data_as(c_int_p),
                      linear_2_output.ctypes.data_as(c_int_p),
                      output_out.ctypes.data_as(c_int_p),
                      predictions.ctypes.data_as(c_uint_p))

    # conv/pool outputs are Q(frac_bits) in C, convert back to float for visualization
    scale = float(1 << frac_bits)

    conv_1 = conv_1_output.reshape(dims["CONV_1_OUT_CHANNELS"],
                                   dims["CONV_1_OUT_HEIGHT"],
                                   dims["CONV_1_OUT_WIDTH"]).astype(np.float32) / scale

    pool_1 = pool_1_output.reshape(dims["CONV_1_OUT_CHANNELS"],
                                   dims["POOL_1_OUT_HEIGHT"],
                                   dims["POOL_1_OUT_WIDTH"]).astype(np.float32) / scale

    conv_2 = conv_2_output.reshape(dims["CONV_2_OUT_CHANNELS"],
                                   dims["CONV_2_OUT_HEIGHT"],
                                   dims["CONV_2_OUT_WIDTH"]).astype(np.float32) / scale

    pool_2 = pool_2_output.reshape(dims["CONV_2_OUT_CHANNELS"],
                                   dims["POOL_2_OUT_HEIGHT"],
                                   dims["POOL_2_OUT_WIDTH"]).astype(np.float32) / scale

    quantized_maps = {"conv1": conv_1, "pool1": pool_1, "conv2": conv_2, "pool2": pool_2}

    return int(predictions[0]), quantized_maps


def normalize_map(feature_map: np.ndarray) -> np.ndarray:
    """
    @brief Normalizes one 2D map to [0, 1] for visualization.

    @param feature_map Input 2D feature map.
    @return Normalized 2D map.
    """
    map_min = float(feature_map.min())
    map_max = float(feature_map.max())

    if map_max - map_min < 1e-8:
        return np.zeros_like(feature_map, dtype=np.float32)

    return (feature_map - map_min) / (map_max - map_min)


def build_channel_mosaic(feature_maps: np.ndarray, channel_gap: int = 2) -> np.ndarray:
    """
    @brief Builds a tiled mosaic image containing all channels of one layer.

    @param feature_maps Layer tensor with shape (C, H, W).
    @param channel_gap Number of pixels used as spacing between channel tiles.
    @return Tiled 2D mosaic image.
    """
    channels, height, width = feature_maps.shape
    grid_cols = int(np.ceil(np.sqrt(channels)))
    grid_rows = int(np.ceil(channels / grid_cols))

    mosaic_height = grid_rows * height + (grid_rows - 1) * channel_gap
    mosaic_width = grid_cols * width + (grid_cols - 1) * channel_gap
    # Fill gaps with NaN so they can be rendered with a visible separator color
    mosaic = np.full((mosaic_height, mosaic_width), np.nan, dtype=np.float32)

    for ch in range(channels):
        row = ch // grid_cols
        col = ch % grid_cols
        h_start = row * (height + channel_gap)
        w_start = col * (width + channel_gap)

        mosaic[h_start:h_start + height,
               w_start:w_start + width] = normalize_map(feature_maps[ch])

    return mosaic


def plot_feature_maps_comparison(original_maps: Dict[str, np.ndarray],
                                 quantized_maps: Dict[str, np.ndarray],
                                 original_input: np.ndarray,
                                 output_path: Path, label: int,
                                 prediction: int):
    """
    @brief Plots original and C-quantized feature maps side by side.

    @param original_maps Original model conv/pool feature maps.
    @param quantized_maps Quantized C model conv/pool feature maps.
    @param original_input Original input image map.
    @param output_path Output image path.
    @param label Ground-truth class label.
    @param prediction Predicted class from C model.
    """
    num_rows = len(LAYER_ORDER) + 1
    fig = plt.figure(figsize=(10, 3 * num_rows))
    grid = fig.add_gridspec(num_rows, 2)
    axes = np.empty((num_rows, 2), dtype=object)

    fig.suptitle(f"Feature Maps: label={label}, C prediction={prediction}", fontsize=14)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")

    def style_axis(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Single shared input image centered on top, since input is identical for both models
    input_ax = fig.add_subplot(grid[0, :])
    input_ax.imshow(normalize_map(original_input), cmap=cmap, interpolation="nearest")
    input_ax.set_title("Input")
    style_axis(input_ax)

    for row_idx, (layer_key, layer_title) in enumerate(LAYER_ORDER):
        original_mosaic = build_channel_mosaic(original_maps[layer_key])
        quantized_mosaic = build_channel_mosaic(quantized_maps[layer_key])
        ax_row = row_idx + 1

        axes[ax_row, 0] = fig.add_subplot(grid[ax_row, 0])
        axes[ax_row, 0].imshow(original_mosaic, cmap=cmap, interpolation="nearest")
        axes[ax_row, 0].set_title(f"Original - {layer_title}")
        style_axis(axes[ax_row, 0])

        axes[ax_row, 1] = fig.add_subplot(grid[ax_row, 1])
        axes[ax_row, 1].imshow(quantized_mosaic, cmap=cmap, interpolation="nearest")
        axes[ax_row, 1].set_title(f"Quantized C - {layer_title}")
        style_axis(axes[ax_row, 1])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def main(args):
    c_lib = load_c_lib(args.lib_path)
    setup_c_signature(c_lib)

    dims = load_params_dims(args.params_header)
    original_model = load_float_model(args.float_model)

    args.images_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.MNIST(root=str(args.data_dir),
                             train=False,
                             download=True,
                             transform=transforms.Compose([transforms.ToTensor(),
                                                           transforms.Normalize((0.5,), (0.5,))]))

    representative_indices = find_representative_indices(dataset)

    if len(representative_indices) < 10:
        raise RuntimeError("Could not find one representative sample for each label from 0 to 9.")

    with torch.no_grad():
        for label in range(10):
            sample_index = representative_indices[label]
            sample, _ = dataset[sample_index]
            original_input = sample.squeeze(0).cpu().numpy().astype(np.float32)

            original_maps = extract_original_feature_maps(sample, original_model)
            
            prediction, quantized_maps = run_c_convnet_forward(sample, c_lib, dims,
                                                               args.frac_bits)

            output_path = args.images_dir / f"feature_maps_{label}.png"

            plot_feature_maps_comparison(original_maps, quantized_maps,
                                         original_input,
                                         output_path, label, prediction)

            print(f"Saved: {output_path} (label={label}, prediction={prediction})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare original vs C quantized feature maps.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--lib_path", type=Path, default=Path("./lib/convnet.so"),
                        help="Path to compiled C shared library.")
    
    parser.add_argument("--params_header", type=Path, default=Path("./include/params.h"),
                        help="Path to generated params.h used to infer C tensor shapes.")
    
    parser.add_argument("--float_model", type=Path, default=Path("./models/model.pt"),
                        help="Path to floating-point model checkpoint.")
    
    parser.add_argument("--data_dir", type=Path, default=Path("./data"),
                        help="Directory used by torchvision MNIST dataset.")
    
    parser.add_argument("--images_dir", type=Path, default=Path("./images"),
                        help="Directory where feature-map comparison images are saved.")
    
    parser.add_argument("--frac_bits", type=int, default=16,
                        help="Fixed-point fractional bits used across C inference.")

    main(parser.parse_args())
