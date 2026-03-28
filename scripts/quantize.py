import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.ao.quantization.observer import HistogramObserver
from torch.utils.data import DataLoader

from model import ConvNet


def get_quantized_layers(model: nn.Module):
    """
    @brief Returns the layers and metadata used by export.

    @param model Trained floating-point model.
    @return Ordered list of tuples with layer index, layer type, and module reference.
    """
    # Layer indices must match the C-side layer_1..layer_5 naming convention
    return [(1, 'conv', model.convolutional_layers[0]),
            (2, 'conv', model.convolutional_layers[3]),
            (3, 'linear', model.linear_layers[0]),
            (4, 'linear', model.linear_layers[2]),
            (5, 'linear', model.linear_layers[4])]


def run_calibration_pass(model: nn.Module, data_loader: DataLoader,
                         hook_modules: dict, hook_callback):
    """
    @brief Runs one forward pass over calibration data with layer hooks.

    @param model Model used for calibration.
    @param data_loader Calibration data loader.
    @param hook_modules Mapping layer index -> module to hook.
    @param hook_callback Callback receiving layer index and input tensor.
    """
    handles = []

    for layer_idx, module in hook_modules.items():
        def pre_hook(_module, inputs, idx=layer_idx):
            # Capture layer input activations, which are what C quantize() uses
            hook_callback(idx, inputs[0].detach())

        handles.append(module.register_forward_pre_hook(pre_hook))

    model.eval()

    # Calibration only collects activation statistics, no gradients/updates
    with torch.no_grad():
        for batch, _ in data_loader:
            model(batch.float())

    for handle in handles:
        handle.remove()


def collect_amax_hist_observer(model: nn.Module, data_loader: DataLoader,
                               layer_modules: dict, num_bins: int):
    """
    @brief Collects activation amax values with PyTorch HistogramObserver.

    @param model Model used for calibration.
    @param data_loader Calibration data loader.
    @param layer_modules Mapping layer index -> module.
    @param num_bins Number of histogram bins.
    @return Mapping layer index -> activation amax.
    """
    observers = {}

    for layer_idx in layer_modules:
        observers[layer_idx] = HistogramObserver(dtype=torch.qint8,
                                                 qscheme=torch.per_tensor_symmetric,
                                                 bins=num_bins)

    def hook_callback(layer_idx: int, tensor: torch.Tensor):
        observers[layer_idx](tensor.cpu())

    run_calibration_pass(model, data_loader, layer_modules, hook_callback)

    activation_amax = {}

    for layer_idx, observer in observers.items():
        scale, _ = observer.calculate_qparams()
        scale_value = float(scale.item())
        # For symmetric int8, amax ~= scale * 127
        activation_amax[layer_idx] = max(scale_value * 127.0, 1e-12)

    return activation_amax


def compute_kl_amax_from_histogram(histogram: np.ndarray, max_abs: float,
                                   num_quantized_bins: int = 128):
    """
    @brief Computes KL-divergence threshold from an absolute-value histogram.

    @param histogram Absolute-value histogram counts.
    @param max_abs Maximum absolute value represented by the histogram.
    @param num_quantized_bins Number of bins used by the simulated quantized distribution.
    @return Estimated amax threshold.
    """
    if max_abs <= 0.0:
        return 1e-12

    hist = histogram.astype(np.float64)
    num_bins = hist.shape[0]

    if np.sum(hist) == 0.0:
        return max_abs

    start_bin = max(num_quantized_bins, 2)
    best_kl = np.inf
    best_threshold_bin = num_bins - 1

    for threshold_bin in range(start_bin, num_bins + 1):
        sliced = hist[:threshold_bin].copy()
        # Fold right-tail outliers into the last kept bin
        sliced[-1] += np.sum(hist[threshold_bin:])

        total = np.sum(sliced)
        if total <= 0.0:
            continue

        p = sliced / total

        quantized = np.zeros(num_quantized_bins, dtype=np.float64)
        for q_idx in range(num_quantized_bins):
            # Merge original bins into quantized bins
            start = int(np.floor(q_idx * threshold_bin / num_quantized_bins))
            end = int(np.floor((q_idx + 1) * threshold_bin / num_quantized_bins))
            if end <= start:
                end = start + 1
            quantized[q_idx] = np.sum(p[start:end])

        q = np.zeros_like(p)
        for q_idx in range(num_quantized_bins):
            # Redistribute each quantized bin uniformly back to original resolution
            start = int(np.floor(q_idx * threshold_bin / num_quantized_bins))
            end = int(np.floor((q_idx + 1) * threshold_bin / num_quantized_bins))
            if end <= start:
                end = start + 1

            if quantized[q_idx] > 0.0:
                q[start:end] = quantized[q_idx] / (end - start)

        p_nonzero = p > 0.0
        q_nonzero = np.where(q > 0.0, q, 1e-12)
        # Choose threshold minimizing KL divergence between p and reconstructed q
        kl_value = np.sum(p[p_nonzero] * np.log(p[p_nonzero] / q_nonzero[p_nonzero]))

        if kl_value < best_kl:
            best_kl = kl_value
            best_threshold_bin = threshold_bin

    return max_abs * (float(best_threshold_bin) / float(num_bins))


def collect_amax_kl_entropy(model: nn.Module, data_loader: DataLoader,
                            layer_modules: dict, num_bins: int):
    """
    @brief Collects activation amax values using custom KL/entropy calibration.

    @param model Model used for calibration.
    @param data_loader Calibration data loader.
    @param layer_modules Mapping layer index -> module.
    @param num_bins Number of bins in calibration histograms.
    @return Mapping layer index -> activation amax.
    """
    # Collect max abs for stable histogram ranges
    layer_max = {layer_idx: 0.0 for layer_idx in layer_modules}

    def collect_max(layer_idx: int, tensor: torch.Tensor):
        current_max = float(tensor.abs().max().item())
        if current_max > layer_max[layer_idx]:
            layer_max[layer_idx] = current_max

    run_calibration_pass(model, data_loader, layer_modules, collect_max)

    # Collect absolute-value histograms with fixed [0, max_abs] ranges
    layer_hists = {layer_idx: np.zeros(num_bins, dtype=np.float64) for layer_idx in layer_modules}

    def collect_hist(layer_idx: int, tensor: torch.Tensor):
        max_abs = layer_max[layer_idx]
        if max_abs <= 0.0:
            return

        abs_tensor = tensor.abs().cpu().flatten()
        hist = torch.histc(abs_tensor, bins=num_bins, min=0.0, max=max_abs)
        layer_hists[layer_idx] += hist.numpy().astype(np.float64)

    run_calibration_pass(model, data_loader, layer_modules, collect_hist)

    activation_amax = {}

    for layer_idx in layer_modules:
        max_abs = layer_max[layer_idx]
        if max_abs <= 0.0:
            activation_amax[layer_idx] = 1e-12
            continue

        # Convert histogram to a KL-selected clipping threshold
        activation_amax[layer_idx] = max(compute_kl_amax_from_histogram(layer_hists[layer_idx],
                                                                         max_abs),
                                         1e-12)

    return activation_amax


def collect_amax_max_value(model: nn.Module, data_loader: DataLoader,
                           layer_modules: dict):
    """
    @brief Collects activation amax values using max-value calibration.

    @param model Model used for calibration.
    @param data_loader Calibration data loader.
    @param layer_modules Mapping layer index -> module.
    @return Mapping layer index -> activation amax.
    """
    layer_max = {layer_idx: 0.0 for layer_idx in layer_modules}

    def collect_max(layer_idx: int, tensor: torch.Tensor):
        current_max = float(tensor.abs().max().item())
        if current_max > layer_max[layer_idx]:
            layer_max[layer_idx] = current_max

    run_calibration_pass(model, data_loader, layer_modules, collect_max)

    return {layer_idx: max(max_abs, 1e-12) for layer_idx, max_abs in layer_max.items()}


def compute_percentile_amax_from_histogram(histogram: np.ndarray, max_abs: float,
                                           percentile: float):
    """
    @brief Computes amax from an absolute-value histogram percentile threshold.

    @param histogram Absolute-value histogram counts.
    @param max_abs Maximum absolute value represented by the histogram.
    @param percentile Percentile in (0, 100].
    @return Estimated amax threshold.
    """
    if max_abs <= 0.0:
        return 1e-12

    hist = histogram.astype(np.float64)
    total = np.sum(hist)

    if total <= 0.0:
        return max_abs

    pct = float(np.clip(percentile, 1e-6, 100.0))
    target = (pct / 100.0) * total
    cdf = np.cumsum(hist)
    threshold_bin = int(np.searchsorted(cdf, target, side='left'))
    threshold_bin = min(max(threshold_bin, 0), hist.shape[0] - 1)

    # Convert selected bin index to its upper-edge value in [0, max_abs]
    return max_abs * (float(threshold_bin + 1) / float(hist.shape[0]))


def collect_amax_percentile(model: nn.Module, data_loader: DataLoader,
                            layer_modules: dict, num_bins: int, percentile: float):
    """
    @brief Collects activation amax values using percentile calibration.

    @param model Model used for calibration.
    @param data_loader Calibration data loader.
    @param layer_modules Mapping layer index -> module.
    @param num_bins Number of bins in calibration histograms.
    @param percentile Percentile used to choose clipping threshold.
    @return Mapping layer index -> activation amax.
    """
    # Collect max abs for stable histogram ranges
    layer_max = {layer_idx: 0.0 for layer_idx in layer_modules}

    def collect_max(layer_idx: int, tensor: torch.Tensor):
        current_max = float(tensor.abs().max().item())
        if current_max > layer_max[layer_idx]:
            layer_max[layer_idx] = current_max

    run_calibration_pass(model, data_loader, layer_modules, collect_max)

    # Collect absolute-value histograms with fixed [0, max_abs] ranges
    layer_hists = {layer_idx: np.zeros(num_bins, dtype=np.float64) for layer_idx in layer_modules}

    def collect_hist(layer_idx: int, tensor: torch.Tensor):
        max_abs = layer_max[layer_idx]
        if max_abs <= 0.0:
            return

        abs_tensor = tensor.abs().cpu().flatten()
        hist = torch.histc(abs_tensor, bins=num_bins, min=0.0, max=max_abs)
        layer_hists[layer_idx] += hist.numpy().astype(np.float64)

    run_calibration_pass(model, data_loader, layer_modules, collect_hist)

    activation_amax = {}

    for layer_idx in layer_modules:
        max_abs = layer_max[layer_idx]
        if max_abs <= 0.0:
            activation_amax[layer_idx] = 1e-12
            continue

        activation_amax[layer_idx] = max(compute_percentile_amax_from_histogram(
            layer_hists[layer_idx], max_abs, percentile), 1e-12)

    return activation_amax


def compute_weight_amax(weight: torch.Tensor):
    """
    @brief Computes per-output-channel amax for conv/linear weights.

    @param weight Weight tensor.
    @return Amax tensor broadcastable over weight dimensions.
    """
    # Per-output-channel amax matches current C layer scaling design
    reduce_dims = tuple(range(1, weight.dim()))
    return weight.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-12)


def quantize_model_params(model: nn.Module, activation_amax: dict):
    """
    @brief Quantizes model weights and exports fixed-point scaling constants.

    @param model Trained model.
    @param activation_amax Mapping layer index -> calibrated activation amax.
    @return Dictionary with quantized weights and scaling constants.
    """
    # int8 symmetric quantization range used by C implementation
    scale_factor = 127.0
    state_dict = dict()

    for layer_idx, layer_type, module in get_quantized_layers(model):
        weight = module.weight.detach().cpu()
        s_w = compute_weight_amax(weight)
        s_x = max(float(activation_amax[layer_idx]), 1e-12)

        scale = weight * (scale_factor / s_w)
        quantized_weight = torch.clamp(scale.round(), min=-127, max=127).to(torch.int32)

        if layer_type == 'linear':
            # Linear weights are transposed to match C mat_mult memory layout
            quantized_weight = quantized_weight.T

        # Export tensors and scale constants with existing checkpoint keys
        state_dict[f'layer_{layer_idx}_weight'] = quantized_weight.numpy()
        state_dict[f'layer_{layer_idx}_s_x'] = scale_factor / s_x
        state_dict[f'layer_{layer_idx}_s_x_inv'] = s_x / scale_factor
        state_dict[f'layer_{layer_idx}_s_w_inv'] = (s_w / scale_factor).squeeze().numpy()

    return state_dict


def build_calibration_loader(data_dir: Path):
    """
    @brief Builds MNIST calibration data loader.

    @param data_dir Directory containing MNIST data.
    @return DataLoader for calibration.
    """
    # Keep preprocessing aligned with train/eval scripts
    dataset = datasets.MNIST(root=data_dir,
                             train=True,
                             download=True,
                             transform=transforms.Compose([transforms.Grayscale(),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize((0.5,), (0.5,))]))

    # Full-dataset single batch reproduces the original calibration workflow
    return DataLoader(dataset, batch_size=len(dataset), num_workers=1, shuffle=False)


def calibrate_activation_amax(model: nn.Module, data_loader: DataLoader,
                              calibrator: str, num_bins: int, percentile: float):
    """
    @brief Calibrates activation amax values with a selected calibrator.

    @param model Model used for calibration.
    @param data_loader Calibration data loader.
    @param calibrator Selected calibrator method.
    @param num_bins Number of histogram bins.
    @param percentile Percentile used by percentile calibrator.
    @return Mapping layer index -> activation amax.
    """
    layer_modules = {}

    for layer_idx, _, module in get_quantized_layers(model):
        layer_modules[layer_idx] = module

    # Dispatch to selected calibration strategy
    if calibrator == 'histogram_observer':
        return collect_amax_hist_observer(model, data_loader, layer_modules, num_bins)

    if calibrator == 'kl_entropy':
        return collect_amax_kl_entropy(model, data_loader, layer_modules, num_bins)

    if calibrator == 'max_value':
        return collect_amax_max_value(model, data_loader, layer_modules)

    if calibrator == 'percentile':
        return collect_amax_percentile(model, data_loader, layer_modules, num_bins, percentile)

    raise ValueError(f'Unsupported calibrator: {calibrator}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for post-training quantization of a pre-trained model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--filename', type=str, default='model.pt',
                        help='Trained checkpoint filename inside --save_dir.')
    
    parser.add_argument('--num_bins', type=int, default=128,
                        help='Number of bins used for histogram-based calibration.')
    
    parser.add_argument('--calibrator', type=str,
                        choices=['histogram_observer', 'kl_entropy', 'max_value', 'percentile'],
                        default='histogram_observer',
                        help='Calibration strategy used to estimate activation amax.')

    parser.add_argument('--percentile', type=float, default=99.9,
                        help='Percentile used when --calibrator=percentile.')
    
    parser.add_argument('--data_dir', default='./data', type=Path,
                        help='Directory where MNIST is downloaded/read for calibration.')
    
    parser.add_argument('--save_dir', default='./models', type=Path,
                        help='Directory containing input checkpoint and output quantized file.')

    args = parser.parse_args()

    # Load trained floating-point checkpoint
    saved_stats = torch.load(os.path.join(args.save_dir, args.filename))
    state_dict = saved_stats['state_dict']

    model = ConvNet(h=28, w=28, inputs=1, outputs=10)
    model.load_state_dict(state_dict)
    model.eval()

    # Calibrate activation amax values and export quantized parameters
    calibration_loader = build_calibration_loader(args.data_dir)

    activation_amax = calibrate_activation_amax(model, calibration_loader,
                                                args.calibrator, args.num_bins,
                                                args.percentile)

    quantized_state_dict = quantize_model_params(model, activation_amax)
    saved_stats['state_dict'] = quantized_state_dict

    torch.save(saved_stats, os.path.join(args.save_dir, 'quantized.pt'))
