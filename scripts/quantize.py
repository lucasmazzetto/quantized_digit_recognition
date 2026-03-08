import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from pytorch_quantization import calib
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor

from model import ConvNet


def collect_stats(model:nn.Module, data_loader: DataLoader, num_bins:int):
    """
    @brief Feed data to the network and collect calibration statistics.

    @param model The model to calibrate.
    @param data_loader The data loader used for calibration.
    @param num_bins Number of histogram bins.
    @return None.
    """
    model.eval()

    # Enable calibrators
    for _, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            # Disable quantization to construct a histogram from the float values
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
                if isinstance(module._calibrator, calib.HistogramCalibrator):
                    module._calibrator._num_bins = num_bins
            else:
                module.disable()

    # Perform inference and re-enable quantization for cases where we want
    # to run inference with a quantized model.
    for batch, _ in data_loader:
        x = batch.float()
        model(x)

        # Disable calibrators
        for _, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()


def compute_amax(model:nn.Module, **kwargs):
    """
    @brief Compute and load amax values for all quantizers.

    @param model The model containing quantizers.
    @param kwargs Extra keyword arguments passed to load_calib_amax.
    @return None.
    """
    # Load calibration results
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(f'{name:40}: {module}')


def quantize_model_params(model:nn.Module):
    """
    @brief Quantize layer weights and prepare scale constants for C-code.

    @param model Trained and calibrated model.
    @return Dictionary with quantized weights and scaling constants.
    """
    # To minimize the number of operations at inference,
    # we calculate the scale factor values offline.

    # To avoid division in fixed-point, we can invert them
    # and multiply the inverted values in C.
    scale_factor = 127  # 127 for 8 bits
    state_dict = dict()

    convolutional_layers = [0, 3]
    linear_layers = [0, 2, 4]

    layer_idx = 1

    # Quantize the convolutional layers
    for idx in convolutional_layers:
        weight = model.state_dict()[f'convolutional_layers.{idx}.weight']
        s_w = model.state_dict()[f'convolutional_layers.{idx}._weight_quantizer._amax'].numpy()
        s_x = model.state_dict()[f'convolutional_layers.{idx}._input_quantizer._amax'].numpy()

        scale = weight * (scale_factor / s_w)
        state_dict[f'layer_{layer_idx}_weight'] = torch.clamp(
            scale.round(), min=-127, max=127
        ).to(int)
        state_dict[f'layer_{layer_idx}_weight'] = state_dict[f'layer_{layer_idx}_weight'].numpy()
        state_dict[f'layer_{layer_idx}_s_x'] = scale_factor / s_x
        state_dict[f'layer_{layer_idx}_s_x_inv'] = s_x / scale_factor
        state_dict[f'layer_{layer_idx}_s_w_inv'] = (s_w / scale_factor).squeeze()
        layer_idx += 1

    # Quantize the linear layers
    for idx in linear_layers:
        weight = model.state_dict()[f'linear_layers.{idx}.weight']
        s_w = model.state_dict()[f'linear_layers.{idx}._weight_quantizer._amax'].numpy()
        s_x = model.state_dict()[f'linear_layers.{idx}._input_quantizer._amax'].numpy()

        scale = weight * (scale_factor / s_w)
        state_dict[f'layer_{layer_idx}_weight'] = torch.clamp(scale.round(), 
                                                              min=-127, max=127).to(int)

        state_dict[f'layer_{layer_idx}_weight'] = state_dict[f'layer_{layer_idx}_weight'].T

        state_dict[f'layer_{layer_idx}_weight'] = state_dict[f'layer_{layer_idx}_weight'].numpy()
        state_dict[f'layer_{layer_idx}_s_x'] = scale_factor / s_x
        state_dict[f'layer_{layer_idx}_s_x_inv'] = s_x / scale_factor
        state_dict[f'layer_{layer_idx}_s_w_inv'] = (s_w / scale_factor).squeeze()
        layer_idx += 1

    return state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for post-training quantization of a pre-trained model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--filename', type=str, default='model.pt',
                        help='Trained checkpoint filename inside --save_dir.')
    parser.add_argument('--num_bins', type=int, default=128,
                        help='Number of bins used by histogram calibration.')
    parser.add_argument('--data_dir', default='./data', type=Path,
                        help='Directory where MNIST is downloaded/read for calibration.')
    parser.add_argument('--save_dir', default='./models', type=Path,
                        help='Directory containing the input checkpoint and output quantized file.')

    args = parser.parse_args()

    # Load model
    saved_stats = torch.load(os.path.join(args.save_dir, args.filename))
    state_dict = saved_stats['state_dict']

    # Define quantization using a histogram calibrator.
    quant_nn.QuantLinear.set_default_quant_desc_input(QuantDescriptor(calib_method='histogram'))
    quant_nn.QuantConv2d.set_default_quant_desc_input(QuantDescriptor(calib_method='histogram'))

    # Monkey-patch PyTorch modules with quantized versions by calling initialize()
    quant_modules.initialize()

    # Load trained model
    model = ConvNet(h=28, w=28, inputs=1, outputs=10)
    model.load_state_dict(state_dict)

    # Use the training split from MNIST, but without augmentation.
    dataset = datasets.MNIST(
        root=args.data_dir, train=True,
        download=True, transform=transforms.Compose([transforms.Grayscale(),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5,), (0.5,))]))

    data_loader = DataLoader(dataset, batch_size=len(dataset), num_workers=1, shuffle=False)

    # It is a bit slow since we collect histograms on CPU
    with torch.no_grad():
        collect_stats(model, data_loader, args.num_bins)
        compute_amax(model, method='entropy')

    state_dict = quantize_model_params(model)
    saved_stats['state_dict'] = state_dict

    torch.save(saved_stats, os.path.join(args.save_dir, 'quantized.pt'))
