"""
Run C ConvNet inference from Python using ctypes.
"""
import argparse
import ctypes
from pathlib import Path

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import ConvNet


def load_c_lib(library_path: Path):
    try:
        return ctypes.CDLL(str(library_path.resolve()))
    except OSError as exc:
        raise RuntimeError(f"Unable to load C library: {library_path}") from exc


def ensure_contiguous(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array) if not array.flags["C_CONTIGUOUS"] else array


def run_convnet(sample: torch.Tensor, c_lib) -> int:
    x = sample.flatten().cpu().numpy().astype(np.intc)
    x = ensure_contiguous(x)

    pred = ensure_contiguous(np.zeros(1, dtype=np.uintc))

    c_int_p = ctypes.POINTER(ctypes.c_int)
    c_uint_p = ctypes.POINTER(ctypes.c_uint)

    c_run_convnet = c_lib.run_convnet
    c_run_convnet(x.ctypes.data_as(c_int_p), pred.ctypes.data_as(c_uint_p))

    return int(pred[0])


def validate_quant_model(quant_model_path: Path):
    if not quant_model_path.exists():
        raise FileNotFoundError(f"Quantized model not found: {quant_model_path}")

    saved_stats = torch.load(quant_model_path, map_location="cpu", weights_only=False)
    state_dict = saved_stats.get("state_dict")
    if state_dict is None or "layer_1_s_x" not in state_dict:
        raise KeyError("Expected key 'state_dict[\"layer_1_s_x\"]' in quantized model.")


def load_float_model(model_path: Path) -> ConvNet:
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


def main(args):
    c_lib = load_c_lib(args.lib_path)
    validate_quant_model(args.quant_model)
    original_model = load_float_model(args.float_model)

    c_int_p = ctypes.POINTER(ctypes.c_int)
    c_uint_p = ctypes.POINTER(ctypes.c_uint)
    c_lib.run_convnet.argtypes = (c_int_p, c_uint_p)
    c_lib.run_convnet.restype = None

    dataset = datasets.MNIST(
        root=str(args.data_dir),
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    total = 0
    correct_quantized = 0
    correct_original = 0
    same_prediction = 0
    quantized_only_correct = 0
    original_only_correct = 0

    with torch.no_grad():
        for samples, labels in data_loader:
            original_logits = original_model(samples)
            original_predictions = torch.argmax(original_logits, dim=1)

            # C-side quantize() expects fixed-point input and applies layer_1_s_x internally.
            samples_q = (samples * (1 << args.input_frac_bits)).round().to(torch.int32)

            batch_size = labels.shape[0]
            for idx in range(batch_size):
                quantized_prediction = run_convnet(samples_q[idx], c_lib)
                original_prediction = int(original_predictions[idx])
                label = int(labels[idx])

                is_quantized_correct = int(quantized_prediction == label)
                is_original_correct = int(original_prediction == label)

                correct_quantized += is_quantized_correct
                correct_original += is_original_correct
                same_prediction += int(quantized_prediction == original_prediction)
                quantized_only_correct += int(is_quantized_correct and not is_original_correct)
                original_only_correct += int(is_original_correct and not is_quantized_correct)
                total += 1

                if args.max_samples > 0 and total >= args.max_samples:
                    break

            if args.max_samples > 0 and total >= args.max_samples:
                break

    quantized_accuracy = 100.0 * correct_quantized / total if total > 0 else 0.0
    original_accuracy = 100.0 * correct_original / total if total > 0 else 0.0
    agreement = 100.0 * same_prediction / total if total > 0 else 0.0

    print(f"Samples: {total}")
    print("--- Accuracy Report ---")
    print(f"Original model correct: {correct_original}")
    print(f"Original model accuracy: {original_accuracy:.2f}%")
    print(f"Quantized model correct: {correct_quantized}")
    print(f"Quantized model accuracy: {quantized_accuracy:.2f}%")
    print("--- Comparison ---")
    print(f"Prediction agreement: {same_prediction} ({agreement:.2f}%)")
    print(f"Quantized only correct: {quantized_only_correct}")
    print(f"Original only correct: {original_only_correct}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Run quantized C ConvNet inference on MNIST test data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lib-path",
        type=Path,
        default=project_root / "lib" / "convnet.so",
        help="Path to compiled C shared library.",
    )
    parser.add_argument(
        "--quant-model",
        type=Path,
        default=project_root / "models" / "quantized.pt",
        help="Path to quantized model checkpoint (used for input scale).",
    )
    parser.add_argument(
        "--float-model",
        type=Path,
        default=project_root / "models" / "model.pt",
        help="Path to floating-point model checkpoint used by Python inference.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data",
        help="Directory used by torchvision MNIST dataset.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=1, help="DataLoader workers.")
    parser.add_argument(
        "--input-frac-bits",
        type=int,
        default=16,
        help="Fixed-point fractional bits for input representation passed to C.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="If > 0, evaluate only this many samples.",
    )

    main(parser.parse_args())
