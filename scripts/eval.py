import argparse
import ctypes
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import ConvNet
from torch.utils.data import DataLoader

# Use a non-interactive backend so plots can be generated in headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def ensure_contiguous(array: np.ndarray) -> np.ndarray:
    """
    @brief Ensures an array is C-contiguous.

    @param array Input NumPy array.
    @return C-contiguous array.
    """
    return np.ascontiguousarray(array) if not array.flags["C_CONTIGUOUS"] else array


def run_convnet(sample: torch.Tensor, c_lib) -> int:
    """
    @brief Runs one quantized C inference sample.

    @param sample Input sample tensor in fixed-point integer format.
    @param c_lib Loaded C shared library.
    @return Predicted class index.
    """
    # Flatten image tensor to match the C API expected 1D input layout
    x = sample.flatten().cpu().numpy().astype(np.intc)
    x = ensure_contiguous(x)

    # Allocate output buffer for the predicted class index
    pred = ensure_contiguous(np.zeros(1, dtype=np.uintc))

    # Mirror C pointer types used by run_convnet(const int*, unsigned int*)
    c_int_p = ctypes.POINTER(ctypes.c_int)
    c_uint_p = ctypes.POINTER(ctypes.c_uint)

    c_run_convnet = c_lib.run_convnet
    c_run_convnet(x.ctypes.data_as(c_int_p), pred.ctypes.data_as(c_uint_p))

    return int(pred[0])


def load_float_model(model_path: Path) -> ConvNet:
    """
    @brief Loads the floating-point model checkpoint.

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


def build_confusion_matrix(labels: np.ndarray, predictions: np.ndarray, num_classes: int):
    """
    @brief Builds a confusion matrix from labels and predictions.

    @param labels Ground-truth labels.
    @param predictions Predicted labels.
    @param num_classes Number of classes.
    @return Confusion matrix of shape num_classes x num_classes.
    """
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion_matrix, (labels, predictions), 1)

    return confusion_matrix


def compute_classification_metrics(confusion_matrix: np.ndarray):
    """
    @brief Computes common classification metrics from a confusion matrix.

    @param confusion_matrix Confusion matrix with rows=true and cols=predicted.
    @return Dictionary with accuracy, precision, recall and f1 averages.
    """
    # Rows are true labels and columns are predicted labels
    true_positives = np.diag(confusion_matrix).astype(np.float64)
    support = confusion_matrix.sum(axis=1).astype(np.float64)
    predicted_totals = confusion_matrix.sum(axis=0).astype(np.float64)

    false_positives = predicted_totals - true_positives
    false_negatives = support - true_positives

    # Use guarded division to avoid NaNs for classes with zero support/predictions
    precision = np.divide(true_positives,
                          true_positives + false_positives,
                          out=np.zeros_like(true_positives),
                          where=(true_positives + false_positives) != 0)
    
    recall = np.divide(true_positives,
                       true_positives + false_negatives,
                       out=np.zeros_like(true_positives),
                       where=(true_positives + false_negatives) != 0)
    
    f1_score = np.divide(2.0 * precision * recall,
                         precision + recall,
                         out=np.zeros_like(true_positives),
                         where=(precision + recall) != 0)

    total_samples = float(confusion_matrix.sum())
    accuracy = float(true_positives.sum() / total_samples) if total_samples > 0 else 0.0

    precision_macro = float(np.mean(precision))
    recall_macro = float(np.mean(recall))
    f1_macro = float(np.mean(f1_score))

    # Weighted metrics account for class imbalance via per-class support
    weighted_precision = (
        float(np.sum(precision * support) / total_samples) if total_samples > 0 else 0.0
    )

    weighted_recall = (
        float(np.sum(recall * support) / total_samples) if total_samples > 0 else 0.0
    )
    
    weighted_f1 = (
        float(np.sum(f1_score * support) / total_samples) if total_samples > 0 else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": weighted_precision,
        "recall_weighted": weighted_recall,
        "f1_weighted": weighted_f1,
    }


def draw_confusion_matrix(ax, confusion_matrix: np.ndarray, title: str):
    """
    @brief Draws one confusion matrix on a given axis.

    @param ax Matplotlib axis where the matrix is drawn.
    @param confusion_matrix Confusion matrix to plot.
    @param title Plot title.
    """
    image = ax.imshow(confusion_matrix, cmap="Blues")

    num_classes = confusion_matrix.shape[0]
    ticks = np.arange(num_classes)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    # Dynamic text color keeps counts readable over light and dark cells
    max_value = confusion_matrix.max() if confusion_matrix.size > 0 else 0
    threshold = max_value * 0.5

    for row_idx in range(num_classes):
        for col_idx in range(num_classes):
            color = "white" if confusion_matrix[row_idx, col_idx] > threshold else "black"
            ax.text(col_idx, row_idx, str(confusion_matrix[row_idx, col_idx]),
                    ha="center", va="center", color=color, fontsize=8)

    return image


def plot_confusion_matrices(original_confusion: np.ndarray, quantized_confusion: np.ndarray,
                            output_path: Path):
    """
    @brief Plots and saves both confusion matrices in a single image.

    @param original_confusion Confusion matrix of original model.
    @param quantized_confusion Confusion matrix of quantized model.
    @param output_path Output image path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    original_image = draw_confusion_matrix(axes[0],
                                           original_confusion,
                                           "Original Model Confusion Matrix")
    
    quantized_image = draw_confusion_matrix(axes[1],
                                            quantized_confusion,
                                            "Quantized Model Confusion Matrix")

    fig.colorbar(original_image, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(quantized_image, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def print_evaluation_report(total: int, correct_original: int,
                            original_accuracy: float, correct_quantized: int,
                            quantized_accuracy: float, same_prediction: int,
                            agreement: float, quantized_only_correct: int,
                            original_only_correct: int, original_metrics: dict,
                            quantized_metrics: dict,
                            confusion_matrices_path: Path):
    """
    @brief Prints the complete evaluation report in one place.

    @param total Number of evaluated samples.
    @param correct_original Number of correct predictions from original model.
    @param original_accuracy Original model accuracy in percentage.
    @param correct_quantized Number of correct predictions from quantized model.
    @param quantized_accuracy Quantized model accuracy in percentage.
    @param same_prediction Number of equal predictions between both models.
    @param agreement Prediction agreement in percentage.
    @param quantized_only_correct Correct only by quantized model.
    @param original_only_correct Correct only by original model.
    @param original_metrics Metrics dictionary for original model.
    @param quantized_metrics Metrics dictionary for quantized model.
    @param confusion_matrices_path Path to combined confusion matrices image.
    """
    print(f"Samples: {total}")
    print("\n")
    print("--- Accuracy Report ---")
    print(f"Original model correct: {correct_original}")
    print(f"Original model accuracy: {original_accuracy:.2f}%")
    print(f"Quantized model correct: {correct_quantized}")
    print(f"Quantized model accuracy: {quantized_accuracy:.2f}%")
    print("\n")
    print("--- Comparison ---")
    print(f"Prediction agreement: {same_prediction} ({agreement:.2f}%)")
    print(f"Quantized only correct: {quantized_only_correct}")
    print(f"Original only correct: {original_only_correct}")
    print("\n")

    metric_rows = [
        ("Accuracy", "accuracy"),
        ("Precision (macro)", "precision_macro"),
        ("Recall (macro)", "recall_macro"),
        ("F1-score (macro)", "f1_macro"),
        ("Precision (weighted)", "precision_weighted"),
        ("Recall (weighted)", "recall_weighted"),
        ("F1-score (weighted)", "f1_weighted"),
    ]

    print("--- Classification Metrics Matrix (%) ---")
    print(f"{'Metric':<24}{'Original':>12}{'Quantized':>14}")
    print("-" * 50)

    for metric_name, metric_key in metric_rows:
        original_value = original_metrics[metric_key] * 100.0
        quantized_value = quantized_metrics[metric_key] * 100.0
        print(f"{metric_name:<24}{original_value:>12.2f}{quantized_value:>14.2f}")

    print("\n")
    print(f"Confusion matrices saved: {confusion_matrices_path}")


def main(args):
    c_lib = load_c_lib(args.lib_path)
    original_model = load_float_model(args.float_model)

    # Declare C function signature for safe ctypes calls
    c_int_p = ctypes.POINTER(ctypes.c_int)
    c_uint_p = ctypes.POINTER(ctypes.c_uint)
    c_lib.run_convnet.argtypes = (c_int_p, c_uint_p)
    c_lib.run_convnet.restype = None

    dataset = datasets.MNIST(root=str(args.data_dir),
                             train=False,
                             download=True,
                             transform=transforms.Compose([transforms.ToTensor(),
                                                           transforms.Normalize((0.5,), (0.5,))]))
    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers)

    # Ensure output directory exists before saving figures
    args.images_dir.mkdir(parents=True, exist_ok=True)

    # Track accuracy, agreement, and disagreement buckets between both models
    total = 0
    correct_quantized = 0
    correct_original = 0
    same_prediction = 0
    quantized_only_correct = 0
    original_only_correct = 0

    # Save all predictions to build confusion matrices at the end
    all_labels = []
    all_original_predictions = []
    all_quantized_predictions = []

    with torch.no_grad():
        for samples, labels in data_loader:
            original_logits = original_model(samples)
            original_predictions = torch.argmax(original_logits, dim=1)

            # Convert normalized float input to fixed-point integer format expected by C
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

                all_labels.append(label)
                all_original_predictions.append(original_prediction)
                all_quantized_predictions.append(quantized_prediction)

                # Stop early when max_samples is set for quick checks
                if args.max_samples > 0 and total >= args.max_samples:
                    break

            if args.max_samples > 0 and total >= args.max_samples:
                break

    # Convert lists to arrays for vectorized confusion matrix and metrics computation
    labels_array = np.asarray(all_labels, dtype=np.int64)
    original_predictions_array = np.asarray(all_original_predictions, dtype=np.int64)
    quantized_predictions_array = np.asarray(all_quantized_predictions, dtype=np.int64)

    num_classes = 10

    original_confusion = build_confusion_matrix(labels_array,
                                                original_predictions_array,
                                                num_classes)
    
    quantized_confusion = build_confusion_matrix(labels_array,
                                                 quantized_predictions_array,
                                                 num_classes)

    original_metrics = compute_classification_metrics(original_confusion)
    quantized_metrics = compute_classification_metrics(quantized_confusion)

    quantized_accuracy = 100.0 * correct_quantized / total if total > 0 else 0.0
    original_accuracy = 100.0 * correct_original / total if total > 0 else 0.0
    agreement = 100.0 * same_prediction / total if total > 0 else 0.0

    confusion_matrices_path = args.images_dir / "confusion_matrices.png"

    # Save both confusion matrices into a single side-by-side image
    plot_confusion_matrices(original_confusion,
                            quantized_confusion,
                            confusion_matrices_path)

    print_evaluation_report(total, correct_original, original_accuracy, correct_quantized,
                            quantized_accuracy, same_prediction, agreement,
                            quantized_only_correct, original_only_correct,
                            original_metrics, quantized_metrics,
                            confusion_matrices_path)


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(description="Run quantized C ConvNet inference "
                                                 "on MNIST test data.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--lib_path", type=Path, default=project_root / "lib" / "convnet.so",
                        help="Path to compiled C shared library.")
    
    parser.add_argument("--float_model", type=Path, default=project_root / "models" / "model.pt",
                        help="Path to floating-point model checkpoint used by Python inference.")
    
    parser.add_argument("--data_dir", type=Path, default=project_root / "data",
                        help="Directory used by torchvision MNIST dataset.")
    
    parser.add_argument("--images_dir", type=Path, default=project_root / "images",
                        help="Directory where confusion matrix plots are saved.")
    
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")

    parser.add_argument("--num_workers", type=int, default=1, help="DataLoader workers.")

    parser.add_argument("--input_frac_bits", type=int, default=16,
                        help="Fixed-point fractional bits for input representation passed to C.")
    
    parser.add_argument("--max_samples", type=int, default=0,
                        help="If > 0, evaluate only this many samples.")

    main(parser.parse_args())
