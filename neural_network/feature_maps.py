import argparse
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from model import ConvNet


def visualize_all_layers(layers_data: List[Tuple[Union[int, str],str,torch.Tensor]]) -> None:
    """
    @brief Visualize feature maps from all collected layers in a single figure.

    This function arranges feature maps in a grid where each row corresponds to a layer and each 
    column corresponds to a channel of that layer's output.

    @param layers_data A list of tuples containing: layer index or identifier, layer name, and a
                       feature map tensor with shape (1, C, H, W)
    @return None
    """
    num_layers = len(layers_data)
    if num_layers == 0:
        return

    # Determine the maximum number of channels across all layers
    max_channels = max(data[2].shape[1] for data in layers_data)

    # Create a subplot grid: rows = layers, columns = channels
    fig, axes = plt.subplots(num_layers, max_channels,
                             figsize=(max_channels, num_layers * 1.5))
    
    fig.suptitle("Feature Maps across Layers", fontsize=16)

    # Ensure axes is always a 2D array for consistent indexing
    if num_layers == 1:
        axes = np.array([axes])
    if max_channels == 1:
        axes = np.expand_dims(axes, axis=1)

    for row_idx, (layer_idx, layer_name, feature_maps) in enumerate(layers_data):
        # Remove batch dimension and convert tensor to NumPy
        fmaps = feature_maps.squeeze(0).cpu().detach().numpy()
        num_channels = fmaps.shape[0]

        # Add layer label to the first column
        if isinstance(layer_idx, int):
            label = f"L{layer_idx}\n{layer_name}"
        else:
            label = layer_name

        axes[row_idx, 0].set_ylabel(label, rotation=0, labelpad=40, fontsize=10, va="center")

        for col_idx in range(max_channels):
            ax = axes[row_idx, col_idx]

            if col_idx < num_channels:
                fm = fmaps[col_idx]

                # Normalize feature map for visualization
                fm_min, fm_max = fm.min(), fm.max()
                if fm_max - fm_min > 1e-5:
                    fm = (fm - fm_min) / (fm_max - fm_min)

                ax.imshow(fm, cmap="viridis")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                # Hide unused subplot cells
                ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, top=0.92)
    plt.show()


def main() -> None:
    """
    @brief Entry point for feature map visualization.

    This function loads a trained CNN model, selects a sample from the MNIST
    test dataset, extracts intermediate feature maps, and visualizes them.

    @return None
    """
    parser = argparse.ArgumentParser(description="Visualize feature maps from the trained model.")
    
    parser.add_argument("--model_path", type=Path, default=Path("./models/model.pt"),
                        help="Path to the trained model checkpoint.")
    
    parser.add_argument("--dataset_path", type=Path, default=Path("./data"),
                        help="Path to the MNIST dataset.")
    
    parser.add_argument("--image_index", type=int, default=0,
                        help="Index of the image in the test dataset to use.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define dataset preprocessing
    transform = transforms.Compose([transforms.Grayscale(), 
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # Load MNIST test dataset
    test_dataset = datasets.MNIST(root=args.dataset_path, 
                                  train=False, 
                                  download=True, 
                                  transform=transform)

    if args.image_index >= len(test_dataset):
        print(f"Error: Image index {args.image_index} is out of bounds.")
        return

    # Retrieve selected image and label
    img_tensor, label = test_dataset[args.image_index]
    print(f"Selected Image Index: {args.image_index}, Label: {label}")

    # Add batch dimension: (1, 1, 28, 28)
    input_batch = img_tensor.unsqueeze(0).to(device)

    # Initialize model and load weights
    model = ConvNet(h=28, w=28, inputs=1, outputs=10).to(device)

    if args.model_path.exists():
        checkpoint = torch.load(args.model_path, map_location=device)
        state_dict = checkpoint["state_dict"]
        model.load_state_dict(state_dict)
        print(f"Loaded model from {args.model_path}")
    else:
        print("Model not found, using random weights.")

    model.eval()

    # Forward pass through convolutional layers
    x = input_batch
    
    print("Extracting feature maps...")

    layers_data = [("Input", "Input Image", x)]
    
    for i, layer in enumerate(model.convolutional_layers):
        x = layer(x)
        
        if isinstance(layer, (nn.Conv2d, nn.ReLU, nn.MaxPool2d)):
            layers_data.append((i, layer.__class__.__name__, x))

    visualize_all_layers(layers_data)


if __name__ == "__main__":
    main()
