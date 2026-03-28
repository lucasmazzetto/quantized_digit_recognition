import os
import argparse
import multiprocessing
from pathlib import Path

import matplotlib
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import ConvNet
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

# Use a non-interactive backend so plots can be generated in headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def train_epoch(model:nn.Module, data_loader: DataLoader, 
                optimizer: Adam, loss_fn:nn.CrossEntropyLoss):
    """
    @brief Trains the model for one epoch.

    @param model The neural network model to train.
    @param data_loader The data loader containing the training data.
    @param optimizer The optimizer used to update model parameters.
    @param loss_fn The loss function used to compute the training loss.
    @return The average training loss for the epoch.
    """
    # Set the model to training mode
    model.train(mode=True)
    num_batches = len(data_loader)

    loss = 0
    
    for x, y in data_loader:
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(x)

        batch_loss = loss_fn(outputs, y)

        # Backpropagation
        batch_loss.backward()
        
        # Update model parameters
        optimizer.step()

        loss += batch_loss.item()
        
    return loss / num_batches


def eval_epoch(model: nn.Module, data_loader: DataLoader, loss_fn: nn.CrossEntropyLoss):
    """
    @brief Evaluates the model on the validation set.

    @param model The neural network model to evaluate.
    @param data_loader The data loader containing the validation data.
    @param loss_fn The loss function used to compute the validation loss.
    @return The average validation loss for the epoch.
    """
    # Set the model to evaluation mode
    model.eval()
    num_batches = len(data_loader)

    loss = 0
    
    # Disable gradient calculation
    with torch.no_grad():
        for x, y in data_loader:
            pred_y = model(x)
            batch_loss = loss_fn(pred_y, y)
            loss += batch_loss.item()
            
    return loss / num_batches


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
          optimizer: Adam, loss_fn: nn.CrossEntropyLoss, num_epochs: int,
          plot_path: Path):
    """
    @brief Runs the training loop for a specified number of epochs.

    @param model The neural network model to train.
    @param train_loader The data loader containing the training data.
    @param val_loader The data loader containing the validation data.
    @param optimizer The optimizer used to update model parameters.
    @param loss_fn The loss function used to compute the loss.
    @param num_epochs The number of epochs to train the model.
    @param plot_path Output path used to refresh the training loss plot every epoch.
    @return Lists containing train and validation loss per epoch.
    """
    print(f'Starting training for {num_epochs} epochs...')

    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
        val_loss = eval_epoch(model, val_loader, loss_fn)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Update training.png at each epoch with the curve accumulated so far
        plot_training_loss(train_losses, val_losses, plot_path)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.5f} | Validation Loss: {val_loss:.5f}")

    return train_losses, val_losses


def plot_training_loss(train_losses:list, val_losses:list, output_path: Path):
    """
    @brief Plots and saves the training/validation loss curves.

    @param train_losses List of train loss values per epoch.
    @param val_losses List of validation loss values per epoch.
    @param output_path Output image path for the loss plot.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Train loss', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def test(model: nn.Module, test_loader: DataLoader):
    """
    @brief Evaluates the model on the test dataset and prints accuracy.

    @param model The neural network model to evaluate.
    @param test_loader The data loader containing the test data.
    """
    print('Evaluating model performance on the test dataset...')
    
    model.eval()
    
    with torch.no_grad():
        acc = 0
        for samples, labels in test_loader:
            # Forward pass
            outputs = model(samples)
            
            # Compute probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get predicted class
            preds = torch.argmax(probs, dim=1)
            
            # Accumulate correct predictions
            acc += (preds == labels).sum()

    print(f"Test Set Accuracy: {(acc / len(test_loader.dataset))*100.0:.3f}%")


def main(args):
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    images_dir = project_root / 'images'

    # Create images directory to store training curves
    images_dir.mkdir(parents=True, exist_ok=True)

    # Augmentation for training and validation
    train_transform = transforms.Compose([transforms.Grayscale(),
                                          transforms.RandomRotation(30),
                                          transforms.RandomAffine(degrees=0, 
                                                                  translate=(0.25, 0.25), 
                                                                  scale=(0.7, 1.3), shear=10),
                                          transforms.RandomPerspective(distortion_scale=0.2, 
                                                                       p=0.5),
                                          transforms.ColorJitter(brightness=0.5, contrast=0.5),
                                          transforms.RandomInvert(p=0.5),
                                          transforms.GaussianBlur(kernel_size=3),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))])

    # No augmentation for testing
    test_transform = transforms.Compose([transforms.Grayscale(),
                                         transforms.RandomInvert(p=0.5),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))])

    # Download and load the MNIST training dataset
    mnist = datasets.MNIST(root=args.dataset_path, train=True,
                           download=True, transform=train_transform)
    
    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(mnist, [round(len(mnist) * args.train_split), 
                                                      round(len(mnist) * (1 - args.train_split))])

    # Download and load the MNIST test dataset
    test_dataset = datasets.MNIST(root=args.dataset_path, train=False, 
                                  download=True, transform=test_transform)

    # Initialize the Convolutional Neural Network
    model = ConvNet(h=28, w=28, inputs=1, outputs=10)

    # Set up the optimizer (Adam) and loss function (CrossEntropy)
    optimizer = Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    # Create DataLoaders for batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              num_workers=args.num_workers, shuffle=True)
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            num_workers=args.num_workers, shuffle=True)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                             num_workers=args.num_workers, shuffle=False)
    
    # Execute the training loop
    train_losses, val_losses = train(model, train_loader, val_loader, optimizer,
                                     loss_fn, args.num_epochs,
                                     images_dir / 'training.png')

    # Save training history figure in images/training.png
    plot_training_loss(train_losses, val_losses, images_dir / 'training.png')

    # Evaluate the model on the test set
    test(model, test_loader)

    os.makedirs(args.model_path, exist_ok=True)
    
    # Save the trained model, optimizer state, and epoch count
    torch.save({'state_dict': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': args.num_epochs},
                os.path.join(args.model_path, 'model.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train a Convolutional Neural Network (CNN) on the MNIST dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Total number of training epochs to run.')
    
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Mini-batch size for training, validation, and test loaders.')
    
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Fraction of MNIST train split used for training (rest for validation).')
    
    parser.add_argument('--dataset_path', type=Path, default='./data',
                        help='Directory where MNIST is downloaded/read.')
    
    parser.add_argument('--model_path', type=Path, default='./models',
                        help='Directory where the output model checkpoint is saved.')
    
    parser.add_argument('--num_workers', type=int, default=(multiprocessing.cpu_count() - 1),
                        help='Number of DataLoader worker subprocesses.')
    
    args = parser.parse_args()

    main(args)
