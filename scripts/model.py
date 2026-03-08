import torch
import torch.nn as nn


class ConvNet(nn.Module):

    def __init__(self, h=28, w=28, inputs=1, outputs=10):
        """
        @brief Initialize the convolutional network architecture.

        @param h Input image height.
        @param w Input image width.
        @param inputs Number of input channels.
        @param outputs Number of output classes.
        @return None.
        """
        super(ConvNet, self).__init__()

        self.convolutional_layers = nn.Sequential(
            # First layer
            nn.Conv2d(inputs, 6, kernel_size=5, stride=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second layer
            nn.Conv2d(6, 16, kernel_size=5, stride=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Get the number of features produced by the convolutional block
        conv_output_size = self._get_conv_output_size(h, w, inputs)

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=conv_output_size, out_features=120, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=outputs, bias=False),
        )
    
    def _get_conv_output_size(self, h, w, inputs):
        """
        @brief Computes the number of output features produced by the convolutional layers.

        This method generates a dummy tensor with the same shape as the actual
        input images and feeds it through the convolutional base. By examining
        the resulting tensor size, it determines the exact number of features
        required by the first fully connected layer.

        @param h The input image height.
        @param w The input image width.
        @param inputs The number of input channels.
        @return The flattened feature size after all convolutional layers.
        """
        # Create a dummy tensor
        x = torch.zeros(1, inputs, h, w)
        
        # Pass dummy input through the convolutional layers
        x = self.convolutional_layers(x)
        
        # Get the total number of elements in x
        return x.numel()

    def forward(self, x):
        """
        @brief Run a forward pass through the network.

        @param x Input tensor with shape (N, C, H, W).
        @return Output logits tensor with shape (N, outputs).
        """
        x = self.convolutional_layers(x)
        x = x.view(x.size(0), -1)
        return self.linear_layers(x)
