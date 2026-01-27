import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from model import ConvNet

def create_digit_image(digit, font, height=256, width=256):
    """
    Creates a 256x256 blank image and writes the specified number on it using the specified font.
    The image is stored in memory (numpy array), no disk write required.
    """
    # Create a black blank image (256x256)
    image = np.zeros((height, width), dtype=np.uint8)
    
    text = str(digit)
    
    font_scale = 8.0
    thickness = 15
    color = 255  # White color for grayscale
    
    # Calculate text size to center it
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate coordinates to center the text
    x = (width - text_width) // 2 # Center horizontally
    y = (height + text_height) // 2 # Center vertically
    
    # Write the number on the image
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness)
    
    return image

def main():
    parser = argparse.ArgumentParser(description="Generate a random digit image and evaluate using the trained model.")
    parser.add_argument('--model_path', type=Path, default='./models/model.pt', 
                        help='Path to the trained model checkpoint.')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model architecture
    model = ConvNet(h=28, w=28, inputs=1, outputs=10).to(device)
    
    # Load model weights
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Model loaded from {args.model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    model.eval()

    correct = 0
    total = 0
    
    # Define fonts to iterate over
    fonts = [cv2.FONT_HERSHEY_SIMPLEX,
             cv2.FONT_HERSHEY_PLAIN,
             cv2.FONT_HERSHEY_DUPLEX,
             cv2.FONT_HERSHEY_COMPLEX,
             cv2.FONT_HERSHEY_TRIPLEX,
             cv2.FONT_HERSHEY_COMPLEX_SMALL,
             cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
             cv2.FONT_HERSHEY_SCRIPT_COMPLEX]

    # 2. Loop over all fonts and digits
    for font_idx, font in enumerate(fonts):
        for digit in range(10):
            image = create_digit_image(digit, font)
            print(f"\nFont Index: {font_idx} - Digit: {digit}")

            # 3. Preprocess the image for the model
            # Resize to 28x28 (model input size)
            resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
            
            # cv2.imshow(f"Font {font_idx} - Digit {digit}", resized_image)
            # cv2.waitKey(0)
            
            # Apply transformations: ToTensor (scales to 0-1) and Normalize (scales to -1 to 1)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            
            # Add batch dimension: (1, 1, 28, 28)
            input_tensor = transform(resized_image).unsqueeze(0).to(device)

            # 4. Run inference
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            print(f"Predicted Digit: {predicted_class} (Actual: {digit})")
            print(f"Confidence: {confidence:.4f}")
            
            if predicted_class == digit:
                correct += 1
            total += 1

    print(f"\nAccuracy over {total} trials: {100 * correct / total:.2f}%")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()