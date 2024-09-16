import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

from autoencoders.mlp import MLP_AUTOENC
from utils.common import DotDict

path = 'results/models/baseline3'

# Set seaborn theme for prettier plots
sns.set_theme(style="whitegrid")

# Function to load the trained autoencoder model
def load_model(model_path, cfg):
    model = MLP_AUTOENC(**cfg)
    model.load_state_dict(torch.load(model_path + "/autoencoder_mnist.pth"))
    model.eval()  # Set model to evaluation mode
    return model

# Unnormalize images for display
def unnormalize(tensor):
    tensor = tensor * 0.5 + 0.5  # reverse the normalization: from (-1,1) to (0,1)
    return tensor

# Function to test and visualize predictions
def test_and_visualize(model, test_loader, num_samples=10):
    # Get a batch of test samples
    data_iter = iter(test_loader)
    images, _ = next(data_iter)

    # Flatten the images for the model
    images_flat = images.view(images.size(0), -1)

    # Generate reconstructions
    with torch.no_grad():
        reconstructions, _ = model(images_flat)

    # Reshape the reconstructions back to image format
    reconstructions = reconstructions.view(-1, 1, 28, 28)

    # Convert tensors to numpy arrays
    images = unnormalize(images).numpy()
    reconstructions = unnormalize(reconstructions).numpy()

    # Plot the original images and their reconstructions
    fig, axs = plt.subplots(num_samples, 2, figsize=(6, num_samples * 2))

    for i in range(num_samples):
        # Original image
        axs[i, 0].imshow(np.transpose(images[i], (1, 2, 0)), cmap='gray')
        axs[i, 0].axis('off')
        axs[i, 0].set_title('Original', fontsize=12)

        # Reconstructed image
        axs[i, 1].imshow(np.transpose(reconstructions[i], (1, 2, 0)), cmap='gray')
        axs[i, 1].axis('off')
        axs[i, 1].set_title('Reconstruction', fontsize=12)

    plt.tight_layout()
    plt.show()


def main():

    with open(path + "/config.yaml") as f:
        cfg = DotDict(yaml.load(f, Loader=yaml.loader.SafeLoader))
    # Load the trained model
    model = load_model(path, cfg)

    # Define the transformation (note: no noise is added for testing)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the test dataset
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False) 

    test_and_visualize(model, test_loader, num_samples=5)

if __name__ == '__main__':
    main()