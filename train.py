import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import yaml

from autoencoders.mlp import MLP_AUTOENC
from utils.common import log_to_csv
from utils.common import DotDict

# batch_size = 256
# learning_rate = 1e-4
# num_epochs = 100
# validation_split = 0.2
# encode_layers = [128, 64]
# decode_layers = [64, 128]
# latent_dim = 64
# use_spectral_norm = False
# use_bn = True
# use_layer_norm = False
# randomize_input = True
# mean = 0.0
# std = 0.2

log_file = 'results/autoencoder_mnist.csv'
device_config = "cuda:1"
path = "results/config.yaml"

def main():

    with open(path) as f:
        cfg = DotDict(yaml.load(f, Loader=yaml.loader.SafeLoader))
    # Load the trained model
    print(cfg)
    if torch.cuda.is_available():
        print("Using CUDA")
        device = torch.device(device_config)
    else:
        print("Using CPU")
        device = torch.device("cpu")

    # Load the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    train_size = int((1 - cfg.train.validation_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = MLP_AUTOENC(**cfg.alg).to(device)
    

    # Training loop
    for epoch in range(cfg.train.num_epochs):
        train_loss = 0
        val_loss = 0
        for data in train_loader:
            img, _ = data  # we only need the images, not the labels
            img = img.view(img.size(0), -1)  # flatten the images
            
            img = img.to(device)

            # Forward pass
            output, red_vec = model(img)
            loss = model.loss(output, img, red_vec)  # reconstruction loss
            
            # Backward pass and optimization
            model.optim.zero_grad()
            loss.backward()
            model.optim.step()

            train_loss += loss.cpu().item()
        
        avg_loss = train_loss / len(train_loader)

        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for data in val_loader:
                img, _ = data  # we only need the images
                img = img.view(img.size(0), -1)  # flatten the images
                
                img = img.to(device)

                output, red_vec = model(img)
                loss = model.loss(output, img, red_vec)
                val_loss += loss.cpu().item()
        
        avg_val_loss = val_loss / len(val_loader)


        print(f'Epoch [{epoch}/{cfg.train.num_epochs}], Loss: {avg_loss:.4f}, Vald Loss: {avg_val_loss:.4f}')
        
        # Log the epoch and average loss to CSV
        log_to_csv(epoch, avg_loss, avg_val_loss, log_file)

    # Save the trained model
    torch.save(model.state_dict(), 'results/autoencoder_mnist.pth')


if __name__ == '__main__':
    main()