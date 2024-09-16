import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


from autoencoders.mlp import MLP_AUTOENC
from utils.common import log_to_csv

batch_size = 256
learning_rate = 1e-4
num_epochs = 100
validation_split = 0.2
encode_layers = [128, 64]
decode_layers = [64, 128]
latent_dim = 64
use_spectral_norm = False
use_bn = True
use_layer_norm = False
randomize_input = True
mean = 0.0
std = 0.1
log_file = 'results/autoencoder_mnist.csv'

def main():
    # Load the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    train_size = int((1 - validation_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = MLP_AUTOENC(learning_rate, 784, latent_dim, 
                        encode_layers, 
                        decode_layers, 
                        use_spectral_norm=use_spectral_norm,
                        use_bn=use_bn,
                        use_layer_norm=use_layer_norm,
                        randomize_input=randomize_input,
                        mean=mean,
                        std=std
                        )
    

    # Training loop
    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0
        for data in train_loader:
            img, _ = data  # we only need the images, not the labels
            img = img.view(img.size(0), -1)  # flatten the images
            
            # Forward pass
            output, _ = model(img)
            loss = model.loss(output, img)  # reconstruction loss
            
            # Backward pass and optimization
            model.optim.zero_grad()
            loss.backward()
            model.optim.step()

            train_loss += loss.item()
        
        avg_loss = train_loss / len(train_loader)

        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for data in val_loader:
                img, _ = data  # we only need the images
                img = img.view(img.size(0), -1)  # flatten the images
                
                output, _ = model(img)
                loss = model.loss(output, img)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)


        print(f'Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}, Vald Loss: {avg_val_loss:.4f}')
        
        # Log the epoch and average loss to CSV
        log_to_csv(epoch, avg_loss, avg_val_loss, log_file)

    # Save the trained model
    torch.save(model.state_dict(), 'results/autoencoder_mnist.pth')


if __name__ == '__main__':
    main()