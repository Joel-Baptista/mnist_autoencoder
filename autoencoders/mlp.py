import torch 
from torch import nn


class MLP_AUTOENC(nn.Module):
    def __init__(self, 
                 lr, 
                 input_dim, 
                 latent_dim, 
                 encoder_layers, 
                 decoder_layers, 
                 use_spectral_norm = False, 
                 use_bn = False, 
                 use_layer_norm = False,
                 randomize_input = False,
                 mean = 0.0,
                 std = 0.1,
                 sparse_lambda = 0.0
                 ) -> None:
        super().__init__()

        self.mean = mean
        self.std = std
        self.randomize_input = randomize_input
        self.sparse_lambda = sparse_lambda

        self.encoder = MPL(input_dim, latent_dim, encoder_layers, use_spectral_norm, use_bn, use_layer_norm)
        self.decoder = MPL(latent_dim, input_dim, decoder_layers, use_spectral_norm, use_bn, use_layer_norm)

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):

        if self.randomize_input:
            x = self.addGaussianNoise(x)
        
        red_vec = self.encoder(x)
        
        output = self.decoder(red_vec)

        return output, red_vec
    
    def loss(self, x, x_hat, red_vec = None):
        loss_function = nn.MSELoss()

        loss = loss_function(x, x_hat)

        if self.sparse_lambda > 0 and red_vec is not None:
            loss += self.sparse_lambda * torch.sum(torch.abs(red_vec))

        return loss
    
    def addGaussianNoise(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        noisy_tensor = torch.clamp(noisy_tensor, 0., 1.)

        return noisy_tensor

class MPL(nn.Module):
    def __init__(self, input, output, hiddens, use_spectral_norm = False, use_bn = False, use_layer_norm = False) -> None:
        super().__init__()
        
        layers = []

        current_dim = input

        for idx, hidden_dim in enumerate(hiddens):
            layers.append(nn.Linear(current_dim,
                                    hidden_dim))
            if use_spectral_norm:
                layers[-1] = nn.utils.spectral_norm(layers[-1])
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim, affine=True))
            elif use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim,
                                           elementwise_affine=True))
            
            layers.append(nn.ReLU())

            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            
        return x 


