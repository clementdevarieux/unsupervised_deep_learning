import numpy as np
from torch import nn
import torch


class VariationalAutoencoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        input_shape = kwargs["input_shape"]  # Now expects 3072 (32*32*3)

        activations = {
            None: nn.ReLU(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(0.2)
        }

        self.activation_function = activations[kwargs.get("activation_function", "relu")]

        if kwargs.get("activation_function") == "tanh":
            final_activation = nn.Tanh()
        else:
            final_activation = nn.Sigmoid()

        # Encoder: 32x32x3 -> 2x2x128 -> latent space
        self.encoder = nn.Sequential(
            # Input: 3x32x32
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # -> 32x16x16
            nn.BatchNorm2d(32),
            self.activation_function,

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> 64x8x8
            nn.BatchNorm2d(64),
            self.activation_function,

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> 128x4x4
            nn.BatchNorm2d(128),
            self.activation_function,

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # -> 128x2x2
            nn.BatchNorm2d(128),
            self.activation_function,

            nn.Flatten(),  # -> 512
            nn.Linear(512, 256),
            self.activation_function,
            nn.Linear(256, 4)  # mu and logvar (2 each)
        )

        # Decoder: latent space -> 2x2x128 -> 32x32x3
        self.decoder = nn.Sequential(
            nn.Linear(2, 256),
            self.activation_function,
            nn.Linear(256, 512),
            self.activation_function,

            nn.Unflatten(1, (128, 2, 2)),  # -> 128x2x2

            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> 128x4x4
            nn.BatchNorm2d(128),
            self.activation_function,

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> 64x8x8
            nn.BatchNorm2d(64),
            self.activation_function,

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0),  # -> 32x16x16
            nn.BatchNorm2d(32),
            self.activation_function,

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, output_padding=0),  # -> 3x32x32
            final_activation
        )

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu)
        std = torch.exp(0.5 * logvar)
        return mu + std * eps

    def encode_to_params(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 3, 32, 32)  # Reshape to 3x32x32
        encoded = self.encoder(x)
        mu = encoded[:, :2]
        logvar = encoded[:, 2:]
        return mu, logvar

    def encode(self, x):
        mu, logvar = self.encode_to_params(x)
        return self.reparameterize(mu, logvar)

    def decode(self, z):
        decoded = self.decoder(z)
        batch_size = decoded.size(0)
        return decoded.view(batch_size, 3072)  # Flatten to 3072 (32*32*3)

    def forward(self, x):
        mu, logvar = self.encode_to_params(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        return decoded.view(decoded.size(0), 3072), mu, logvar  # Return flattened output

    def reconstruct(self, x):
        output, _, _ = self.forward(x)
        return output