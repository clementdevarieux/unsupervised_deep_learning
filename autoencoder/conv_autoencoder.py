from torch import nn

class ConvAutoencoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        input_shape = kwargs["input_shape"]

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

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            self.activation_function,

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            self.activation_function,

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            self.activation_function,

            nn.Conv2d(128, 1, kernel_size=3, stride=2, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            self.activation_function,

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            self.activation_function,
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            self.activation_function,

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1, output_padding=0),
            final_activation
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 28, 28)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(batch_size, 784)
        return decoded