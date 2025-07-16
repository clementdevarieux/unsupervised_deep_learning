from torch import nn

class Autoencoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        input_shape = kwargs["input_shape"]

        activations = {
            None: nn.ReLU(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid()
        }

        self.activation_function = activations[kwargs.get("activation_function")]

        if kwargs.get("activation_function") == "tanh":
            final_activation = nn.Tanh()
        else:
            final_activation = nn.Sigmoid()


        self.encoder = nn.Sequential(
            nn.Linear(input_shape, 512),
            self.activation_function,
            nn.Linear(512, 256),
            self.activation_function,
            nn.Linear(256, 64),
            self.activation_function,
            nn.Linear(64, 16),
            self.activation_function,
            nn.Linear(16, 4),
            self.activation_function,
            nn.Linear(4, 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            self.activation_function,
            nn.Linear(4, 16),
            self.activation_function,
            nn.Linear(16, 64),
            self.activation_function,
            nn.Linear(64, 256),
            self.activation_function,
            nn.Linear(256, 512),
            self.activation_function,
            nn.Linear(512, input_shape),
            final_activation
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded