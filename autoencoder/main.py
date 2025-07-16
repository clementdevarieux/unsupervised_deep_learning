from torch import nn
import torch.optim as optim
from autoencoder import Autoencoder
from autoencoder_trainer import train_autoencoder
import data_normalizer
from sklearn.datasets import fetch_openml
from conv_autoencoder import ConvAutoencoder  # Your new file
import torch

if __name__ == '__main__':
    mnist = fetch_openml('mnist_784', version=1)

    activation_function = "leaky_relu"
    batch_size = 64
    epochs = 20
    lr = 1e-4

    normalizer = data_normalizer.DataNormalizer(method=activation_function)
    normalized_data = normalizer.fit_transform(mnist.data)

    #model = Autoencoder(input_shape=784, activation_function=activation_function)
    model = ConvAutoencoder(input_shape=784, activation_function=activation_function)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    sample = torch.FloatTensor(normalized_data[:1])
    output = model(sample)
    print(f"Before training - Output range: {output.min():.3f} to {output.max():.3f}")

    train_autoencoder(model, train_data=normalized_data, optimizer=optimizer, criterion=criterion, epochs=epochs, batch_size=batch_size, visualize_every=1)
