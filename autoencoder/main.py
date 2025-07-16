from torch import nn
import torch.optim as optim
from autoencoder import Autoencoder
from autoencoder_trainer import train_autoencoder
import data_normalizer
from sklearn.datasets import fetch_openml

if __name__ == '__main__':
    mnist = fetch_openml('mnist_784', version=1)

    activation_function = "relu"
    batch_size = 64
    epochs = 20

    normalizer = data_normalizer.DataNormalizer(method=activation_function)
    normalized_data = normalizer.fit_transform(mnist.data)

    model = Autoencoder(input_shape=784, activation_function=activation_function)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_autoencoder(model, train_data=normalized_data, optimizer=optimizer, criterion=criterion, epochs=epochs, batch_size=batch_size, visualize_every=1)
