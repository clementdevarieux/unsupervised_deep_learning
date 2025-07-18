from torch import nn
import torch.optim as optim
#from autoencoder import Autoencoder
#from autoencoder_trainer import train_autoencoder
import data_normalizer
from sklearn.datasets import fetch_openml
#from conv_autoencoder import ConvAutoencoder
import torch
from variational_latent_analysis import encode_dataset, generate_samples
#from visualisation import visualize_latent_space
from vae_visualization import visualize_latent_space
from vae import VariationalAutoencoder
from train_vae import train_variational_autoencoder

if __name__ == '__main__':
    mnist = fetch_openml('mnist_784', version=1)

    activation_function = "relu"
    batch_size = 64
    epochs = 50
    lr = 1e-4
    criterion = nn.MSELoss()

    normalizer = data_normalizer.DataNormalizer(method=activation_function)
    normalized_data = normalizer.fit_transform(mnist.data)

    #model = Autoencoder(input_shape=784, activation_function=activation_function)
    #model = ConvAutoencoder(input_shape=784, activation_function=activation_function)
    model = VariationalAutoencoder(input_shape=784, activation_function=activation_function)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    sample = torch.FloatTensor(normalized_data[:1])
    output, mu, logvar = model(sample)  # Unpack the tuple
    print(f"Before training - Output range: {output.min():.3f} to {output.max():.3f}")

    train_variational_autoencoder(
        model,
        train_data=normalized_data,
        optimizer=optimizer,
        criterion=criterion,
        epochs=epochs,
        batch_size=batch_size,
        visualize_every=1,
        coef_KL=.01
    )
    latents = encode_dataset(model, normalized_data)
    new_samples = generate_samples(model, latents, n_samples=100)
    visualize_latent_space(model, latents)