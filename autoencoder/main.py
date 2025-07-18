from torch import nn
import torch.optim as optim
import data_normalizer
from data_loader import load_images_from_folder, load_and_normalize_images
import torch
from variational_latent_analysis import encode_dataset, generate_samples
from vae_visualization import visualize_latent_space
from vae import VariationalAutoencoder
from train_vae import train_variational_autoencoder
import numpy as np

if __name__ == '__main__':
    # Load fruit dataset
    # UPDATE THIS PATH to point to your fruit dataset folder
    dataset_path = "./fruit_dataset"  # Change this to your actual dataset path

    print("Loading fruit dataset...")
    images, labels = load_images_from_folder(
        dataset_path,
        target_size=(32, 32),
        max_images_per_class=200,
        n_classes=5
    )

    # Convert to flattened normalized data
    print("Normalizing images...")
    normalized_data = load_and_normalize_images(images, flatten=True)
    normalized_data = np.array(normalized_data)

    print(f"Dataset shape: {normalized_data.shape}")
    print(f"Number of classes: {len(set(labels))}")
    print(f"Classes: {set(labels)}")

    activation_function = "tanh"
    batch_size = 64
    epochs = 50
    lr = 1e-4
    criterion = nn.MSELoss()
    coef_KL = 0.1

    # Use data normalizer if needed (though images are already 0-1 normalized)
    normalizer = data_normalizer.DataNormalizer(method=activation_function)
    normalized_data = normalizer.fit_transform(normalized_data)

    # Create VAE model for RGB 32x32 images (3072 input dimensions)
    model = VariationalAutoencoder(input_shape=3072, activation_function=activation_function)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Test the model with a sample
    sample = torch.FloatTensor(normalized_data[:1])
    output, mu, logvar = model(sample)
    print(f"Before training - Output range: {output.min():.3f} to {output.max():.3f}")
    print(f"Sample input shape: {sample.shape}")
    print(f"Sample output shape: {output.shape}")

    # Train the VAE
    train_variational_autoencoder(
        model,
        train_data=normalized_data,
        optimizer=optimizer,
        criterion=criterion,
        epochs=epochs,
        batch_size=batch_size,
        visualize_every=1,
        coef_KL=coef_KL
    )

    # Generate latent representations and samples
    latents = encode_dataset(model, normalized_data)
    new_samples = generate_samples(model, latents, n_samples=100)
    visualize_latent_space(model, latents)