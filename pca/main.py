from algorithm import pca
from compression import decompress_data
from generation import generate_new_data, denormalize_images, sample_latent_space
from utils import load_mnist
from visualisation import visualize_latent_space, plot_reconstructed_images, plot_images_grid
import numpy as np

def main(n_components):
    train_loader, test_loader = load_mnist(50000)
    
    images, labels = next(iter(train_loader))
    X = images.view(images.size(0), -1).numpy()
    
    original_mean = np.mean(X)
    original_std = np.std(X)

    X = (X - original_mean) / original_std

    # ESPACE LATENT
    X_transformed, eigenvalues, eigenvectors = pca(X, n_components)
    visualize_latent_space(X_transformed, labels.numpy())

    # RECONSTRUCTION
    reconstructed_data = decompress_data(X_transformed, eigenvectors[:, :n_components])
    reconstructed_data_denorm = denormalize_images(reconstructed_data, original_mean, original_std)
    original_flat = images[:10].view(10, -1).numpy()
    plot_reconstructed_images(original_flat, reconstructed_data_denorm[:10], n=10)

    # CREATION NEW DATA
    latent_samples = sample_latent_space(num_samples=10, n_components=n_components, X_transformed=X_transformed)
    new_images = generate_new_data(latent_samples, eigenvectors[:, :n_components])
    new_images_denorm = denormalize_images(new_images, original_mean, original_std)
    dummy_labels = np.arange(len(new_images_denorm))
    plot_images_grid(new_images_denorm[:10], dummy_labels[:10], "Generated Images from Latent Space")

if __name__ == "__main__":
    main(n_components=500)  