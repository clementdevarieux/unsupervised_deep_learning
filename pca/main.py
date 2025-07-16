from algorithm import pca
from compression import decompress_data
from generation import generate_new_data
from utils import load_mnist
from visualisation import visualize_latent_space, plot_reconstructed_images, plot_images_grid
import numpy as np
import matplotlib.pyplot as plt

def main(n_components):
    # Load the MNIST dataset
    train_loader, test_loader = load_mnist(50000)
    
    # Get a batch of images and labels
    images, labels = next(iter(train_loader))
    # print(len(images), "images shape:", images.shape)
    # exit(0)
    
    # Flatten the images to 2D array (batch_size, num_features)
    X = images.view(images.size(0), -1).numpy()
    
    # Standardize the dataset (mean=0, std=1)
    X = (X - np.mean(X)) / np.std(X)
    
    # Apply PCA first to get all components
    X_transformed, eigenvalues, eigenvectors = pca(X, n_components)
    
    # # Compress the data using the compress_data function
    # compressed_data = compress_data(X, n_components)
    
    # Decompress the data
    reconstructed_data = decompress_data(X_transformed, eigenvectors[:, :n_components])
    
    # Visualize the latent space
    visualize_latent_space(X_transformed, labels.numpy())
    
    # Generate new data from the latent space
    from generation import sample_latent_space
    latent_samples = sample_latent_space(num_samples=10, n_components=n_components)
    new_images = generate_new_data(latent_samples, eigenvectors[:, :n_components])
    
    # Display original images
    plot_images_grid(images[:8], labels[:8], "Original MNIST Images")
    
    # Display original vs reconstructed images
    original_flat = images[:8].view(8, -1).numpy()
    plot_reconstructed_images(original_flat, reconstructed_data[:8], n=8)
    
    # Display generated images
    # Create dummy labels for generated images
    dummy_labels = np.arange(len(new_images))
    plot_images_grid(new_images[:8], dummy_labels[:8], "Generated Images from Latent Space")

if __name__ == "__main__":
    main(n_components=200)  # You can change the number of components as needed