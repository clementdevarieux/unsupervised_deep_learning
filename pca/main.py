from algorithm import pca
from compression import decompress_data
from generation import generate_new_data
from utils import load_mnist, print_mnist_images
from visualisation import visualize_latent_space
import numpy as np
import matplotlib.pyplot as plt
import torch

def main(n_components):
    # Load the MNIST dataset
    train_loader, test_loader = load_mnist(100)
    
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
    
    # Print some images
    print_mnist_images(images[:8], labels[:8])

    reconstructed_tensor = torch.from_numpy(reconstructed_data[:8]).float().reshape(-1, 1, 28, 28)
    print_mnist_images(reconstructed_tensor, labels[:8])
    
    # Convert new_images back to PyTorch tensor for visualization
    new_images_tensor = torch.from_numpy(new_images).float().reshape(-1, 1, 28, 28)
    print_mnist_images(new_images_tensor, labels[:10])
    
if __name__ == "__main__":
    main(n_components=2)  # You can change the number of components as needed