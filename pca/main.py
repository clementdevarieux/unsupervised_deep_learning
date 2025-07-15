from algorithm import pca
from utils import load_mnist, print_mnist_images
import numpy as np
import matplotlib.pyplot as plt


def main(n_components):
    train_loader, test_loader = load_mnist()
    print("train_loader shape:", len(train_loader.dataset))
    print("test_loader shape:", len(test_loader.dataset))

    # Get a batch of images and labels
    images, labels = next(iter(train_loader))
    print("images shape:", images.shape)
    print("labels shape:", labels.shape)
    # Print some images
    print_mnist_images(images[:8], labels[:8])
    
    # Flatten the images to 2D array (batch_size, num_features)
    X = images.view(images.size(0), -1).numpy()
    print("Flattened images shape:", X.shape)
    # Standardize the dataset (mean=0, std=1)
    X = (X - np.mean(X)) / np.std(X)
    print("Standardized images shape:", X.shape)
    # Print the first few rows of the standardized dataset
    print("First few rows of standardized dataset:\n", X[:5])

    # Apply PCA
    X_transformed, eigenvalues, eigenvectors = pca(X, n_components)

    # Plot the transformed data
    plt.figure(figsize=(8, 6))
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=labels.numpy(), cmap='viridis', alpha=0.5)
    plt.colorbar()  
    plt.title('PCA of MNIST Dataset')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()


if __name__ == "__main__":
    main(n_components=2) 
    # print_mnist_images(torchvision.datasets.MNIST('./data', train=True, download=True).data[:64].unsqueeze(1).float(),
    #                    torchvision.datasets.MNIST('./data', train=True, download=True).targets[:64])