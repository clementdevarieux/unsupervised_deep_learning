import matplotlib.pyplot as plt

def visualize_latent_space(X_transformed, labels):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Classes')
    plt.title('PCA Latent Space Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

def plot_reconstructed_images(original_images, reconstructed_images, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Original images
        plt.subplot(2, n, i + 1)
        plt.imshow(original_images[i].reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # Reconstructed images
        plt.subplot(2, n, i + n + 1)
        plt.imshow(reconstructed_images[i].reshape(28, 28), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
    plt.show()