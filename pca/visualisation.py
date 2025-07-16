import matplotlib.pyplot as plt
import numpy as np

def visualize_latent_space(X_transformed, labels):
    plt.figure(figsize=(10, 8))
    
    # Get unique classes and create a colormap with better contrast
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    
    # Use a colormap with high contrast for better class separation
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    # Plot each class separately for better control
    for i, class_label in enumerate(unique_classes):
        mask = labels == class_label
        plt.scatter(X_transformed[mask, 0], X_transformed[mask, 1], 
                   c=[colors[i]], label=f'Class {class_label}', s=5, alpha=0.7)
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Latent Space Visualization')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_reconstructed_images(original_images, reconstructed_images, n=10, image_shape=(28, 28)):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Original images
        plt.subplot(2, n, i + 1)
        plt.imshow(original_images[i].reshape(image_shape), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # Reconstructed images
        plt.subplot(2, n, i + n + 1)
        plt.imshow(reconstructed_images[i].reshape(image_shape), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
    plt.suptitle('Original vs Reconstructed Images', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_images_grid(images, labels, title, n=8, image_shape=(28, 28)):
    """Display a grid of images with a title"""
    # Convert to numpy if it's a PyTorch tensor
    if hasattr(images, 'numpy'):
        images = images.numpy()
    if hasattr(labels, 'numpy'):
        labels = labels.numpy()
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        if i < len(images) and i < n:
            ax.imshow(images[i].reshape(image_shape), cmap='gray')
            ax.set_title(f'Label: {labels[i]}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()