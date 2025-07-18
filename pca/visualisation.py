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


# def plot_reconstructed_images(original_images, reconstructed_images, n=10, image_shape=(28, 28)):
#     plt.figure(figsize=(20, 4))
#     for i in range(n):
#         # Original images
#         plt.subplot(2, n, i + 1)
#         plt.imshow(original_images[i].reshape(image_shape), cmap='gray')
#         plt.title("Original")
#         plt.axis('off')

#         # Reconstructed images
#         plt.subplot(2, n, i + n + 1)
#         plt.imshow(reconstructed_images[i].reshape(image_shape), cmap='gray')
#         plt.title("Reconstructed")
#         plt.axis('off')
#     plt.suptitle('Original vs Reconstructed Images', fontsize=16, fontweight='bold')
#     plt.tight_layout()
#     plt.show()

def plot_reconstructed_images(original_images, reconstructed_images, n=10):
    fig, axes = plt.subplots(2, n, figsize=(15, 4))
    
    for i in range(n):
        if original_images[i].size == 784:  # MNIST: 28*28
            image_shape = (28, 28)
            cmap = 'gray'
            channels = 1
        elif original_images[i].size == 3072:  # RGB 32x32: 32*32*3
            image_shape = (32, 32, 3)
            cmap = None
            channels = 3
        else:
            raise ValueError(f"Taille d'image non supportée: {original_images[i].size}")
        
        # Image originale
        if channels == 1:
            axes[0, i].imshow(original_images[i].reshape(image_shape), cmap=cmap)
        else:
            # Pour RGB, s'assurer que les valeurs sont dans [0, 1] ou [0, 255]
            img = original_images[i].reshape(image_shape)
            if img.max() <= 1.0:
                axes[0, i].imshow(img)
            else:
                axes[0, i].imshow(img.astype('uint8'))
        
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Image reconstruite
        if channels == 1:
            axes[1, i].imshow(reconstructed_images[i].reshape(image_shape), cmap=cmap)
        else:
            img_recon = reconstructed_images[i].reshape(image_shape)
            # Clip les valeurs pour éviter les problèmes d'affichage
            img_recon = np.clip(img_recon, 0, 255 if img_recon.max() > 1 else 1)
            if img_recon.max() <= 1.0:
                axes[1, i].imshow(img_recon)
            else:
                axes[1, i].imshow(img_recon.astype('uint8'))
        
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_images_grid(images, labels, title, n=8, image_shape=(28, 28), dataset_type='mnist'):
    
    # if dataset_type != 'mnist':
    #     images = images.numpy()
    #     labels = labels.numpy()
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        if i < len(images) and i < n:
            # Gérer les différents formats d'image
            if dataset_type == 'mnist':  # Niveaux de gris
                ax.imshow(images[i].reshape(image_shape), cmap='gray')
            else:  # RGB
                img = images[i].reshape(image_shape)
                # Normaliser les valeurs dans la plage [0, 1] pour l'affichage
                img_normalized = (img - img.min()) / (img.max() - img.min()) if img.max() > img.min() else img
                ax.imshow(np.clip(img_normalized, 0, 1))
            ax.set_title(f'Label: {labels[i]}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()