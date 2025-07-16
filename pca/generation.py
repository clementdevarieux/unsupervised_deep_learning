import numpy as np
import matplotlib.pyplot as plt

def denormalize_images(images, original_mean, original_std):
    return images * original_std + original_mean

def generate_new_data(latent_space_samples, eigenvectors):
    return np.dot(latent_space_samples, eigenvectors.T)

def sample_latent_space(num_samples, n_components, X_transformed=None):

    if X_transformed is not None:
        mean = np.mean(X_transformed, axis=0)
        std = np.std(X_transformed, axis=0)
        samples = np.random.normal(mean, std, size=(num_samples, n_components))
    else:
        samples = np.random.randn(num_samples, n_components)
    
    return samples

def visualize_generated_images(images, num_images=10):
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()