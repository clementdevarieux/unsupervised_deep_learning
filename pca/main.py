from algorithm import pca
from compression import decompress_data
from generation import generate_new_data, denormalize_images, sample_latent_space
from utils import load_mnist
from visualisation import visualize_latent_space, plot_reconstructed_images, plot_images_grid
import numpy as np
from data_loader import load_images_from_folder, load_and_normalize_images

def main(n_components):
    # MNIST DIGIT
    # train_loader, test_loader = load_mnist(5000)
    
    # images, labels = next(iter(train_loader))

    # X = images.view(images.size(0), -1).numpy()
    
    # FRUITS DATASET
    input_dir = 'data/fruits_dataset'

    images, labels = load_images_from_folder(input_dir, max_images_per_class=800, target_size=(32, 32))
    images = np.array(images)
    labels = np.array(labels)
    X = np.array(images)
    X = X.reshape(X.shape[0], -1) 

    # BOTH
    original_mean = np.mean(X)
    original_std = np.std(X)

    X = (X - original_mean) / original_std

    # ESPACE LATENT
    X_transformed, eigenvalues, eigenvectors = pca(X, n_components)
    # visualize_latent_space(X_transformed, labels.numpy()) # MNIST
    visualize_latent_space(X_transformed, labels) # FRUITS

    # RECONSTRUCTION
    reconstructed_data = decompress_data(X_transformed, eigenvectors[:, :n_components])
    reconstructed_data_denorm = denormalize_images(reconstructed_data, original_mean, original_std)

    num_images_to_show = 10
    random_indices = np.random.choice(len(images), size=num_images_to_show, replace=False)
    # original_flat = images[:10].view(10, -1).numpy() # MNIST
    original_flat = images[random_indices].reshape(num_images_to_show, -1) # FRUITS

    plot_reconstructed_images(original_flat, reconstructed_data_denorm[random_indices], n=10)

    # CREATION NEW DATA
    latent_samples = sample_latent_space(num_samples=10, n_components=n_components, X_transformed=X_transformed)
    new_images = generate_new_data(latent_samples, eigenvectors[:, :n_components])
    new_images_denorm = denormalize_images(new_images, original_mean, original_std)
    dummy_labels = np.arange(len(new_images_denorm))
    # plot_images_grid(new_images_denorm[:10], 
    #                  dummy_labels[:10], 
    #                  title="Generated Images from Latent Space") # MNIST
    plot_images_grid(new_images_denorm[:10], 
                     dummy_labels[:10], 
                     title="Generated Images from Latent Space",
                     n=10,
                     image_shape=(32, 32, 3),
                     dataset_type='fruits') # FRUITS

if __name__ == "__main__":
    main(n_components=200)  