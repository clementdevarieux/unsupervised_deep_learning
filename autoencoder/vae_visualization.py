import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
import data_normalizer
from vae import VariationalAutoencoder
from variational_latent_analysis import encode_dataset, generate_samples


def visualize_latent_space(latent_representations, labels=None, title="VAE Latent Space"):
    plt.figure(figsize=(10, 8))

    if labels is not None:
        unique_classes = np.unique(labels)
        n_classes = len(unique_classes)

        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

        for i, class_label in enumerate(unique_classes):
            mask = labels == class_label
            plt.scatter(latent_representations[mask, 0], latent_representations[mask, 1],
                        c=[colors[i]], label=f'Class {int(class_label)}', s=5, alpha=0.7)

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(latent_representations[:, 0], latent_representations[:, 1],
                    s=5, alpha=0.7)

    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


import torch
import numpy as np
import matplotlib.pyplot as plt


def show_reconstruction(model, sample, epoch=None, image_shape=(3, 32, 32), n_samples=5):
    """
    Show reconstructions of n_samples in a single figure.
    Creates a 2xn_samples grid: top row = originals, bottom row = reconstructions
    """
    device = next(model.parameters()).device
    model.eval()

    # If we only got one sample, we need to get more samples from the dataset
    # This is a bit tricky since we only have one sample passed in
    # For now, let's just duplicate the sample to show the concept
    # In practice, you might want to pass a batch of samples instead

    fig, axes = plt.subplots(2, n_samples, figsize=(15, 6))

    with torch.no_grad():
        for i in range(n_samples):
            # Use the same sample for demonstration (you could modify this to use different samples)
            sample_tensor = torch.FloatTensor(sample).to(device)
            if len(sample_tensor.shape) == 1:
                sample_tensor = sample_tensor.unsqueeze(0)

            # Get reconstruction
            reconstruction, _, _ = model(sample_tensor)
            reconstruction = reconstruction.cpu().numpy().squeeze()
            original = sample_tensor.cpu().numpy().squeeze()

            # Reshape for display
            if image_shape == (3, 32, 32):
                # RGB image
                original_img = original.reshape(32, 32, 3)
                reconstructed_img = reconstruction.reshape(32, 32, 3)

                # Ensure values are in [0, 1] range for display
                original_img = np.clip(original_img, 0, 1)
                reconstructed_img = np.clip(reconstructed_img, 0, 1)
            else:
                # Grayscale fallback
                original_img = original.reshape(28, 28)
                reconstructed_img = reconstruction.reshape(28, 28)

            # Plot original (top row)
            axes[0, i].imshow(original_img, cmap='gray' if len(original_img.shape) == 2 else None)
            axes[0, i].set_title(f'Original {i + 1}')
            axes[0, i].axis('off')

            # Plot reconstruction (bottom row)
            axes[1, i].imshow(reconstructed_img, cmap='gray' if len(reconstructed_img.shape) == 2 else None)
            axes[1, i].set_title(f'Reconstructed {i + 1}')
            axes[1, i].axis('off')

    title = f'Fruit VAE Reconstruction - Epoch {epoch}' if epoch else 'Fruit VAE Reconstruction'
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

    model.train()


def show_multiple_reconstructions(model, data_samples, epoch=None, image_shape=(3, 32, 32), n_samples=5):
    """
    Show reconstructions of multiple different samples.
    data_samples should be a list/array of samples to reconstruct.
    """
    device = next(model.parameters()).device
    model.eval()

    # Select n_samples random samples from the data
    if len(data_samples) >= n_samples:
        selected_indices = np.random.choice(len(data_samples), n_samples, replace=False)
        selected_samples = [data_samples[i] for i in selected_indices]
    else:
        selected_samples = data_samples
        n_samples = len(selected_samples)

    fig, axes = plt.subplots(2, n_samples, figsize=(3 * n_samples, 6))

    # Handle case where n_samples = 1
    if n_samples == 1:
        axes = axes.reshape(2, 1)

    with torch.no_grad():
        for i, sample in enumerate(selected_samples):
            sample_tensor = torch.FloatTensor(sample).to(device)
            if len(sample_tensor.shape) == 1:
                sample_tensor = sample_tensor.unsqueeze(0)

            # Get reconstruction
            reconstruction, _, _ = model(sample_tensor)
            reconstruction = reconstruction.cpu().numpy().squeeze()
            original = sample_tensor.cpu().numpy().squeeze()

            # Reshape for display
            if image_shape == (3, 32, 32):
                # RGB image
                original_img = original.reshape(32, 32, 3)
                reconstructed_img = reconstruction.reshape(32, 32, 3)

                # Ensure values are in [0, 1] range for display
                original_img = np.clip(original_img, 0, 1)
                reconstructed_img = np.clip(reconstructed_img, 0, 1)
            else:
                # Grayscale fallback
                original_img = original.reshape(28, 28)
                reconstructed_img = reconstruction.reshape(28, 28)

            # Plot original (top row)
            axes[0, i].imshow(original_img, cmap='gray' if len(original_img.shape) == 2 else None)
            axes[0, i].set_title(f'Original {i + 1}')
            axes[0, i].axis('off')

            # Plot reconstruction (bottom row)
            axes[1, i].imshow(reconstructed_img, cmap='gray' if len(reconstructed_img.shape) == 2 else None)
            axes[1, i].set_title(f'Reconstructed {i + 1}')
            axes[1, i].axis('off')

    title = f'Fruit VAE Reconstructions - Epoch {epoch}' if epoch else 'Fruit VAE Reconstructions'
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

    model.train()


def show_sample_images(images, labels, title="Sample Images", n=8):
    if hasattr(images, 'numpy'):
        images = images.numpy()
    if hasattr(labels, 'numpy'):
        labels = labels.numpy()

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        if i < len(images) and i < n:
            ax.imshow(images[i].reshape(28, 28), cmap='gray')
            ax.set_title(f'Class {int(labels[i])}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def show_generated_samples(generated_images, n_samples=8):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle('Generated Images from VAE Latent Sampling', fontsize=16, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        if i < n_samples:
            ax.imshow(generated_images[i].reshape(28, 28), cmap='gray')
            ax.set_title(f'Generated #{i+1}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def interpolate_between_digits(model, latent_representations, labels, digit1=0, digit2=1, n_steps=8):
    device = next(model.parameters()).device
    model.eval()

    digit1_indices = np.where(labels == digit1)[0]
    digit2_indices = np.where(labels == digit2)[0]

    if len(digit1_indices) == 0 or len(digit2_indices) == 0:
        print(f"Could not find examples of digits {digit1} or {digit2}")
        return

    start_latent = latent_representations[digit1_indices[0]]
    end_latent = latent_representations[digit2_indices[0]]

    interpolation_steps = np.linspace(0, 1, n_steps)
    interpolated_latents = []

    for step in interpolation_steps:
        interpolated = start_latent * (1 - step) + end_latent * step
        interpolated_latents.append(interpolated)

    interpolated_latents = np.array(interpolated_latents)

    with torch.no_grad():
        latent_tensor = torch.FloatTensor(interpolated_latents).to(device)
        interpolated_images = model.decode(latent_tensor).cpu().numpy()

    fig, axes = plt.subplots(1, n_steps, figsize=(2 * n_steps, 3))
    fig.suptitle(f'VAE Interpolation from Digit {digit1} to Digit {digit2}', fontsize=16, fontweight='bold')

    for i in range(n_steps):
        axes[i].imshow(interpolated_images[i].reshape(28, 28), cmap='gray')
        axes[i].set_title(f'Step {i + 1}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    model.train()
    return interpolated_images


def explore_latent_space_grid(model, latent_representations, grid_size=5, scale=2.0):
    device = next(model.parameters()).device
    model.eval()

    latent_mean = np.mean(latent_representations, axis=0)
    latent_std = np.std(latent_representations, axis=0)

    x_range = np.linspace(-scale * latent_std[0], scale * latent_std[0], grid_size)
    y_range = np.linspace(-scale * latent_std[1], scale * latent_std[1], grid_size)

    grid_latents = []
    for y in y_range:
        for x in x_range:
            latent_point = latent_mean.copy()
            latent_point[0] = latent_mean[0] + x
            latent_point[1] = latent_mean[1] + y
            grid_latents.append(latent_point)

    grid_latents = np.array(grid_latents)

    with torch.no_grad():
        latent_tensor = torch.FloatTensor(grid_latents).to(device)
        generated_images = model.decode(latent_tensor).cpu().numpy()

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    fig.suptitle('VAE Latent Space Grid Exploration', fontsize=16, fontweight='bold')

    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            axes[i, j].imshow(generated_images[idx].reshape(28, 28), cmap='gray')
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()

    model.train()
    return generated_images


def perturb_existing_samples(model, subset_data, subset_labels, latent_representations, n_samples=8, noise_scale=0.5):
    device = next(model.parameters()).device
    model.eval()

    indices = np.random.choice(len(latent_representations), n_samples, replace=False)

    fig, axes = plt.subplots(2, n_samples, figsize=(2 * n_samples, 6))
    fig.suptitle('Original vs Perturbed VAE Latent Representations', fontsize=16, fontweight='bold')

    for i, idx in enumerate(indices):
        original_img = subset_data[idx].reshape(28, 28)
        axes[0, i].imshow(original_img, cmap='gray')
        axes[0, i].set_title(f'Original: {subset_labels[idx]}')
        axes[0, i].axis('off')

        original_latent = latent_representations[idx]
        noise = np.random.normal(0, noise_scale, original_latent.shape)
        perturbed_latent = original_latent + noise

        with torch.no_grad():
            latent_tensor = torch.FloatTensor(perturbed_latent).unsqueeze(0).to(device)
            generated_image = model.decode(latent_tensor).cpu().numpy().squeeze()

        axes[1, i].imshow(generated_image.reshape(28, 28), cmap='gray')
        axes[1, i].set_title('Perturbed')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

    model.train()


# Main execution
if __name__ == '__main__':
    activation_function = "relu"  # Match what you used in training
    model_path = "google_model.pth"  # Your saved model path

    print("Loading MNIST data...")
    mnist = fetch_openml('mnist_784', version=1)
    labels = mnist.target.astype(int)

    normalizer = data_normalizer.DataNormalizer(method=activation_function)
    normalized_data = normalizer.fit_transform(mnist.data)

    print(f"Loading VAE model from {model_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VariationalAutoencoder(input_shape=784, activation_function=activation_function)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("VAE model loaded successfully!")

    n_samples = 5000
    subset_data = normalized_data[:n_samples]
    subset_labels = labels[:n_samples]

    print("Showing what the original data looks like...")
    show_sample_images(subset_data[:8], subset_labels[:8], "Original MNIST Samples")

    print("Showing sample reconstructions...")
    sample_indices = [0, 100, 200, 300, 500]
    for idx in sample_indices:
        show_reconstruction(model, subset_data[idx])

    print("Encoding dataset to latent space...")
    latents = encode_dataset(model, subset_data)

    print("Visualizing VAE latent space...")
    visualize_latent_space(latents, subset_labels, "VAE Latent Space - MNIST Digits")

    print("\nGenerating new samples from VAE...")
    generated_samples = generate_samples(model, latents, n_samples=8)
    show_generated_samples(generated_samples)

    print("\nInterpolating between different digits in VAE latent space...")
    interpolate_between_digits(model, latents, subset_labels, digit1=0, digit2=1, n_steps=8)
    interpolate_between_digits(model, latents, subset_labels, digit1=3, digit2=8, n_steps=8)

    print("\nExploring VAE latent space systematically...")
    explore_latent_space_grid(model, latents, grid_size=5, scale=2.0)

    print("\nPerturbing existing samples in VAE latent space...")
    perturb_existing_samples(model, subset_data, subset_labels, latents, n_samples=8, noise_scale=0.3)

    print("\nVAE visualization complete!")