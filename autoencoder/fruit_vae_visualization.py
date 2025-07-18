import torch
import matplotlib.pyplot as plt
import numpy as np


def show_reconstruction(model, sample, epoch=None, image_shape=(3, 32, 32), n_samples=5):
    """
    Show reconstructions of n_samples in a single figure.
    Creates a 2xn_samples grid: top row = originals, bottom row = reconstructions
    """
    device = next(model.parameters()).device
    model.eval()

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
            original_img = original.reshape(32, 32, 3)
            reconstructed_img = reconstruction.reshape(32, 32, 3)

            # Ensure values are in [0, 1] range for display
            original_img = np.clip(original_img, 0, 1)
            reconstructed_img = np.clip(reconstructed_img, 0, 1)

            # Plot original (top row)
            axes[0, i].imshow(original_img)
            axes[0, i].set_title(f'Original {i + 1}')
            axes[0, i].axis('off')

            # Plot reconstruction (bottom row)
            axes[1, i].imshow(reconstructed_img)
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
            original_img = original.reshape(32, 32, 3)
            reconstructed_img = reconstruction.reshape(32, 32, 3)

            # Ensure values are in [0, 1] range for display
            original_img = np.clip(original_img, 0, 1)
            reconstructed_img = np.clip(reconstructed_img, 0, 1)

            # Plot original (top row)
            axes[0, i].imshow(original_img)
            axes[0, i].set_title(f'Original {i + 1}')
            axes[0, i].axis('off')

            # Plot reconstruction (bottom row)
            axes[1, i].imshow(reconstructed_img)
            axes[1, i].set_title(f'Reconstructed {i + 1}')
            axes[1, i].axis('off')

    title = f'Fruit VAE Reconstructions - Epoch {epoch}' if epoch else 'Fruit VAE Reconstructions'
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

    model.train()


def visualize_latent_space(model, data, labels=None, title="Fruit VAE Latent Space"):
    """
    Visualize the 2D latent space with fruit classes color-coded.
    """
    from variational_latent_analysis import encode_dataset

    # Get latent representations
    if hasattr(data, 'shape') and len(data.shape) == 2:
        latent_representations = encode_dataset(model, data)
    else:
        latent_representations = data  # Assume it's already encoded

    plt.figure(figsize=(12, 8))

    if labels is not None:
        unique_classes = np.unique(labels)
        n_classes = len(unique_classes)

        # Use distinct colors for each fruit class
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))

        for i, class_label in enumerate(unique_classes):
            mask = np.array(labels) == class_label
            plt.scatter(latent_representations[mask, 0], latent_representations[mask, 1],
                        c=[colors[i]], label=f'{class_label}', s=10, alpha=0.7)

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(latent_representations[:, 0], latent_representations[:, 1],
                    s=10, alpha=0.7, c='blue')

    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def show_sample_images(images, labels=None, title="Sample Fruit Images", n=8):
    """
    Display sample fruit images from the dataset.
    """
    if hasattr(images, 'numpy'):
        images = images.numpy()

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        if i < len(images) and i < n:
            img = images[i].reshape(32, 32, 3)
            img = np.clip(img, 0, 1)  # Ensure values are in [0, 1]
            ax.imshow(img)
            if labels is not None:
                ax.set_title(f'{labels[i]}')
            else:
                ax.set_title(f'Sample {i + 1}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def show_generated_samples(model, latent_representations, n_samples=8, title="Generated Fruit Images"):
    """
    Generate and display new fruit images by sampling from the latent space.
    """
    from variational_latent_analysis import generate_samples

    generated_images = generate_samples(model, latent_representations, n_samples=n_samples)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    for i, ax in enumerate(axes.flat):
        if i < n_samples:
            img = generated_images[i].reshape(32, 32, 3)
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_title(f'Generated #{i + 1}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def interpolate_between_fruits(model, latent_representations, labels, fruit1, fruit2, n_steps=8):
    """
    Interpolate between two different fruit classes in the latent space.
    """
    device = next(model.parameters()).device
    model.eval()

    # Find indices for each fruit class
    fruit1_indices = np.where(np.array(labels) == fruit1)[0]
    fruit2_indices = np.where(np.array(labels) == fruit2)[0]

    if len(fruit1_indices) == 0 or len(fruit2_indices) == 0:
        print(f"Could not find examples of {fruit1} or {fruit2}")
        return

    # Use random examples from each class
    start_idx = np.random.choice(fruit1_indices)
    end_idx = np.random.choice(fruit2_indices)

    start_latent = latent_representations[start_idx]
    end_latent = latent_representations[end_idx]

    # Create interpolation steps
    interpolation_steps = np.linspace(0, 1, n_steps)
    interpolated_latents = []

    for step in interpolation_steps:
        interpolated = start_latent * (1 - step) + end_latent * step
        interpolated_latents.append(interpolated)

    interpolated_latents = np.array(interpolated_latents)

    # Generate images from interpolated latents
    with torch.no_grad():
        latent_tensor = torch.FloatTensor(interpolated_latents).to(device)
        interpolated_images = model.decode(latent_tensor).cpu().numpy()

    # Display the interpolation
    fig, axes = plt.subplots(1, n_steps, figsize=(2 * n_steps, 3))
    fig.suptitle(f'Fruit VAE Interpolation: {fruit1} → {fruit2}', fontsize=16, fontweight='bold')

    for i in range(n_steps):
        img = interpolated_images[i].reshape(32, 32, 3)
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(f'Step {i + 1}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
    model.train()


def explore_latent_space_grid(model, latent_representations, grid_size=5, scale=2.0):
    """
    Explore the latent space by generating images on a regular grid.
    """
    device = next(model.parameters()).device
    model.eval()

    # Calculate latent space statistics
    latent_mean = np.mean(latent_representations, axis=0)
    latent_std = np.std(latent_representations, axis=0)

    # Create grid points
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

    # Generate images from grid points
    with torch.no_grad():
        latent_tensor = torch.FloatTensor(grid_latents).to(device)
        generated_images = model.decode(latent_tensor).cpu().numpy()

    # Display the grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle('Fruit VAE Latent Space Grid Exploration', fontsize=16, fontweight='bold')

    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            img = generated_images[idx].reshape(32, 32, 3)
            img = np.clip(img, 0, 1)
            axes[i, j].imshow(img)
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()
    model.train()


def perturb_existing_samples(model, data_samples, labels, latent_representations, n_samples=8, noise_scale=0.5):
    """
    Show original samples vs their perturbed versions in latent space.
    """
    device = next(model.parameters()).device
    model.eval()

    # Select random samples
    indices = np.random.choice(len(latent_representations), n_samples, replace=False)

    fig, axes = plt.subplots(2, n_samples, figsize=(2 * n_samples, 6))
    fig.suptitle('Original vs Perturbed Fruit Images', fontsize=16, fontweight='bold')

    for i, idx in enumerate(indices):
        # Show original
        original_img = data_samples[idx].reshape(32, 32, 3)
        original_img = np.clip(original_img, 0, 1)
        axes[0, i].imshow(original_img)
        axes[0, i].set_title(f'Original: {labels[idx]}')
        axes[0, i].axis('off')

        # Generate perturbed version
        original_latent = latent_representations[idx]
        noise = np.random.normal(0, noise_scale, original_latent.shape)
        perturbed_latent = original_latent + noise

        with torch.no_grad():
            latent_tensor = torch.FloatTensor(perturbed_latent).unsqueeze(0).to(device)
            generated_image = model.decode(latent_tensor).cpu().numpy().squeeze()

        perturbed_img = generated_image.reshape(32, 32, 3)
        perturbed_img = np.clip(perturbed_img, 0, 1)
        axes[1, i].imshow(perturbed_img)
        axes[1, i].set_title('Perturbed')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()
    model.train()


def comprehensive_fruit_analysis(model, data, labels, model_name="Fruit VAE"):
    """
    Run a comprehensive analysis of the trained fruit VAE.
    """
    from variational_latent_analysis import encode_dataset

    print(f"=== {model_name} Comprehensive Analysis ===\n")

    # 1. Show sample original images
    print("1. Showing sample original fruit images...")
    show_sample_images(data[:8], labels[:8], "Original Fruit Dataset Samples")

    # 2. Show reconstructions
    print("2. Showing sample reconstructions...")
    random_indices = np.random.choice(len(data), 5, replace=False)
    sample_data = [data[i] for i in random_indices]
    show_multiple_reconstructions(model, sample_data, image_shape=(3, 32, 32))

    # 3. Encode to latent space
    print("3. Encoding dataset to latent space...")
    latents = encode_dataset(model, data)

    # 4. Visualize latent space
    print("4. Visualizing latent space...")
    visualize_latent_space(model, latents, labels, f"{model_name} Latent Space")

    # 5. Generate new samples
    print("5. Generating new fruit images...")
    show_generated_samples(model, latents, n_samples=8)

    # 6. Interpolate between fruit classes
    print("6. Interpolating between different fruit classes...")
    unique_fruits = list(set(labels))
    if len(unique_fruits) >= 2:
        fruit1, fruit2 = np.random.choice(unique_fruits, 2, replace=False)
        interpolate_between_fruits(model, latents, labels, fruit1, fruit2)

    # 7. Explore latent space systematically
    print("7. Exploring latent space grid...")
    explore_latent_space_grid(model, latents, grid_size=5)

    # 8. Perturb existing samples
    print("8. Showing perturbations of existing samples...")
    perturb_existing_samples(model, data, labels, latents, n_samples=6, noise_scale=0.3)

    print(f"\n=== {model_name} Analysis Complete! ===")

    return latents