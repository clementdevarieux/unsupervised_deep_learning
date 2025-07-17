import torch
import matplotlib.pyplot as plt
import numpy as np


def show_reconstruction(model, sample, epoch=None):
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        sample_tensor = torch.FloatTensor(sample).to(device)
        if len(sample_tensor.shape) == 1:
            sample_tensor = sample_tensor.unsqueeze(0)

        reconstruction = model(sample_tensor).cpu().numpy().squeeze()
        original = sample_tensor.cpu().numpy().squeeze()

    original_img = original.reshape(28, 28)
    reconstructed_img = reconstruction.reshape(28, 28)

    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_img, cmap='gray')
    plt.title('Reconstructed')
    plt.axis('off')

    title = f'MNIST Reconstruction - Epoch {epoch}' if epoch else 'MNIST Reconstruction'
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

    model.train()


def visualize_latent(model, latent_representations, labels=None):
    device = next(model.parameters()).device
    model.eval()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Scatter plot of latent representations
    if labels is not None:
        scatter = ax1.scatter(latent_representations[:, 0], latent_representations[:, 1],
                              c=labels, cmap='tab10', alpha=0.6, s=1)
        plt.colorbar(scatter, ax=ax1)
    else:
        ax1.scatter(latent_representations[:, 0], latent_representations[:, 1],
                    alpha=0.6, s=1)

    ax1.set_title('Latent Space Representation')
    ax1.set_xlabel('Latent Dimension 1')
    ax1.set_ylabel('Latent Dimension 2')
    ax1.grid(True, alpha=0.3)

    x_min, x_max = latent_representations[:, 0].min(), latent_representations[:, 0].max()
    y_min, y_max = latent_representations[:, 1].min(), latent_representations[:, 1].max()

    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding

    x_grid = np.linspace(x_min, x_max, 10)
    y_grid = np.linspace(y_min, y_max, 10)

    grid_images = np.zeros((10, 10, 28, 28))

    with torch.no_grad():
        for i, x in enumerate(x_grid):
            for j, y in enumerate(y_grid):
                latent_point = torch.FloatTensor([[x, y]]).to(device)
                decoded = model.decoder(latent_point).cpu().numpy().squeeze()
                grid_images[j, i] = decoded.reshape(28, 28)

    full_grid = np.zeros((280, 280))  # 10*28 x 10*28
    for i in range(10):
        for j in range(10):
            full_grid[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = grid_images[i, j]

    ax2.imshow(full_grid, cmap='gray')
    ax2.set_title('Latent Space Grid (10x10)')
    ax2.set_xlabel('Latent Dimension 1')
    ax2.set_ylabel('Latent Dimension 2')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

    model.train()