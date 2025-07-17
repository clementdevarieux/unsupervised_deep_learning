import numpy as np
import matplotlib.pyplot as plt

def plot_latent_space(W, map_lines, map_columns, shape=(28, 28)):
    fig, axes = plt.subplots(map_lines, map_columns, figsize=(map_columns * 2, map_lines * 2))
    
    for i in range(map_lines):
        for j in range(map_columns):
            image = W[(i, j)] * 255
            image = np.clip(image, 0, 255)
            axes[i, j].imshow(image.reshape(shape), cmap='gray')
            axes[i, j].axis('off')
            axes[i, j].set_title(f'({i},{j})', fontsize=8)
    
    plt.tight_layout()
    plt.suptitle('Kohonen Map Latent Space', fontsize=16, y=1.02)
    plt.show()

def display_latent_distribution(dataset, W):
    plt.figure(figsize=(10, 6))
    plt.scatter(dataset[:, 0], dataset[:, 1], alpha=0.5, label='Data Points')
    
    for (i, j), weight in W.items():
        plt.scatter(weight[0], weight[1], color='red', marker='x', s=100, label=f'Neuron ({i},{j})')
    
    plt.title('Latent Space Distribution')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()