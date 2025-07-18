import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from tqdm import tqdm
import os

def konoha_map(map_lines, 
               map_columns, 
               learning_rate, 
               gamma, 
               dataset, 
               num_iterations, 
               batch_size=200,
               save_every=1000,
               image_shape=(28, 28)):
    
    W = {}
    initial_lr = learning_rate
    timestamp = str(np.datetime64('now')).replace(':', '-').replace('T', '_')
   
    # on initialise les poids W de LA CARTE
    for i in range(map_lines):
        for j in range(map_columns):
            random_index = np.random.randint(0, len(dataset))
            W[(i, j)] = dataset[random_index]
    

    tqdm_iter = tqdm(range(num_iterations))
    for iter in tqdm_iter:
        
        batch_indices = np.random.choice(len(dataset), size=batch_size, replace=False)
        
        for idx in batch_indices:
            S_j = dataset[idx]

            current_learning_rate = initial_lr * np.exp(-iter / num_iterations)

            distances = {}

            for i in range(map_lines):
                for j in range(map_columns):
                    # on calcule la distance entre l'échantillon et les poids
                    distances[(i , j)] = np.linalg.norm(W[(i, j)] - S_j)

            # on trouve le neurone gagnant
            winner_index = min(distances, key=distances.get)

            # on met à jour les poids en comparant avec le neurone gagnant
            for i in range(map_lines):
                for j in range(map_columns):
                    distance = np.linalg.norm(np.array([i, j]) - np.array(winner_index))
                    W[(i, j)] += current_learning_rate * np.exp(-(distance ** 2) / (2 * gamma)) * (S_j - W[(i, j)])

        if iter % save_every == 0 or iter == 0:
            gamma *= 0.995 
            fig, axes = display_kohonen_map(W, map_lines, map_columns, shape=image_shape,show=False)
            os.makedirs('data/kohonen_evolution/mnist', exist_ok=True)
            fig.savefig(f'data/kohonen_evolution/mnist/{timestamp}_iter_{iter:06d}_batch_size_{batch_size}_gamma_{gamma}_lr_{current_learning_rate:.6f}.png', 
                       dpi=150, bbox_inches='tight')
            plt.close(fig)

    return W


def display_image(image, shape):
    image = image*255
    image = np.clip(image, 0, 255)
    plt.imshow(image.reshape(shape), cmap='gray')
    plt.show()

def display_kohonen_map(W, map_lines, map_columns, shape=(28, 28), show=True):
    """Affiche la carte de Kohonen complète sous forme de grille"""
    fig, axes = plt.subplots(map_lines, map_columns, figsize=(map_columns*2, map_lines*2))
    
    for i in range(map_lines):
        for j in range(map_columns):
            if isinstance(W, dict):
                image = W[(i, j)] * 255
            else:
                image = W[i, j] * 255
            
            image = np.clip(image, 0, 255)
            if len(shape) == 3:  # RGB
                reshaped_image = image.reshape(shape).astype(np.uint8)
                axes[i, j].imshow(reshaped_image)
            else:  # Niveaux de gris
                axes[i, j].imshow(image.reshape(shape), cmap='gray')
            axes[i, j].axis('off')
            axes[i, j].set_title(f'({i},{j})', fontsize=8)
    
    plt.tight_layout()
    plt.suptitle('Carte de Kohonen - Poids des neurones', fontsize=16, y=1.02)

    if show:
        plt.show()

    return fig, axes

if __name__ == "__main__":
    mnist = fetch_openml('mnist_784', version=1)
    dataset = mnist.data.values / 255.0  

    map_lines = 20
    map_columns = 20    
    learning_rate = 0.5
    gamma = 1.0
    num_iterations = 100

    W = konoha_map(map_lines, 
                   map_columns, 
                   learning_rate, 
                   gamma, 
                   dataset, 
                   num_iterations,
                   batch_size=200,
                   save_every=10)