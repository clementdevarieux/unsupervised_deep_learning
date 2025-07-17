import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

def konoha_map(map_lines, map_columns, learning_rate, gamma, dataset, num_iterations):
    
    W = {}
    
    # on initialise les poids W de LA CARTE
    for i in range(map_lines):
        for j in range(map_columns):
            random_index = np.random.randint(0, len(dataset))
            W[(i, j)] = dataset[random_index]
    

    for iter in range(num_iterations):
        # on choisit un échantillon aléatoire
        random_index = np.random.randint(0, len(dataset))
        S_j = dataset[random_index]

        distances = {}

        for i in range(map_lines):
            for j in range(map_columns):
                # on calcule la distance entre l'échantillon et les poids
                distances[(i , j)] = np.linalg.norm(W[(i, j)] - S_j)

        # on trouve le neurone gagnant
        winner_index = min(distances, key=distances.get)

        # on met à jour les poids du neurone gagnant
        W[winner_index] += learning_rate * (S_j - W[winner_index]) \
            * np.exp((np.linalg.norm(np.array(winner_index) - np.array([map_lines, map_columns])) ** 2)/(-2 * gamma))
        
        if iter % 1000 == 0:
            gamma *= 0.99 

    return W

def display_image(image, shape):
    image = image*255
    plt.imshow(image.reshape(shape), cmap='gray')
    plt.show()

def display_kohonen_map(W, map_lines, map_columns, shape=(28, 28)):
    """Affiche la carte de Kohonen complète sous forme de grille"""
    fig, axes = plt.subplots(map_lines, map_columns, figsize=(map_columns*2, map_lines*2))
    
    for i in range(map_lines):
        for j in range(map_columns):
            if isinstance(W, dict):
                image = W[(i, j)] * 255
            else:
                image = W[i, j] * 255
            
            axes[i, j].imshow(image.reshape(shape), cmap='gray')
            axes[i, j].axis('off')
            axes[i, j].set_title(f'({i},{j})', fontsize=8)
    
    plt.tight_layout()
    plt.suptitle('Carte de Kohonen - Poids des neurones', fontsize=16, y=1.02)
    plt.show()

if __name__ == "__main__":
    mnist = fetch_openml('mnist_784', version=1)
    dataset = mnist.data.values / 255.0  
    map_lines = 10
    map_columns = 10    
    learning_rate = 0.5
    gamma = 10.0
    num_iterations = 100000

    W = konoha_map(map_lines, map_columns, learning_rate, gamma, dataset, num_iterations)
    
    display_kohonen_map(W, map_lines, map_columns)
    
    # Sauvegarder les poids
    np.save('data/kohonen_map_weights.npy', W)
    
    # Puis les charger et afficher
    with open('data/kohonen_map_weights.npy', 'rb') as f:
        W_loaded = np.load(f, allow_pickle=True).item()
        display_kohonen_map(W_loaded, map_lines, map_columns)