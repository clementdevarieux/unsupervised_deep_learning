import numpy as np
import matplotlib.pyplot as plt

def generate_new_data(W, num_samples, map_lines, map_columns):
    """Generate new data samples using the trained Kohonen map weights."""
    new_samples = []
    
    for _ in range(num_samples):
        # Randomly select a position on the Kohonen map
        i = np.random.randint(0, map_lines)
        j = np.random.randint(0, map_columns)
        
        # Retrieve the corresponding weight vector
        new_sample = W[(i, j)]
        new_samples.append(new_sample)
    
    return np.array(new_samples)

# def generate_new_samples(weights, map_lines, map_columns, num_samples=10, image_shape=(28, 28)):
#     new_samples = generate_new_data(weights, num_samples, map_lines, map_columns)

#     fig, axes = plt.subplots(2, 5, figsize=(15, 6))
#     for i in range(min(10, num_samples)):
#         row = i // 5
#         col = i % 5
#         axes[row, col].imshow(new_samples[i].reshape(28, 28), cmap="gray")
#         axes[row, col].set_title(f"Échantillon {i + 1}")
#         axes[row, col].axis("off")

#     plt.tight_layout()
#     plt.suptitle("Nouveaux échantillons générés", fontsize=16, y=1.02)
#     plt.show()

#     return new_samples

def generate_new_samples(weights, map_lines, map_columns, num_samples, image_shape=(28, 28)):
    # Générer des positions aléatoires sur la carte
    positions = []
    for _ in range(num_samples):
        i = np.random.randint(0, map_lines)
        j = np.random.randint(0, map_columns)
        positions.append((i, j))
    
    # Générer les échantillons à partir des poids correspondants
    new_samples = []
    for pos in positions:
        new_samples.append(weights[pos])
    
    new_samples = np.array(new_samples)
    
    # Visualisation
    cols = min(5, num_samples)
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_samples):
        row = i // cols
        col = i % cols
        
        # Normaliser l'image pour l'affichage
        sample_image = new_samples[i] * 255
        sample_image = np.clip(sample_image, 0, 255)
        
        if len(image_shape) == 3:  # RGB
            reshaped_image = sample_image.reshape(image_shape).astype(np.uint8)
            axes[row, col].imshow(reshaped_image)
        else:  # Niveaux de gris
            axes[row, col].imshow(sample_image.reshape(image_shape), cmap="gray")
        
        axes[row, col].set_title(f'Generated {i+1}\nPos: {positions[i]}', fontsize=10)
        axes[row, col].axis('off')
    
    # Masquer les axes inutilisés
    for i in range(num_samples, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Generated New Samples from Kohonen Map', fontsize=16, y=1.02)
    plt.show()
    
    return new_samples, positions