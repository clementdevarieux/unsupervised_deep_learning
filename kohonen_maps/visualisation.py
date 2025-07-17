import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from compression import compress_data

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

def analyze_class_distribution_on_map(data, labels, weights, map_lines, map_columns):
    
    compressed_positions = compress_data(data, weights)

    position_class_counts = {}
    
    for pos, label in zip(compressed_positions, labels):
        if isinstance(pos, np.ndarray):
            pos = tuple(pos)
        
        if pos not in position_class_counts:
            position_class_counts[pos] = Counter()
        position_class_counts[pos][label] += 1

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Carte des classes dominantes
    dominant_class_map = np.full((map_lines, map_columns), -1, dtype=int)
    class_counts_map = np.zeros((map_lines, map_columns))
    
    for (i, j), class_counter in position_class_counts.items():
        if class_counter:
            dominant_class = class_counter.most_common(1)[0][0]
            total_count = sum(class_counter.values())
            dominant_class_map[i, j] = dominant_class
            class_counts_map[i, j] = total_count

    im1 = axes[0, 0].imshow(dominant_class_map, cmap='tab10', vmin=0, vmax=9)
    axes[0, 0].set_title('Classe dominante par neurone')
    axes[0, 0].set_xlabel('Colonne')
    axes[0, 0].set_ylabel('Ligne')

    for i in range(map_lines):
        for j in range(map_columns):
            if dominant_class_map[i, j] != -1:
                axes[0, 0].text(j, i, str(dominant_class_map[i, j]), 
                               ha='center', va='center', fontweight='bold', fontsize=8)
    
    plt.colorbar(im1, ax=axes[0, 0], label='Classe (0-9)')
    
    # 2. Carte du nombre d'échantillons par neurone
    im2 = axes[0, 1].imshow(class_counts_map, cmap='viridis')
    axes[0, 1].set_title('Nombre d\'échantillons par neurone')
    axes[0, 1].set_xlabel('Colonne')
    axes[0, 1].set_ylabel('Ligne')
    plt.colorbar(im2, ax=axes[0, 1], label='Nombre d\'échantillons')
    
    # 3. Distribution globale des classes
    all_labels = [int(label) for label in labels]
    class_distribution = Counter(all_labels)
    classes = sorted(class_distribution.keys())
    counts = [class_distribution[c] for c in classes]
    
    axes[1, 0].bar(classes, counts, color='skyblue', alpha=0.7)
    axes[1, 0].set_title('Distribution globale des classes')
    axes[1, 0].set_xlabel('Classe')
    axes[1, 0].set_ylabel('Nombre d\'échantillons')
    axes[1, 0].set_xticks(classes)
    
    plt.tight_layout()
    plt.suptitle('Analyse de la Distribution des Classes sur la Carte de Kohonen', 
                 fontsize=14, y=1.02)
    plt.show()

    return position_class_counts, dominant_class_map