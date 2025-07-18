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
            
            if len(shape) == 3:  # RGB
                reshaped_image = image.reshape(shape).astype(np.uint8)
                axes[i, j].imshow(reshaped_image)
            else:  # Niveaux de gris
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
    unique_labels = sorted(list(set(labels)))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    dominant_class_map = np.full((map_lines, map_columns), -1, dtype=int)
    class_counts_map = np.zeros((map_lines, map_columns))
    
    for (i, j), class_counter in position_class_counts.items():
        if class_counter:
            dominant_class_label = class_counter.most_common(1)[0][0]
            total_count = sum(class_counter.values())
            dominant_class_map[i, j] = label_to_int[dominant_class_label]
            class_counts_map[i, j] = total_count

    im1 = axes[0, 0].imshow(dominant_class_map, cmap='tab10', vmin=0, vmax=len(unique_labels)-1)
    axes[0, 0].set_title('Classe dominante par neurone')
    axes[0, 0].set_xlabel('Colonne')
    axes[0, 0].set_ylabel('Ligne')

    for i in range(map_lines):
        for j in range(map_columns):
            if dominant_class_map[i, j] != -1:
                class_label = unique_labels[dominant_class_map[i, j]]
                axes[0, 0].text(j, i, class_label, 
                               ha='center', va='center', fontweight='bold', fontsize=8)
    
    cbar1 = plt.colorbar(im1, ax=axes[0, 0])
    cbar1.set_label('Classes')
    cbar1.set_ticks(range(len(unique_labels)))
    cbar1.set_ticklabels([label[:6] for label in unique_labels])
    
    # 2. Carte du nombre d'échantillons par neurone
    im2 = axes[0, 1].imshow(class_counts_map, cmap='viridis')
    axes[0, 1].set_title('Nombre d\'échantillons par neurone')
    axes[0, 1].set_xlabel('Colonne')
    axes[0, 1].set_ylabel('Ligne')
    plt.colorbar(im2, ax=axes[0, 1], label='Nombre d\'échantillons')
    
    # 3. Distribution globale des classes
    class_distribution = Counter(labels)
    classes = sorted(class_distribution.keys())
    counts = [class_distribution[c] for c in classes]
    
    axes[1, 0].bar(range(len(classes)), counts, color='skyblue', alpha=0.7)
    axes[1, 0].set_title('Distribution globale des classes')
    axes[1, 0].set_xlabel('Classe')
    axes[1, 0].set_ylabel('Nombre d\'échantillons')
    axes[1, 0].set_xticks(range(len(classes)))
    axes[1, 0].set_xticklabels([label[:6] for label in classes], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.suptitle('Analyse de la Distribution des Classes sur la Carte de Kohonen', 
                 fontsize=14, y=1.02)
    plt.show()

    return position_class_counts, dominant_class_map