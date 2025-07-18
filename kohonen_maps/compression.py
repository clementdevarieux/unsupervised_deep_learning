import numpy as np
import matplotlib.pyplot as plt

def compress_data(data, weights):
    compressed_data = []
    for sample in data:
        distances = {index: np.linalg.norm(weight - sample) for index, weight in weights.items()}
        winner_index = min(distances, key=distances.get)
        compressed_data.append(winner_index)
    return compressed_data 

def decompress_data(compressed_data, weights):
    decompressed_data = []
    for index in compressed_data:
        if isinstance(index, np.ndarray):
            index = tuple(index)
        decompressed_data.append(weights[index])
    return np.array(decompressed_data)

def compression_decompression_pipeline(data, 
                                       weights, 
                                       num_samples=10, 
                                       map_lines=10, 
                                       map_columns=10, 
                                       image_shape=(28, 28)):
    test_indices = np.random.choice(len(data), num_samples, replace=False)
    test_data = data[test_indices]

    compressed = compress_data(test_data, weights)

    decompressed = decompress_data(compressed, weights)

    fig, axes = plt.subplots(3, min(5, num_samples), figsize=(15, 9))
    for i in range(min(5, num_samples)):
        # Original
        original_image = test_data[i] * 255
        original_image = np.clip(original_image, 0, 255)

        if len(image_shape) == 3:  # RGB
            original_reshaped = original_image.reshape(image_shape).astype(np.uint8)
            axes[0, i].imshow(original_reshaped)
        else:  # Niveaux de gris
            axes[0, i].imshow(original_image.reshape(image_shape), cmap='gray')
        
        axes[0, i].set_title(f"Original {i + 1}")
        axes[0, i].axis("off")

        # Position compressée - afficher le représentant à cette position
        pos = compressed[i]
        neuron_weight = weights[pos]

        neuron_image = neuron_weight * 255
        neuron_image = np.clip(neuron_image, 0, 255)
        
        if len(image_shape) == 3:  # RGB
            neuron_reshaped = neuron_image.reshape(image_shape).astype(np.uint8)
            axes[1, i].imshow(neuron_reshaped)
        else:  # Niveaux de gris
            axes[1, i].imshow(neuron_image.reshape(image_shape), cmap="gray")
        
        axes[1, i].set_title(f"Représentant ({pos[0]}, {pos[1]})")
        axes[1, i].axis("off")

        # Décompressé (même chose que le représentant)
        # en fait, quand on compresse, on envoie juste l'info du représentant,
        # donc dans un sens ça compresse parce qu'on envoie moins d'infos
        # et du coup la décompression c'est juste vraiment de récupérer le représentant, 
        # donc on s'approche par exemple d'un 9 si c'était le 9 qui a été compressé, 
        # et juste ce 9 ressemblera forcément au 9 qui a été compressé
        decompressed_image = decompressed[i] * 255
        decompressed_image = np.clip(decompressed_image, 0, 255)
        
        if len(image_shape) == 3:  # RGB
            decompressed_reshaped = decompressed_image.reshape(image_shape).astype(np.uint8)
            axes[2, i].imshow(decompressed_reshaped)
        else:  # Niveaux de gris
            axes[2, i].imshow(decompressed_image.reshape(image_shape), cmap='gray')
        
        axes[2, i].set_title(f'Decompressed {i+1}')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.suptitle("Pipeline Compression/Décompression", fontsize=16, y=1.02)
    plt.show()

    return compressed, decompressed