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

def compression_decompression_pipeline(data, weights, num_samples=10, map_lines=10, map_columns=10):
    test_indices = np.random.choice(len(data), num_samples, replace=False)
    test_data = data[test_indices]

    compressed = compress_data(test_data, weights)

    decompressed = decompress_data(compressed, weights)

    fig, axes = plt.subplots(3, min(5, num_samples), figsize=(15, 9))
    for i in range(min(5, num_samples)):
        # Original
        axes[0, i].imshow(test_data[i].reshape(28, 28), cmap="gray")
        axes[0, i].set_title(f"Original {i + 1}")
        axes[0, i].axis("off")

        # Position compressée - afficher le représentant à cette position
        pos = compressed[i]
        neuron_weight = weights[pos]
        axes[1, i].imshow(neuron_weight.reshape(28, 28), cmap="gray")
        axes[1, i].set_title(f"Représentant ({pos[0]}, {pos[1]})")
        axes[1, i].axis("off")

        # Décompressé (même chose que le représentant)
        # en fait, quand on compresse, on envoie juste l'info du représentant,
        # donc dans un sens ça compresse parce qu'on envoie moins d'infos
        # et du coup la décompression c'est juste vraiment de récupérer le représentant, 
        # donc on s'approche par exemple d'un 9 si c'était le 9 qui a été compressé, 
        # et juste ce 9 ressemblera forcément au 9 qui a été compressé
        axes[2, i].imshow(decompressed[i].reshape(28, 28), cmap="gray")
        axes[2, i].set_title(f"Reconstruit {i + 1}")
        axes[2, i].axis("off")

    plt.tight_layout()
    plt.suptitle("Pipeline Compression/Décompression", fontsize=16, y=1.02)
    plt.show()

    return compressed, decompressed