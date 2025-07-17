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

def generate_new_samples(weights, map_lines, map_columns, num_samples=10):
    new_samples = generate_new_data(weights, num_samples, map_lines, map_columns)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(min(10, num_samples)):
        row = i // 5
        col = i % 5
        axes[row, col].imshow(new_samples[i].reshape(28, 28), cmap="gray")
        axes[row, col].set_title(f"Échantillon {i + 1}")
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.suptitle("Nouveaux échantillons générés", fontsize=16, y=1.02)
    plt.show()

    return new_samples