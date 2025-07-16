import numpy as np
import matplotlib.pyplot as plt

def decompress_image(index, file_path):
    centers = np.load(file_path)
    
    return centers[index]
    
if __name__ == "__main__":
    
    centers = decompress_image(0, "kmeans/data/pred/centers_kmeans.npy")
    print(centers.shape)
    
    plt.imshow(centers.reshape(28, 28), cmap='gray')
    plt.show()
    
    