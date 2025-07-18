import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from data_loader import load_images_from_folder, load_and_normalize_images
import os


def k_means(inputs_train, number_of_clusters, dimensions_of_inputs, number_of_points, max_iter=100):
    vec_of_mu_k = KMeans(n_clusters=number_of_clusters, init='k-means++', n_init=1).fit(inputs_train).cluster_centers_

    for _ in range(max_iter):
        clusters = [[] for _ in range(number_of_clusters)]

        for n in range(number_of_points):
            point = inputs_train[n]
            distances = np.linalg.norm(vec_of_mu_k - point, axis=1)
            closest_cluster = np.argmin(distances)
            clusters[closest_cluster].append(point)

        new_vec_of_mu_k = []
        for k in range(number_of_clusters):
            if clusters[k]:
                new_center = np.mean(clusters[k], axis=0)
            else:
                new_center = np.random.uniform(0, 1, dimensions_of_inputs)
            new_vec_of_mu_k.append(new_center)

        new_vec_of_mu_k = np.array(new_vec_of_mu_k)

        if np.allclose(vec_of_mu_k, new_vec_of_mu_k, atol=1e-4):
            break

        vec_of_mu_k = new_vec_of_mu_k

    return vec_of_mu_k, clusters

def predict_kmeans(image, centers):
    distances = np.linalg.norm(centers - image, axis=1)
    return np.argmin(distances)

def display_image(image, shape):
    image = image*255
    plt.imshow(image.reshape(shape), cmap='gray')
    plt.show()

def display_image_color(image, shape):
    if len(shape) == 3:  # Color image
        reshaped_img = np.clip(image.reshape(shape), 0, 1)
        plt.imshow(reshaped_img)
    else:  # Grayscale image
        reshaped_img = image.reshape(shape) * 255
        plt.imshow(reshaped_img, cmap='gray')
    plt.show()

def decompress_image(index, file_path):
    centers = np.load(file_path)
    return centers[index]

def plot_cluster_distribution(predicted_index, number_of_clusters, centers_file_path, dataset):
    centers = np.load(centers_file_path)
    number_of_clusters = len(centers)
    
    predicted_index = []
    for value in dataset:
        predicted_index.append(predict_kmeans(value, centers))
         
    cluster_sizes = [predicted_index.count(i) for i in range(number_of_clusters)]
    
    cmap = plt.get_cmap('tab20' if number_of_clusters <= 20 else 'hsv')
    colors = [cmap(i / number_of_clusters) for i in range(number_of_clusters)]

    plt.figure(figsize=(8, 5))
    plt.bar(range(number_of_clusters), cluster_sizes, color=colors)
    plt.xlabel('Index du cluster')
    plt.ylabel("Nombre d'images")
    plt.title("Répartition des images dans les clusters")
    plt.xticks(range(number_of_clusters))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def generate_data_from_cluster(cluster_index, centers_file_path, noise_level=0.1):
    centers = np.load(centers_file_path)
    center = centers[cluster_index]
    noise = np.random.normal(0, noise_level, size=center.shape)
    generated = center + noise
    generated = np.clip(generated, 0, 1)
    return generated

def display_comparison(X, centers_file_path, indices, noise_level=0.1, shape=(28, 28)):
    centers = np.load(centers_file_path)
    n = len(indices)
    plt.figure(figsize=(9, 3 * n))
    for i, idx in enumerate(indices):
        original = X[idx]
        cluster_idx = predict_kmeans(original, centers)
        reconstructed = centers[cluster_idx]
        generated = generate_data_from_cluster(cluster_idx, centers_file_path, noise_level)

        for j, (img, title) in enumerate(zip([original, reconstructed, generated],
                                             ["Originale", "Reconstruction", "Génération"])):
            plt.subplot(n, 3, i * 3 + j + 1)
            plt.imshow(img.reshape(shape) * 255, cmap='gray')
            plt.axis('off')
            plt.title(f"{title} (idx={idx})" if j == 0 else title)
    plt.tight_layout()
    plt.show()

def display_comparison_fruits(X, centers_file_path, indices, noise_level=0.1, shape=(32, 32, 3)):
    centers = np.load(centers_file_path)
    n = len(indices)
    plt.figure(figsize=(9, 3 * n))
    for i, idx in enumerate(indices):
        original = X[idx]
        cluster_idx = predict_kmeans(original, centers)
        reconstructed = centers[cluster_idx]
        generated = generate_data_from_cluster(cluster_idx, centers_file_path, noise_level)

        for j, (img, title) in enumerate(zip([original, reconstructed, generated],
                                             ["Originale", "Reconstruction", "Génération"])):
            plt.subplot(n, 3, i * 3 + j + 1)
            
            # Handle different image types
            if len(shape) == 3:  # Color image (H, W, C)
                reshaped_img = img.reshape(shape)
                # Clip values to valid range [0, 1] for color images
                reshaped_img = np.clip(reshaped_img, 0, 1)
                plt.imshow(reshaped_img)
            else:  # Grayscale image (H, W)
                reshaped_img = img.reshape(shape) * 255
                reshaped_img = np.clip(reshaped_img, 0, 255)
                plt.imshow(reshaped_img, cmap='gray')
            
            plt.axis('off')
            plt.title(f"{title} (idx={idx})" if j == 0 else title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # mnist = fetch_openml('mnist_784', version=1)
    # X = (mnist.data.astype(np.float32) / 255.0).to_numpy()

    input_dir = 'data/fruits_dataset'

    images, labels = load_images_from_folder(input_dir, max_images_per_class=20, target_size=(32, 32))

    normalized_images = load_and_normalize_images(images, flatten=True)

    X = np.array(normalized_images)

    dimensions_of_inputs = X.shape[1]
    number_of_clusters = 100
    save_name = "fruits_kmeans_100"

    os.makedirs("kmeans/data/fruits/pred", exist_ok=True)

    # centers, clusters = k_means(
    # X, number_of_clusters=number_of_clusters, dimensions_of_inputs=784, number_of_points=len(X)
    # )

    centers, clusters = k_means(
        X, 
        number_of_clusters=number_of_clusters, 
        dimensions_of_inputs=dimensions_of_inputs, 
        number_of_points=len(X)
    )
    
    np.save(f"kmeans/data/fruits/pred/{save_name}.npy", centers)
    
    # fig, axes = plt.subplots(1, number_of_clusters, figsize=(15, 3))
    # for i, ax in enumerate(axes):
    #     ax.imshow(centers[i].reshape(28, 28), cmap='gray')
    #     ax.axis("off")
    #     ax.set_title(f"Cluster {i}")
    # plt.suptitle("Centres des clusters (k=50)")
    # plt.show()


   ### PREDICTION
    # image = X[4]
    
    # display_image(image, (28, 28))

    # centers = np.load("kmeans/data/pred/centers_kmeans.npy")

    # predicted_index = predict_kmeans(image, centers)

    # print(predicted_index)

    # decompressed_image = decompress_image(predicted_index, "kmeans/data/pred/centers_kmeans.npy")
    # display_image(decompressed_image, (28, 28))
    
    # generated_image = generate_data_from_cluster(predicted_index, "kmeans/data/pred/centers_kmeans.npy")
    # display_image(generated_image, (28, 28))
    
    
    ## VISUALISATION
    # plot_cluster_distribution(X, number_of_clusters, "kmeans/data/pred/centers_kmeans_100.npy", X)
    # display_comparison(X, f"kmeans/data/fruits/pred/{save_name}.npy", indices=[0, 1, 2, 3, 4], noise_level=0.2, shape=(32, 32))

    display_comparison_fruits(X, f"kmeans/data/fruits/pred/{save_name}.npy", indices=[0, 1, 2, 3, 4], noise_level=0.2, shape=(32, 32, 3))