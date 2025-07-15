# fn k_means(
#     inputs_train: &[f32],
#     number_of_clusters: usize,
#     dimensions_of_inputs: usize,
#     number_of_points: usize,
# ) -> Vec<Vec<f32>> {
#     // initialiser des centres randoms
#     let mut vec_of_mu_k: Vec<Vec<f32>> = Vec::with_capacity(number_of_clusters);
#     // vec_of_mu_k= [mu_0, mu_1 ..., mu_k]
#     // mu_x = le centre du cluster x
#     // mu_x a le meme nombre d'éléments que une image
#     for k in 0..number_of_clusters{
#         let mut mu_k : Vec<f32> = Vec::with_capacity(dimensions_of_inputs);
#         for j in 0..dimensions_of_inputs{
#             mu_k.push(rand::thread_rng().gen_range(-1f32..1f32));
#         }
#         vec_of_mu_k.push(mu_k);
#     }

#     // création ensemble Sk
#     // Pour chaque points on vérifie à quel centre il appartient
#     // X -> pour tout point -> on vérifie s'il est plus proche d'un centre ou d'un autre
#     let mut old_vec_of_mu_k : Vec<Vec<f32>> = Vec::with_capacity(number_of_clusters);
#     for k in 0..number_of_clusters{
#         let temp_vec: Vec<f32> = vec![0.0; dimensions_of_inputs];
#         old_vec_of_mu_k.push(temp_vec);
#     }
#     let mut count = 0;
#     while old_vec_of_mu_k != vec_of_mu_k && count <= 100 {
#         let mut vec_of_Sk: Vec<Vec<Vec<f32>>> = Vec::with_capacity(number_of_clusters);

#         for k in 0..number_of_clusters {
#             let mut S_k: Vec<Vec<f32>> = Vec::new();
#             for n in 0..number_of_points {
#                 let mut distance_k: f32 = 0.0;
#                 for j in 0..dimensions_of_inputs {
#                     distance_k += (inputs_train[n * dimensions_of_inputs + j] - vec_of_mu_k[k][j])*(inputs_train[n * dimensions_of_inputs + j] - vec_of_mu_k[k][j]);
#                 }
#                 distance_k = distance_k.sqrt();
#                 for l in 0..number_of_clusters {
#                     if l != k {
#                         let mut distance_l: f32 = 0.0;
#                         for j in 0..dimensions_of_inputs {
#                             distance_l += (inputs_train[n * dimensions_of_inputs + j] - vec_of_mu_k[l][j])*(inputs_train[n * dimensions_of_inputs + j] - vec_of_mu_k[l][j]);
#                         }
#                         distance_l = distance_l.sqrt();
#                         if distance_k <= distance_l {
#                             let mut vec_to_push = Vec::with_capacity(dimensions_of_inputs);
#                             for i in 0..dimensions_of_inputs {
#                                 vec_to_push.push(inputs_train[n * dimensions_of_inputs + i]);
#                             }
#                             S_k.push(vec_to_push);
#                         }
#                     }
#                 }
#             }
#             vec_of_Sk.push(S_k);
#         }

#         //update mu_k
#         old_vec_of_mu_k = vec_of_mu_k;
#         vec_of_mu_k = Vec::with_capacity(number_of_clusters);
#         for k in 0..number_of_clusters {
#             let mut mu_k: Vec<f32> = vec![0.0; dimensions_of_inputs];
#             for n in &vec_of_Sk[k] {
#                 for i in 0..dimensions_of_inputs {
#                     mu_k[i] += n[i] / vec_of_Sk[k].len() as f32;
#                 }
#             }
#             vec_of_mu_k.push(mu_k);
#         }
#         count += 1;
#     }
#     vec_of_mu_k

# }


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from kmeans.decompression import decompress_image
from sklearn.cluster import KMeans


def k_means(inputs_train, number_of_clusters, dimensions_of_inputs, number_of_points, max_iter=100):
    # Initialisation aléatoire des centres
    # vec_of_mu_k = np.random.uniform(0, 1, (number_of_clusters, dimensions_of_inputs))
    vec_of_mu_k = KMeans(n_clusters=number_of_clusters, init='k-means++', n_init=1).fit(inputs_train).cluster_centers_

    for _ in range(max_iter):
        # Étape 1 : Affectation des points au cluster le plus proche
        clusters = [[] for _ in range(number_of_clusters)]

        for n in range(number_of_points):
            point = inputs_train[n]
            distances = np.linalg.norm(vec_of_mu_k - point, axis=1)
            closest_cluster = np.argmin(distances)
            clusters[closest_cluster].append(point)

        # Étape 2 : Mise à jour des centres
        new_vec_of_mu_k = []
        for k in range(number_of_clusters):
            if clusters[k]:  # Si le cluster n'est pas vide
                new_center = np.mean(clusters[k], axis=0)
            else:  # Si cluster vide, on réinitialise aléatoirement
                new_center = np.random.uniform(0, 1, dimensions_of_inputs)
            new_vec_of_mu_k.append(new_center)
        
        new_vec_of_mu_k = np.array(new_vec_of_mu_k)

        # Vérifier la convergence (si les centres ne changent plus beaucoup)
        if np.allclose(vec_of_mu_k, new_vec_of_mu_k, atol=1e-4):
            break

        vec_of_mu_k = new_vec_of_mu_k

    return vec_of_mu_k, clusters

def predict_kmeans(image, centers):
    distances = np.linalg.norm(centers - image, axis=1) 
    return np.argmin(distances)

def display_image(image, shape):
    plt.imshow(image.reshape(shape), cmap='gray')
    plt.show()

if __name__ == "__main__":
    mnist = fetch_openml('mnist_784', version=1)
    X = (mnist.data.astype(np.float32) / 255.0).to_numpy()
    # y = mnist.target.astype(int)
    
    # # n_samples = 2000
    
    # number_of_clusters = 50
    
    # centers, clusters = k_means(
    # X, number_of_clusters=number_of_clusters, dimensions_of_inputs=784, number_of_points=len(X)
    # )
    
    # np.save("kmeans/data/pred/centers_kmeans.npy", centers)
    
    # fig, axes = plt.subplots(1, number_of_clusters, figsize=(15, 3))
    # for i, ax in enumerate(axes):
    #     ax.imshow(centers[i].reshape(28, 28), cmap='gray')
    #     ax.axis("off")
    #     ax.set_title(f"Cluster {i}")
    # plt.suptitle("Centres des clusters (k=50)")
    # plt.show()
    
    
   ###PREDICTION
    image = X[5]
    
    display_image(image, (28, 28))
    
    centers = np.load("kmeans/data/pred/centers_kmeans.npy")

    predicted_index = predict_kmeans(image, centers)
    
    print(predicted_index)
    
    decompressed_image = decompress_image(predicted_index, "kmeans/data/pred/centers_kmeans.npy")
    display_image(decompressed_image, (28, 28))