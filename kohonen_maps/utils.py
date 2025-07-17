import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from sklearn.datasets import fetch_openml
import os

def load_mnist(number_of_samples=10000):
    # Define transforms
    transform = transforms.Compose([transforms.ToTensor()])

    # Download and load training dataset
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )

    # Download and load test dataset
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=number_of_samples, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=number_of_samples, shuffle=False)

    return train_loader, test_loader

def print_mnist_images(images, labels):
    images = images.numpy()
    num_images = images.shape[0]
    
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(8, 8, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(labels[i].item())
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def load_and_standardize_data(num_samples=10000):
    mnist = fetch_openml("mnist_784", version=1, parser="auto")
    data = mnist.data[:num_samples].values.astype(np.float32)
    labels = mnist.target[:num_samples].values.astype(int)

    data_normalized = data / 255.0

    return data_normalized, labels


def save_results(weights, map_lines, map_columns, filename="kohonen_map_weights.npy"):
    weights_array = np.zeros((map_lines, map_columns, len(list(weights.values())[0])))

    for i in range(map_lines):
        for j in range(map_columns):
            weights_array[i, j] = weights[(i, j)]

    os.makedirs("data/kohonen/mnist/weights", exist_ok=True)
    np.save(f"data/kohonen/mnist/weights/{filename}", weights_array)