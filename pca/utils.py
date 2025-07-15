import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms

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
