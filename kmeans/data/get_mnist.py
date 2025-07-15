import torch
import torchvision
from torchvision import transforms

# Define transforms
transform = transforms.Compose([transforms.ToTensor()])

# Download and load training dataset
train_dataset = torchvision.datasets.MNIST(
    root='kmeans/data', 
    train=True, 
    download=True, 
    transform=transform
)

# # Download and load test dataset
# test_dataset = torchvision.datasets.MNIST(
#     root='./data', 
#     train=False, 
#     download=True, 
#     transform=transform
# )

# # Create data loaders
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)