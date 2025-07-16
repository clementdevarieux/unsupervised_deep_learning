import torch
import matplotlib.pyplot as plt

def show_reconstruction(model, sample, epoch=None):
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        sample_tensor = torch.FloatTensor(sample).to(device)
        if len(sample_tensor.shape) == 1:
            sample_tensor = sample_tensor.unsqueeze(0)

        reconstruction = model(sample_tensor).cpu().numpy().squeeze()
        original = sample_tensor.cpu().numpy().squeeze()

    original_img = original.reshape(28, 28)
    reconstructed_img = reconstruction.reshape(28, 28)

    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_img, cmap='gray')
    plt.title('Reconstructed')
    plt.axis('off')

    title = f'MNIST Reconstruction - Epoch {epoch}' if epoch else 'MNIST Reconstruction'
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

    model.train()