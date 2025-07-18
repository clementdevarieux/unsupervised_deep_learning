import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from fruits_visualization import show_reconstruction


def create_batches(data, batch_size, shuffle=True):
    n_samples = len(data)
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    batches = []
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        batches.append(data[batch_indices])

    return batches


def train_autoencoder(model, train_data, optimizer, criterion, epochs, batch_size=64,
                      visualize_every=1, image_shape=(3, 32, 32), model_save_path="fruit_ae_model.pth",
                      log_dir="./runs/fruit_autoencoder"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.train()

    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        batches = create_batches(train_data, batch_size, shuffle=True)

        epoch_loss = 0
        num_batches = len(batches)

        for batch_features in batches:
            batch_features = torch.FloatTensor(batch_features).to(device)

            optimizer.zero_grad()

            outputs = model(batch_features)
            loss = criterion(outputs, batch_features)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.6f}")

        writer.add_scalar('Loss', avg_epoch_loss, epoch + 1)

        if visualize_every and (epoch + 1) % visualize_every == 0:
            random_idx = np.random.randint(0, len(train_data))
            viz_sample = train_data[random_idx]
            show_reconstruction(model, viz_sample, epoch + 1, image_shape=image_shape)

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    writer.close()