import torch
import numpy as np
from visualisation import show_reconstruction  # Import the simple visualization function


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
                      visualize_every=1, image_shape=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    model.train()

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

        if visualize_every and (epoch + 1) % visualize_every == 0:
            random_idx = np.random.randint(0, len(train_data))
            viz_sample = train_data[random_idx]
            show_reconstruction(model, viz_sample, epoch + 1)

def validate_autoencoder(model, val_data, criterion, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    batches = create_batches(val_data, batch_size, shuffle=False)
    total_loss = 0
    num_batches = len(batches)

    with torch.no_grad():
        for batch_features in batches:
            batch_features = torch.FloatTensor(batch_features).to(device)
            outputs = model(batch_features)
            loss = criterion(outputs, batch_features)
            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Validation Loss: {avg_loss:.6f}")
    return avg_loss