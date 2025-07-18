import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from vae_visualization import show_reconstruction


def train_variational_autoencoder(model, train_data, optimizer, criterion, epochs, batch_size=64,
                                  visualize_every=1, image_shape=None, model_save_path="google_model.pth",
                                  log_dir="./runs/autoencoder", coef_KL=.01, beta_schedule=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.train()

    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        batches = create_batches(train_data, batch_size, shuffle=True)

        if beta_schedule:
            beta = min(coef_KL, coef_KL * (epoch / 10.0))
        else:
            beta = coef_KL

        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        num_batches = len(batches)

        for batch_idx, batch_features in enumerate(batches):
            try:
                batch_features = torch.FloatTensor(batch_features).to(device)

                optimizer.zero_grad()

                outputs, mu, logvar = model(batch_features)
                recon_loss = criterion(outputs, batch_features)
                kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

                total_loss = recon_loss + beta * kl_loss  # Use beta instead of coef_KL

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_loss += total_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()

            except Exception as e:
                print(f"Error at epoch {epoch}, batch {batch_idx}: {str(e)}")
                raise e

        avg_epoch_loss = epoch_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_kl_loss = epoch_kl_loss / num_batches

        print(f"Epoch {epoch + 1}: Beta: {beta:.4f}, "
              f"Total Loss: {avg_epoch_loss:.6f}, "
              f"Recon Loss: {avg_recon_loss:.6f}, "
              f"KL Loss: {avg_kl_loss:.6f}")

        writer.add_scalar('Loss/Total', avg_epoch_loss, epoch + 1)
        writer.add_scalar('Loss/Reconstruction', avg_recon_loss, epoch + 1)
        writer.add_scalar('Loss/KL', avg_kl_loss, epoch + 1)
        writer.add_scalar('Training/Beta', beta, epoch + 1)

        if visualize_every and (epoch + 1) % visualize_every == 0:
            random_idx = np.random.randint(0, len(train_data))
            viz_sample = train_data[random_idx]
            show_reconstruction(model, viz_sample, epoch + 1)

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    writer.close()


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