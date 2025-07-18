import torch
import numpy as np


def encode_dataset(model, data, batch_size=1000, use_mean=True):
    device = next(model.parameters()).device
    model.eval()

    latents = []

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_tensor = torch.FloatTensor(batch).to(device)

            if use_mean:
                mu, logvar = model.encode_to_params(batch_tensor)
                encoded = mu
            else:
                encoded = model.encode(batch_tensor)

            latents.append(encoded.cpu().numpy())

    model.train()
    return np.vstack(latents)


def generate_samples(model, latents, n_samples=100):
    device = next(model.parameters()).device
    model.eval()

    latent_mean = np.mean(latents, axis=0)
    latent_std = np.std(latents, axis=0)

    generated_samples = []

    with torch.no_grad():
        for _ in range(n_samples):
            random_latent = np.random.normal(latent_mean, latent_std, size=(1, latents.shape[1]))
            latent_tensor = torch.FloatTensor(random_latent).to(device)

            decoded = model.decode(latent_tensor)
            generated_samples.append(decoded.cpu().numpy())

    model.train()
    return np.vstack(generated_samples)