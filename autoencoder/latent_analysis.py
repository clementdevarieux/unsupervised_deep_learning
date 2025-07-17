import torch
import numpy as np
from sklearn.mixture import GaussianMixture


def encode_dataset(model, dataset, batch_size=1000):
    device = next(model.parameters()).device
    model.eval()

    latent_representations = []

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch_end = min(i + batch_size, len(dataset))
            batch_data = torch.FloatTensor(dataset[i:batch_end]).to(device)

            encoded = model.encoder(batch_data)
            latent_representations.append(encoded.cpu().numpy())

    latent_representations = np.concatenate(latent_representations, axis=0)

    return latent_representations


def generate_samples(model, latent_representations, n_samples=100):
    device = next(model.parameters()).device
    model.eval()

    mean = np.mean(latent_representations, axis=0)
    cov = np.cov(latent_representations.T)

    sampled_latents = np.random.multivariate_normal(mean, cov, n_samples)

    with torch.no_grad():
        sampled_latents_tensor = torch.FloatTensor(sampled_latents).to(device)
        generated_samples = model.decoder(sampled_latents_tensor)

    return generated_samples.cpu().numpy()