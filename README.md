# Apprentissage Profond Non Supervisé

Ce projet explore différentes méthodes d'apprentissage profond non supervisé appliquées à deux datasets principaux : les chiffres MNIST et un dataset d'images de fruits de Kaggle.

## 📖 Description

Le projet implémente et compare plusieurs algorithmes d'apprentissage non supervisé :
- **Autoencodeurs** (classiques et convolutionnels)
- **Autoencodeurs Variationnels (VAE)**
- **Réseaux Antagonistes Génératifs (GAN)**
- **Cartes de Kohonen (SOM)**
- **K-means**
- **Analyse en Composantes Principales (PCA)**

## 🗂️ Datasets

### MNIST
- Dataset classique de chiffres manuscrits (0-9)
- Images en niveaux de gris 28x28 pixels
- Utilisé pour tester et valider les différents algorithmes

### Dataset de Fruits
- Source : [Kaggle Fruit Recognition Dataset](https://www.kaggle.com/datasets/chrisfilo/fruit-recognition)
- Images couleur de différents fruits (pommes, bananes, etc.)
- Images redimensionnées en 32x32 pixels RGB
- Utilisé pour tester les algorithmes sur des données plus complexes

## 🏗️ Structure du Projet

```
├── autoencoder/          # Implémentations des autoencodeurs et VAE
├── GAN/                  # Réseaux antagonistes génératifs
├── kohonen_maps/         # Cartes auto-organisatrices de Kohonen
├── kmeans/              # Algorithme K-means
├── pca/                 # Analyse en composantes principales
├── data/                # Données et datasets
├── logs/                # Logs d'entraînement (TensorBoard)
└── processed_data/      # Données préprocessées
```

## 🔧 Technologies Utilisées

- **PyTorch** - Framework d'apprentissage profond
- **NumPy** - Calculs numériques
- **Matplotlib** - Visualisations
- **Scikit-learn** - Outils d'apprentissage automatique
- **TensorBoard** - Suivi des métriques d'entraînement
- **OpenCV** - Traitement d'images

## 🚀 Installation

1. Cloner le repository
```bash
git clone https://github.com/clementdevarieux/unsupervised_deep_learning.git
cd unsupervised_deep_learning
```

2. Installer les dépendances
```bash
pip install -r requirements.txt
```

## 📊 Fonctionnalités Principales

### Autoencodeurs
- Autoencodeurs classiques et convolutionnels
- Compression et reconstruction d'images
- Visualisation de l'espace latent 2D
- Génération de nouvelles images

### VAE (Variational Autoencoders)
- Autoencodeurs variationnels avec régularisation KL
- Génération probabiliste d'images
- Interpolation dans l'espace latent
- Exploration systématique de l'espace latent

### GAN (Generative Adversarial Networks)
- Implémentation vanilla et Wasserstein GAN
- Génération d'images réalistes
- Visualisation de l'espace latent

### Cartes de Kohonen
- Cartes auto-organisatrices pour la visualisation
- Compression de données
- Analyse de la structure des données

### Clustering
- K-means pour le regroupement non supervisé
- PCA pour la réduction de dimensionnalité

## 📈 Visualisations

Le projet inclut de nombreuses visualisations :
- Reconstructions d'images originales vs. générées
- Espaces latents 2D avec classes colorées
- Grilles d'exploration de l'espace latent
- Interpolations entre différentes classes
- Métriques d'entraînement en temps réel

## 🎯 Objectifs Pédagogiques

Ce projet permet d'apprendre et de comparer :
- Les différences entre les approches génératives
- L'impact de l'architecture sur la qualité des reconstructions
- Les techniques de régularisation (KL divergence pour VAE)
- L'optimisation adversariale (pour les GAN)
- La visualisation et l'interprétation des espaces latents

## 📝 Notes

- Les modèles sont configurables (fonctions d'activation, dimensions latentes, etc.)
- Support CUDA pour l'entraînement GPU
- Sauvegarde automatique des modèles entraînés
- Logs TensorBoard pour le suivi des métriques

## 🔄 Usage

Chaque dossier contient ses propres scripts d'entraînement et de visualisation. Par exemple :

```bash
# Pour les autoencodeurs
cd autoencoder
python main.py

# Pour les GAN
cd GAN
python algorithme_vanilla.py
```

Les paramètres peuvent être ajustés directement dans les fichiers de configuration ou les scripts principaux.
