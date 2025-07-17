import numpy as np
import torch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
from tensorflow.keras.callbacks import TensorBoard
import datetime
import os

def initialize_generator(latent_dim, image_dim):
    return Sequential([
        Dense(100, input_dim=latent_dim, activation='relu'),
        Dense(256, activation='relu'),
        Dense(image_dim, activation='sigmoid')    
    ])

def initialize_discriminator(image_dim):
    return Sequential([
        Dense(32, input_dim=image_dim, activation='relu'),
        Dense(1, activation='sigmoid')    
    ])

def train_generator(generator_model,discriminator_model,n,latent_dim,batch_size, lr):
    generator_optimizer = Adam(learning_rate=lr)
    total_loss = 0
    for i in range(n):
        random_noise = np.random.normal(-1, 1, (batch_size, latent_dim))
        fake_labels = np.ones((batch_size, 1))
        with tf.GradientTape() as tape:
            fake_batch = generator_model(random_noise)
            y_pred = discriminator_model(fake_batch, trainable=False) # on ne veut pas que le discriminateur soit trainable
            loss = tf.keras.losses.binary_crossentropy(fake_labels, y_pred)
        gradients = tape.gradient(loss, generator_model.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients, generator_model.trainable_variables))
        total_loss += tf.reduce_mean(loss).numpy()
    return generator_model, total_loss / n

def prepare_discriminator_data(real_data, fake_data):
    y_real = np.ones((len(real_data), 1))
    y_fake = np.zeros((len(fake_data), 1))
    y = np.vstack((y_real, y_fake))
    x = np.concatenate([real_data, fake_data])
    return x, y

def train_discriminator(generator_model, discriminator_model, real_data, n, batch_size, latent_dim, lr):
    discriminator_optimizer = Adam(learning_rate=lr)
    total_loss = 0
    for _ in range(n):
        fake_data = generator_model(np.random.normal(-1, 1, (batch_size, latent_dim)))
        x_train, y_train = prepare_discriminator_data(real_data, fake_data)
        with tf.GradientTape() as tape:
            y_pred = discriminator_model(x_train)
            loss = tf.keras.losses.binary_crossentropy(y_train, y_pred)
        gradients = tape.gradient(loss, discriminator_model.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients, discriminator_model.trainable_variables))
        total_loss += tf.reduce_mean(loss).numpy()
    return discriminator_model, total_loss / n


def train_gan(generator_model, discriminator_model, real_data,
              n_generator, n_discriminator, batch_size, latent_dim, n_epochs, log_dir="logs/gan", lr_generator=0.001, lr_discriminator=0.0001):
    writer = tf.summary.create_file_writer(log_dir)
    for epoch in tqdm(range(n_epochs)):
        indices = np.random.choice(len(real_data), batch_size)
        real_data_batch = real_data[indices]
        generator_model, g_loss = train_generator(generator_model,discriminator_model,n_generator,latent_dim,batch_size,lr_generator)
        discriminator_model, d_loss = train_discriminator(generator_model,discriminator_model,real_data_batch,n_discriminator,batch_size,latent_dim,lr_discriminator)
        with writer.as_default():
            tf.summary.scalar("Generator_loss", g_loss, step=epoch)
            tf.summary.scalar("Discriminator_loss", d_loss, step=epoch)
        if epoch % 30 == 0:
            fake_data = generator_model(np.random.normal(-1, 1, (5, latent_dim)))
            fake_data = fake_data.numpy().reshape(5, 28, 28)
            fake_data = (fake_data * 255).astype(np.uint8)
            for i in range(5):
                if not os.path.exists(f"GAN/data/run_{n_epochs}_{batch_size}_{lr_generator}_{lr_discriminator}"):
                    os.makedirs(f"GAN/data/run_{n_epochs}_{batch_size}_{lr_generator}_{lr_discriminator}")
                Image.fromarray(fake_data[i]).save(f"GAN/data/run_{n_epochs}_{batch_size}_{lr_generator}_{lr_discriminator}/fake_data_{epoch}_{i}.png")
    return generator_model, discriminator_model


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    real_data = x_train.reshape(60000, 784).astype('float32') / 255.0
    latent_dim = 100
    image_dim = 784
    batch_size = 16
    n_epochs = 10000
    n_generator = 1
    n_discriminator = 3
    lr_generator = 0.001
    lr_discriminator = 0.0001
    log_dir = "logs/gan/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    generator_model = initialize_generator(latent_dim, image_dim)
    discriminator_model = initialize_discriminator(image_dim)
    generator_model, discriminator_model = train_gan(generator_model, discriminator_model, real_data, n_generator, n_discriminator, batch_size, latent_dim, n_epochs, log_dir=log_dir, lr_generator=lr_generator, lr_discriminator=lr_discriminator)
    if not os.path.exists(f"GAN/data/save_models/run_{n_epochs}_{batch_size}_{lr_generator}_{lr_discriminator}"):
        os.makedirs(f"GAN/data/save_models/run_{n_epochs}_{batch_size}_{lr_generator}_{lr_discriminator}")
    generator_model.save(f"GAN/data/save_models/run_{n_epochs}_{batch_size}_{lr_generator}_{lr_discriminator}/generator_model.keras")
    discriminator_model.save(f"GAN/data/save_models/run_{n_epochs}_{batch_size}_{lr_generator}_{lr_discriminator}/discriminator_model.keras")



