import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
from tensorflow.keras.callbacks import TensorBoard
import datetime
import os
import matplotlib.pyplot as plt
from data_loader import load_images_from_folder, load_and_normalize_images

def initialize_generator(latent_dim, image_dim, dropout_rate_generator):
    return Sequential([
        Dense(128, input_dim=latent_dim, activation='relu'),
        Dropout(dropout_rate_generator),
        Dense(256, activation='relu'),
        Dropout(dropout_rate_generator),
        Dense(512, activation='relu'),
        Dropout(dropout_rate_generator),
        Dense(1024, activation='relu'),
        Dense(image_dim, activation='sigmoid')    
    ])

def initialize_discriminator(image_dim, dropout_rate_discriminator):
    return Sequential([
        Dense(256, input_dim=image_dim, activation='relu'),
        Dropout(dropout_rate_discriminator),
        Dense(512, activation='relu'),
        Dropout(dropout_rate_discriminator),
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
            y_pred = discriminator_model(fake_batch, trainable=False)
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
              n_generator, n_discriminator, batch_size, latent_dim, n_epochs, image_reshape_size, log_dir="logs/gan", image_dir="GAN/data", lr_generator=0.001, lr_discriminator=0.0001):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = tf.summary.create_file_writer(log_dir)
    for epoch in tqdm(range(n_epochs)):
        indices = np.random.choice(len(real_data), batch_size)
        real_data_batch = real_data[indices]
        generator_model, g_loss = train_generator(generator_model,discriminator_model,n_generator,latent_dim,batch_size,lr_generator)
        discriminator_model, d_loss = train_discriminator(generator_model,discriminator_model,real_data_batch,n_discriminator,batch_size,latent_dim,lr_discriminator)
        with writer.as_default():
            tf.summary.scalar("Generator_loss", g_loss, step=epoch)
            tf.summary.scalar("Discriminator_loss", d_loss, step=epoch)
        if epoch % 100 == 0:
            fake_data = generator_model(np.random.normal(-1, 1, (5, latent_dim)))
            fake_data = fake_data.numpy().reshape(5, image_reshape_size[0], image_reshape_size[1], image_reshape_size[2])
            fake_data = (fake_data * 255).astype(np.uint8)
            for i in range(5):
                if not os.path.exists(image_dir):
                    os.makedirs(image_dir)
                Image.fromarray(fake_data[i]).save(f"{image_dir}/fake_data_{epoch}_{i}.png")
            if latent_dim == 2:
                visualize_latent_space(generator_model, latent_dim=latent_dim, image_reshape_size=image_reshape_size, grid_size=10, save_path=f"{image_dir}/latent_space_visualisation_{epoch}.png")
    return generator_model, discriminator_model

def get_run_name(latent_dim, batch_size, n_epochs, n_generator, n_discriminator, lr_generator, lr_discriminator, dropout_rate_generator, dropout_rate_discriminator):
    return f"FRUITS_ld{latent_dim}_bs{batch_size}_ep{n_epochs}_ng{n_generator}_nd{n_discriminator}_lrg{lr_generator}_lrd{lr_discriminator}_dpg{dropout_rate_generator}_dpd{dropout_rate_discriminator}"

def visualize_latent_space(generator_model, latent_dim, image_reshape_size, grid_size=10, figsize=(10, 10), save_path=None):
    if latent_dim != 2:
        raise ValueError("latent_dim != 2")
    x = np.random.normal(-1, 1, grid_size)
    y = np.random.normal(-1, 1, grid_size)
    grid = np.array([[xi, yi] for yi in y for xi in x])
    
    generated_images = generator_model.predict(grid)
    generated_images = generated_images.reshape(-1, image_reshape_size[0], image_reshape_size[1], image_reshape_size[2])

    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            axes[i, j].imshow(generated_images[idx])
            axes[i, j].axis("off")
            idx += 1
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)


if __name__ == "__main__":
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    input_dir = 'data/fruits_dataset'
    output_dir = 'processed_data'
    images, labels = load_images_from_folder(input_dir, max_images_per_class=5, target_size=(32, 32))
    normalized_images = load_and_normalize_images(images, flatten=True)
    real_data = np.array(normalized_images)
    # real_data = x_train.reshape(60000, 784).astype('float32') / 255.0
    latent_dim = 100
    image_dim = real_data.shape[1]
    image_reshape_size = (32, 32, 3)
    batch_size = 128
    n_epochs = 1001
    n_generator = 1
    n_discriminator = 3
    lr_generator = 0.001
    lr_discriminator = 0.0001
    dropout_rate_generator = 0.1
    dropout_rate_discriminator = 0.3
    run_name = get_run_name(latent_dim, batch_size, n_epochs, n_generator, n_discriminator, lr_generator, lr_discriminator, dropout_rate_generator, dropout_rate_discriminator)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = f"GAN/runs/{run_name}_{timestamp}"
    image_dir = f"{run_dir}/images"
    model_dir = f"{run_dir}/models"
    log_dir = f"logs/gan/{run_name}_{timestamp}"

    generator_model = initialize_generator(latent_dim, image_dim, dropout_rate_generator)
    discriminator_model = initialize_discriminator(image_dim, dropout_rate_discriminator)
    generator_model, discriminator_model = train_gan(generator_model, discriminator_model
    , real_data, n_generator, n_discriminator, batch_size, latent_dim, n_epochs, image_reshape_size, log_dir=log_dir, lr_generator=lr_generator
    , lr_discriminator=lr_discriminator, image_dir=image_dir
    )
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    generator_model.save(f"{model_dir}/generator_model.keras")
    discriminator_model.save(f"{model_dir}/discriminator_model.keras")

    # load model
    # generator_model = tf.keras.models.load_model("GAN/runs/ld2_bs128_ep10000_ng1_nd3_lrg0.001_lrd0.0001_dpg0.1_dpd0.3_20250717-224359/models/generator_model.keras")
    # latent_dim = 2
    # save_path = f"GAN/data/latent_space_visualisation.png"
    # visualize_latent_
