from algorithm import konoha_map
from compression import compression_decompression_pipeline
from generation import generate_new_samples
from visualisation import plot_latent_space, analyze_class_distribution_on_map
from utils import load_and_standardize_data, save_results
from data_loader import load_images_from_folder
import numpy as np


def main_mnist():
    config = {
        "num_samples": 10000,
        "map_lines": 15,
        "map_columns": 15,
        "learning_rate": 0.5,
        "gamma": 1.0,
        "num_iterations": 15,
        "batch_size": 200,
        "test_samples": 10,
        "generated_samples": 10,
        "save_every": 3,
        "image_shape": (28, 28),  
    }

    data, labels = load_and_standardize_data(num_samples=config["num_samples"])

    W = konoha_map(
        map_lines=config["map_lines"],
        map_columns=config["map_columns"],
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        dataset=data,
        num_iterations=config["num_iterations"],
        batch_size=config["batch_size"],
        save_every=config["save_every"],
        image_shape=config["image_shape"],  
    )

    plot_latent_space(W, config["map_lines"], config["map_columns"], shape=config["image_shape"])

    analyze_class_distribution_on_map(
        data=data,
        labels=labels,
        weights=W,
        map_lines=config["map_lines"],
        map_columns=config["map_columns"]
    )

    compression_decompression_pipeline(
        data=data,
        weights=W,
        num_samples=config["test_samples"],
        map_lines=config["map_lines"],
        map_columns=config["map_columns"],
    )

    generate_new_samples(
        weights=W,
        map_lines=config["map_lines"],
        map_columns=config["map_columns"],
        num_samples=config["generated_samples"],
    )

    save_results(W, config["map_lines"], config["map_columns"])

def main_fruits():
    config = {
        "num_samples": 10000,
        "map_lines": 15,
        "map_columns": 15,
        "learning_rate": 0.5,
        "gamma": 1.0,
        "num_iterations": 100,
        "batch_size": 200,
        "test_samples": 10,
        "generated_samples": 10,
        "save_every": 10,
        "image_shape": (32, 32, 3),
        "max_images_per_class": 750,
    }

    # data, labels = load_and_standardize_data(num_samples=config["num_samples"])
    input_dir = 'data/fruits_dataset'

    images, labels = load_images_from_folder(input_dir, max_images_per_class=config["max_images_per_class"], target_size=(32, 32))
    images = np.array(images)
    labels = np.array(labels)
    X = np.array(images)
    X = X.reshape(X.shape[0], -1) 

    # BOTH
    original_mean = np.mean(X)
    original_std = np.std(X)

    X = (X - original_mean) / original_std

    W = konoha_map(
        map_lines=config["map_lines"],
        map_columns=config["map_columns"],
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        dataset=X,
        num_iterations=config["num_iterations"],
        batch_size=config["batch_size"],
        save_every=config["save_every"],
        image_shape=config["image_shape"],
    )

    plot_latent_space(W, config["map_lines"], config["map_columns"], shape=config["image_shape"])

    analyze_class_distribution_on_map(
        data=X,
        labels=labels,
        weights=W,
        map_lines=config["map_lines"],
        map_columns=config["map_columns"]
    )

    compression_decompression_pipeline(
        data=X,
        weights=W,
        num_samples=config["test_samples"],
        map_lines=config["map_lines"],
        map_columns=config["map_columns"],
        image_shape=config["image_shape"],
    )

    generate_new_samples(
        weights=W,
        map_lines=config["map_lines"],
        map_columns=config["map_columns"],
        num_samples=config["generated_samples"],
        image_shape=config["image_shape"],
    )

    save_results(W, config["map_lines"], config["map_columns"])

if __name__ == "__main__":
    # main_mnist()
    main_fruits()
