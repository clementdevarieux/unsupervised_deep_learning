from algorithm import konoha_map
from compression import compression_decompression_pipeline
from generation import generate_new_samples
from visualisation import plot_latent_space
from utils import load_and_standardize_data, save_results


def main():
    config = {
        "num_samples": 10000,
        "map_lines": 15,
        "map_columns": 15,
        "learning_rate": 0.5,
        "gamma": 1.0,
        "num_iterations": 50,
        "batch_size": 200,
        "test_samples": 10,
        "generated_samples": 10,
        "save_every": 5,
    }

    data = load_and_standardize_data(num_samples=config["num_samples"])

    W = konoha_map(
        map_lines=config["map_lines"],
        map_columns=config["map_columns"],
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        dataset=data,
        num_iterations=config["num_iterations"],
        batch_size=config["batch_size"],
        save_every=config["save_every"],
    )

    plot_latent_space(W, config["map_lines"], config["map_columns"])

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


if __name__ == "__main__":
    main()
