import random
import subprocess
from dataclasses import dataclass

import numpy as np

import train
from src import utils


@dataclass
class RunExperimentConfig:
    experiment_name: str


def run_experiment(config: RunExperimentConfig):
    batch_sizes = [10, 20, 30, 40]
    epochs = [20, 30, 40, 50, 60, 70, 80]
    lr_range = [5e-5, 5e-4]
    lr_decay_range = [0.9, 1]
    music_transformer_sizes = [3, 4, 5]
    video_transformer_sizes = [3, 4, 5]

    number_of_tests_per_size_combination = 5

    for music_transformer_size in music_transformer_sizes:
        for video_transformer_size in video_transformer_sizes:
            for _ in range(number_of_tests_per_size_combination):
                train.train(config=train.TrainConfig(
                    experiment_name=config.experiment_name,
                    batch_size=random.choice(batch_sizes),
                    epochs=random.choice(epochs),
                    lr=np.random.uniform(*lr_range),
                    lr_decay=np.random.uniform(*lr_decay_range),
                    music_transformer_size=music_transformer_size,
                    video_transformer_size=video_transformer_size
                ))


if __name__ == "__main__":
    parser = utils.build_parser_from_dataclass(cls=RunExperimentConfig)
    config = RunExperimentConfig(**vars(parser.parse_args()))
    run_experiment(config=config)