from dataclasses import dataclass

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import base_dataset
from src import models, utils


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 10
    lr: float = 1e-4


def train_one_epoch(
    music_transformer: models.MusicTransformer,
    music_converter: models.MusicConverter,
    video_transformer: models.VideoTransformer,
    video_converter: models.VideoConverter,
    train_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device
):
    music_transformer.train()
    video_transformer.train()
    total_loss = 0.0

    for batch in train_dataloader:
        ### POBIERZ DANE Z DATALOADERA ###
        ### PRZERÓB DANE VIDEO NA FEATURE'Y PRZEZ VIDEOCONVERTER ###
        ### PRZERÓB DANE MUZYCZNE NA FEATURE'Y PRZEZ MUSICCONVERTER ###
        pass

    return total_loss / len(train_dataloader)


def train(config: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Detected device: {device}")

    embed_dim = 512
    music_transformer = models.MusicTransformer(num_layers=4, query_dim=embed_dim).to(device)
    music_converter = models.MusicConverter()
    video_transformer = models.VideoTransformer(num_layers=4, query_dim=embed_dim).to(device)
    video_converter = models.VideoConverter()
    
    train_dataset = base_dataset.MusicVideoDataset()
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    optimizer = optim.AdamW(
        list(music_transformer.parameters() + video_transformer.parameters()),
        lr=config.lr,
        weight_decay=0.01
    )

    for epoch in range(config.epochs):
        avg_loss = train_one_epoch(
            music_transformer=music_transformer,
            music_converter=music_converter,
            video_transformer=video_transformer,
            video_converter=video_converter,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            device=device
        )

    torch.save({
        "music_transformer": music_transformer.state_dict(),
        "video_transformer": video_transformer.state_dict(),
    })


if __name__ == "__main__":
    parser = utils.build_parser_from_dataclass(cls=TrainConfig)
    config = TrainConfig(**vars(parser.parse_args()))
    train(config=config)