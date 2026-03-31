from dataclasses import dataclass

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src import dataset, loss, models, utils


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 10
    lr: float = 1e-4
    segment_length: float = 5


def train_one_epoch(
    music_transformer: models.MusicTransformer,
    music_converter: models.MusicConverter,
    video_transformer: models.VideoTransformer,
    video_converter: models.VideoConverter,
    train_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    segment_length: float
):
    music_transformer.train()
    video_transformer.train()
    total_loss = 0.0

    for batch in train_dataloader:
        music = batch['music'].to(device)
        music_features = music_converter(music=music, segment_length=segment_length)

        videos = batch['video'].to(device)
        video_features = video_converter(videos=videos, segment_length=segment_length)

        optimizer.zero_grad()

        music_emb = music_transformer(music_features).mean(dim=1)
        video_emb = video_transformer(video_features).mean(dim=1)

        model_loss = loss.infonce_loss(music_emb=music_emb, video_emb=video_emb)
        
        model_loss.backward()
        optimizer.step()

        total_loss += model_loss.item()

    return total_loss / len(train_dataloader)


def train(config: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Detected device: {device}")

    embed_dim = 512
    music_transformer = models.MusicTransformer(num_layers=4, query_dim=embed_dim).to(device)
    music_converter = models.MusicConverter()
    video_transformer = models.VideoTransformer(num_layers=4, query_dim=embed_dim).to(device)
    video_converter = models.VideoConverter()
    
    train_dataset = dataset.MusicVideoDataset()
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
            device=device,
            segment_length=config.segment_length
        )
        print(f"Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.4f}")

    torch.save({
        "music_transformer": music_transformer.state_dict(),
        "video_transformer": video_transformer.state_dict(),
    })


if __name__ == "__main__":
    parser = utils.build_parser_from_dataclass(cls=TrainConfig)
    config = TrainConfig(**vars(parser.parse_args()))
    train(config=config)
