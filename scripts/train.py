import datetime
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import dataset, loss, models
from src import utils


@dataclass
class TrainConfig:
    batch_size: int = 1
    epochs: int = 1
    lr: float = 1e-4
    segment_length: float = 5


def train_one_epoch(
    music_transformer: models.MusicTransformer,
    video_transformer: models.VideoTransformer,
    train_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch_idx: int = 0,
):
    music_transformer.train()
    video_transformer.train()
    total_loss = 0.0

    pbar = tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
        desc=f"Epoch {epoch_idx}",
    )

    for batch_idx, (video_features, music_features) in pbar:
        video_features = video_features.to(device)
        music_features = music_features.to(device)
        optimizer.zero_grad()

        music_emb = music_transformer(music_features).mean(dim=1)
        video_emb = video_transformer(video_features).mean(dim=1)

        model_loss = loss.infonce_loss(music_emb=music_emb, video_emb=video_emb)

        model_loss.backward()
        optimizer.step()

        current_loss = model_loss.item()
        total_loss += current_loss

        pbar.set_postfix(
            {
                "loss": f"{current_loss:.4f}",
                "avg_loss": f"{total_loss/(batch_idx+1):.4f}",
            }
        )

    avg_epoch_loss = total_loss / len(train_dataloader)
    print(f"\n>>> Epoch {epoch_idx} Finished. Average Loss: {avg_epoch_loss:.4f}\n")

    return avg_epoch_loss


def train(config: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Detected device: {device}")

    embed_dim = 512
    music_transformer = models.MusicTransformer(num_layers=4, query_dim=embed_dim).to(
        device
    )
    video_transformer = models.VideoTransformer(num_layers=4, query_dim=embed_dim).to(
        device
    )
    train_dataset = dataset.EncodedMusicVideoDataset(
        path_to_dataset=Path("data", "splits", "train")
    )

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=config.batch_size, shuffle=True
    )

    optimizer = optim.AdamW(
        list(music_transformer.parameters()) + list(video_transformer.parameters()),
        lr=config.lr,
        weight_decay=0.01,
    )

    for epoch in range(config.epochs):
        avg_loss = train_one_epoch(
            music_transformer=music_transformer,
            video_transformer=video_transformer,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
        )
        print(f"Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.4f}")

    current_timestamp_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    torch.save(
        music_transformer.state_dict(),
        Path("outputs", current_timestamp_str, "music_transformer.pth"),
    )
    torch.save(
        video_transformer.state_dict(),
        Path("outputs", current_timestamp_str, "video_transformer.pth"),
    )


if __name__ == "__main__":
    parser = utils.build_parser_from_dataclass(cls=TrainConfig)
    config = TrainConfig(**vars(parser.parse_args()))
    train(config=config)
