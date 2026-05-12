import datetime
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import dataset, loss, models
from src import utils
from torch.nn.utils.rnn import pad_sequence
import numpy as np


@dataclass
class TrainConfig:
    batch_size: int = 20
    epochs: int = 40
    lr: float = 3e-4
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

        video_mask = (video_features.abs().sum(dim=-1) != 0).float().unsqueeze(-1)
        music_mask = (music_features.abs().sum(dim=-1) != 0).float().unsqueeze(-1)

        music_out = music_transformer(music_features)
        video_out = video_transformer(video_features)

        music_emb = (music_out * music_mask).sum(dim=1) / music_mask.sum(dim=1).clamp(
            min=1
        )
        video_emb = (video_out * video_mask).sum(dim=1) / video_mask.sum(dim=1).clamp(
            min=1
        )

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


def collate_fn(batch):
    video_features = [item[0] for item in batch]
    music_features = [item[1] for item in batch]

    video_padded = pad_sequence(video_features, batch_first=True, padding_value=0.0)
    music_padded = pad_sequence(music_features, batch_first=True, padding_value=0.0)

    return video_padded, music_padded


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
    val_dataset = dataset.EncodedMusicVideoDataset(
        path_to_dataset=Path("data", "splits", "val")
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=10,
        shuffle=False,
        collate_fn=collate_fn,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = optim.AdamW(
        list(music_transformer.parameters()) + list(video_transformer.parameters()),
        lr=config.lr,
        weight_decay=0.01,
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    losses_over_epochs = []

    for epoch in range(config.epochs):
        avg_loss = train_one_epoch(
            music_transformer=music_transformer,
            video_transformer=video_transformer,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            epoch_idx=epoch,
        )
        losses_over_epochs.append(avg_loss)
        scheduler.step()
        print(
            f"Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.4f}, Accuracy on Validation: {loss.get_accuracy(val_dataloader, music_transformer, video_transformer, device)}"
        )

    current_timestamp_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = Path("outputs", current_timestamp_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        music_transformer.state_dict(),
        output_dir / "music_transformer.pth",
    )
    torch.save(
        video_transformer.state_dict(),
        output_dir / "video_transformer.pth",
    )
    print(losses_over_epochs)
    np.save(output_dir / "loss.npy", np.array(losses_over_epochs))


if __name__ == "__main__":
    parser = utils.build_parser_from_dataclass(cls=TrainConfig)
    config = TrainConfig(**vars(parser.parse_args()))
    train(config=config)
