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
import argparse


def collate_fn(batch):
    video_features = [item[0] for item in batch]
    music_features = [item[1] for item in batch]

    video_padded = pad_sequence(video_features, batch_first=True, padding_value=0.0)
    music_padded = pad_sequence(music_features, batch_first=True, padding_value=0.0)

    return video_padded, music_padded


@torch.no_grad()
def evaluate_retrieval(dataset_path: Path, model_version: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("outputs", model_version)
    embed_dim = 512
    music_model = models.MusicTransformer(num_layers=4, query_dim=embed_dim).to(device)
    music_model.load_state_dict(torch.load(output_dir / "music_transformer.pth"))
    video_model = models.VideoTransformer(num_layers=4, query_dim=embed_dim).to(device)
    video_model.load_state_dict(torch.load(output_dir / "video_transformer.pth"))
    train_dataset = dataset.EncodedMusicVideoDataset(path_to_dataset=dataset_path)
    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=10,
        shuffle=True,
        collate_fn=collate_fn,
    )
    music_model.eval()
    video_model.eval()

    all_music = []
    all_video = []

    for video_feat, music_feat in dataloader:
        video_feat, music_feat = video_feat.to(device), music_feat.to(device)

        v_mask = (video_feat.abs().sum(dim=-1) != 0).float().unsqueeze(-1)
        m_mask = (music_feat.abs().sum(dim=-1) != 0).float().unsqueeze(-1)

        v_out = video_model(video_feat)
        m_out = music_model(music_feat)

        v_emb = (v_out * v_mask).sum(dim=1) / v_mask.sum(dim=1).clamp(min=1)
        m_emb = (m_out * m_mask).sum(dim=1) / m_mask.sum(dim=1).clamp(min=1)

        all_video.append(torch.nn.functional.normalize(v_emb, dim=-1))
        all_music.append(torch.nn.functional.normalize(m_emb, dim=-1))

    all_video = torch.cat(all_video, dim=0)
    all_music = torch.cat(all_music, dim=0)

    sim_matrix = torch.matmul(all_video, all_music.T)

    num_samples = all_video.size(0)
    labels = torch.arange(num_samples).to(device)

    _, sorted_indices = torch.sort(sim_matrix, descending=True, dim=1)

    r1 = (sorted_indices[:, 0] == labels).float().mean().item()
    r5 = (sorted_indices[:, :5] == labels.unsqueeze(1)).any(dim=1).float().mean().item()

    return {"R@1": r1, "R@5": r5}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Music-Video Retrieval Model")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/media/mikic202/Nowy/uczelnia/semestr_10/data/data_split/train",
        help="Path to the directory containing encoded features",
    )
    parser.add_argument(
        "run_id",
        type=str,
        help="Timestamp of the training run (e.g., 20260509203227)",
    )

    args = parser.parse_args()
    results = evaluate_retrieval(Path(args.data_path), args.run_id)

    print(f"\nResults: {results}")


if __name__ == "__main__":
    main()
