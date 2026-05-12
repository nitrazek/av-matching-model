from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src import dataset, loss, models
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
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
    )
    return loss.get_accuracy(dataloader, music_model, video_model, device)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Music-Video Retrieval Model")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/media/mikic202/Nowy1/uczelnia/semestr_10/data/data_split_3/val",
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
