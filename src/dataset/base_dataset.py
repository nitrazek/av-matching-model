from torch.utils import data
from pathlib import Path
import glob
from typing import Any
import torch
import torchvision.io as v_io
import torchaudio


### WERSJA MOCK, DO WYMIANY ###
class MusicVideoDataset(data.Dataset):
    def __init__(self, path_to_dataset: Path) -> None:
        super().__init__()
        path_to_dataset_description = glob.glob(
            str(path_to_dataset / "**V2M-bench.txt")
        )
        self._dataset_path = path_to_dataset
        with open(path_to_dataset_description[0]) as f:
            self._files_in_dataset = f.read().split("\n")

        self._opened_files = {}

    def _load_video(self, files_to_load):
        loaded_video = []
        for file in sorted(files_to_load):
            video_tensor, _, __ = v_io.read_video(file, pts_unit="sec")
            loaded_video.append(video_tensor)
        return torch.tensor(loaded_video)

    def _load_audio(self, files_to_load):
        loaded_audio = []
        for file in sorted(files_to_load):
            audio_tensor, _ = torchaudio.load(file)
            loaded_audio.append(audio_tensor)
        return torch.tensor(loaded_audio)

    def __getitem__(self, index) -> Any:
        file_to_open = self._files_in_dataset[index]
        if file_to_open not in self._opened_files:
            files_to_open = glob.glob(
                str(self._dataset_path / f"**/{file_to_open}*.*"), recursive=True
            )
            self._opened_files[file_to_open] = files_to_open

        return self._load_video(
            [file for file in self._opened_files[file_to_open] if file.endswith("mp4")]
        ), self._load_audio(
            [file for file in self._opened_files[file_to_open] if file.endswith("wav")]
        )
