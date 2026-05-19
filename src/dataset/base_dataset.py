from torch.utils import data
from pathlib import Path
import glob
from typing import Any
import torch
import torchvision.io as v_io
import torchaudio


class MusicVideoDataset(data.Dataset):
    def __init__(self, path_to_dataset: Path) -> None:
        super().__init__()

        self._dataset_path = Path(path_to_dataset)
        description_files = glob.glob(
            str(self._dataset_path / "**V2M-bench.txt"), recursive=True
        )
        if not description_files:
            raise FileNotFoundError("File 'V2M-bench.txt' not found")

        with open(description_files[0]) as f:
            file_ids = [line.strip() for line in f.read().split("\n") if line.strip()]

        self._samples = []
        print(f"Indexing {len(file_ids)} samples...")

        for file_id in file_ids:
            pattern = str(self._dataset_path / f"**/{file_id}*.*")
            found_paths = glob.glob(pattern, recursive=True)

            v_files = sorted([f for f in found_paths if f.endswith(".mp4")])
            a_files = sorted([f for f in found_paths if f.endswith(".wav")])

            if v_files and a_files:
                self._samples.append({"video_paths": v_files, "audio_paths": a_files})

    def _load_video(self, files_to_load):
        loaded_video = []
        for file in sorted(files_to_load):
            video_tensor, _, __ = v_io.read_video(file, pts_unit="sec")
            loaded_video.append(video_tensor)

        min_frames = min(t.shape[0] for t in loaded_video)
        formatted_tensors = [t[:min_frames, ...] for t in loaded_video]
        return torch.stack(formatted_tensors)

    def _load_audio(self, files_to_load):
        loaded_audio = []
        for file in sorted(files_to_load):
            audio_tensor, _ = torchaudio.load(file)
            loaded_audio.append(audio_tensor)

        min_frames = min(t.shape[0] for t in loaded_audio)
        formatted_tensors = [t[:min_frames, ...] for t in loaded_audio]
        return torch.stack(formatted_tensors)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index) -> Any:
        sample = self._samples[index]
        video_data = self._load_video(files_to_load=sample["video_paths"][:1])
        audio_data = self._load_audio(files_to_load=sample["audio_paths"][:1])

        return video_data, audio_data


class EncodedMusicVideoDataset(data.Dataset):
    def __init__(self, path_to_dataset: Path) -> None:
        super().__init__()

        self._dataset_path = Path(path_to_dataset)
        description_files = glob.glob(
            str(self._dataset_path / "**V2M-bench.txt"), recursive=True
        )
        if not description_files:
            raise FileNotFoundError("File 'V2M-bench.txt' not found.")

        with open(description_files[0]) as f:
            # Filtrujemy puste linie, aby uniknąć błędów
            file_ids = [line.strip() for line in f.read().split("\n") if line.strip()]

        self._samples = []
        print(f"Indexing {len(file_ids)} samples...")
        for file_id in file_ids:
            pattern = str(self._dataset_path / f"**/{file_id}*.*")
            found_paths = glob.glob(pattern, recursive=True)

            v_files = sorted([f[:-4] for f in found_paths if f.endswith(".mp4")])
            a_files = sorted([f[:-4] for f in found_paths if f.endswith(".wav")])

            if v_files and a_files:
                self._samples.append({"video_paths": v_files, "audio_paths": a_files})

    def _load_video(self, files_to_load):
        loaded_video = []
        for file in sorted(files_to_load):
            video_tensor = torch.load(file + ".pt", map_location=torch.device("cpu"))
            loaded_video.append(video_tensor)

        min_frames = min(t.shape[0] for t in loaded_video)
        formatted_tensors = [t[:min_frames, ...] for t in loaded_video]
        return torch.stack(formatted_tensors)

    def _load_audio(self, files_to_load):
        loaded_audio = []
        for file in sorted(files_to_load):
            audio_tensor = torch.load(file + ".pt", map_location=torch.device("cpu"))
            loaded_audio.append(audio_tensor)

        min_frames = min(t.shape[0] for t in loaded_audio)
        formatted_tensors = [t[:min_frames, ...] for t in loaded_audio]
        return torch.stack(formatted_tensors)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index) -> Any:
        sample = self._samples[index]
        video_data = self._load_video(files_to_load=sample["video_paths"])
        audio_data = self._load_audio(files_to_load=sample["audio_paths"])

        return video_data, audio_data
