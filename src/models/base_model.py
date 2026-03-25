from typing import Any
from torch import nn
import torch


class VideoTransformer(nn.Module):
    def __init__(self, embedding_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._embedding_size = embedding_size

    def forward(
        video_clips: torch.Tensor,
    ):  # input: [batch, video_size, clip_emb_size]; output: [batch, video_size, our_emb_size]
        pass


class MusicTransformer(nn.Module):
    def __init__(self, embedding_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._embedding_size = embedding_size

    def forward(
        music_clips: torch.Tensor,
    ):  # input: [batch, video_size, clip_emb_size]; output: [batch, video_size, our_emb_size]
        pass


class VideoConverter:
    def __call__(
        self, video, segment_length
    ) -> (
        Any
    ):  # input: [batch, video_size]; output: [batch, video_size / segment_length, our_emb_size]
        pass


class MusicConverter:
    def __call__(
        self, music, segment_length
    ) -> (
        Any
    ):  # input: [batch, video_size]; output: [batch, video_size / segment_length, our_emb_size]
        pass
