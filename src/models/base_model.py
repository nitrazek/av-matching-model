from typing import Any, Optional
from torch import nn
import torch
import math
import clip
from PIL import Image
import musicsections


class VideoTransformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        query_dim: int,
        key_dim: Optional[int] = None,
        value_dim: Optional[int] = None,
        embed_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_causal_mask: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    query_dim,
                    key_dim,
                    value_dim,
                    embed_dim,
                    num_heads,
                    dropout,
                    use_causal_mask,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        video_clips: torch.Tensor,
    ):  # input: [batch, video_size, clip_emb_size]; output: [batch, video_size, our_emb_size]
        dec_output = video_clips
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output)

        return dec_output


class MusicTransformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        query_dim: int,
        key_dim: Optional[int] = None,
        value_dim: Optional[int] = None,
        embed_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_causal_mask: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    query_dim,
                    key_dim,
                    value_dim,
                    embed_dim,
                    num_heads,
                    dropout,
                    use_causal_mask,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        music_clips: torch.Tensor,
    ):  # input: [batch, video_size, clip_emb_size]; output: [batch, video_size, our_emb_size]
        dec_output = music_clips
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output)

        return dec_output


class VideoConverter:
    def __init__(self, clip_model_type: str = "ViT-B/32") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(clip_model_type, device=self.device)

    def __call__(
        self,
        videos: list[list[Image.Image]],
        segment_length: float,
        framerate: int = 20,
    ) -> (
        Any
    ):  # input: [batch, video_size]; output: [batch, video_size / segment_length, our_emb_size]
        encoded_videos = []
        for video in videos:
            video_encodding = []
            for frame in video:
                image = preprocess(frame).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    image_features = model.encode_image(image)
                video_encodding.append(image_features)
            encoded_videos.append(video_encodding)
        return encoded_videos


import musicsections

# Load models
model_deepsim = musicsections.load_deepsim_model(deepsim_model_folder)
# Segment the audio
segmentations, features = musicsections.segment_file(
    audiofile, deepsim_model=model_deepsim
)


class MusicConverter:
    def __init__(self, deepsim_model_folder) -> None:
        self.model_deepsim = musicsections.load_deepsim_model(deepsim_model_folder)

    def __call__(
        self, music, segment_length
    ) -> (
        Any
    ):  # input: [batch, video_size]; output: [batch, video_size / segment_length, our_emb_size]
        segmentations, features = musicsections.segment_file(
            music, deepsim_model=self.model_deepsim
        )


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        key_dim: Optional[int] = None,
        value_dim: Optional[int] = None,
        embed_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_causal_mask: bool = False,
    ):
        super().__init__()

        self.query_dim = query_dim
        self.key_dim = key_dim if key_dim is not None else query_dim
        self.value_dim = value_dim if value_dim is not None else self.key_dim
        self.embed_dim = embed_dim if embed_dim is not None else query_dim

        assert (
            self.embed_dim % num_heads == 0
        ), "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = self.embed_dim // num_heads
        self.use_causal_mask = use_causal_mask

        self.W_q = nn.Linear(self.query_dim, self.embed_dim)
        self.W_k = nn.Linear(self.key_dim, self.embed_dim)
        self.W_v = nn.Linear(self.value_dim, self.embed_dim)
        self.W_o = nn.Linear(self.embed_dim, self.embed_dim)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(
            1, 2
        )

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return (
            x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        )

    def forward(self, Q, K=None, V=None, mask=None):
        batch_size = Q.size(0)
        query_len = Q.size(1)

        if K is None:
            K = Q
        if V is None:
            V = K

        key_len = K.size(1)
        value_len = V.size(1)

        assert key_len == value_len, "Key and value must have same sequence length"

        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class DecoderLayer(nn.Module):
    def __init__(
        self,
        query_dim: int,
        key_dim: Optional[int] = None,
        value_dim: Optional[int] = None,
        embed_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_causal_mask: bool = False,
    ):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(
            query_dim,
            key_dim,
            value_dim,
            embed_dim,
            num_heads,
            dropout,
            use_causal_mask,
        )
        embed_dim = embed_dim if embed_dim is not None else query_dim
        self.cross_attn = MultiHeadAttention(
            query_dim,
            key_dim,
            value_dim,
            embed_dim,
            num_heads,
            dropout,
            use_causal_mask,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x)
        x = self.norm2(x + self.dropout(attn_output))
        return x
