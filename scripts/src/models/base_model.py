from typing import Any, Optional

import torch
import torch.nn.functional as F
import math
import clip
from PIL import Image
from torch import nn
from transformers import ClapModel, AutoProcessor


class EmbeddingWithProjection(nn.Module):
    def __init__(self, embedding_dimensions) -> None:
        super(EmbeddingWithProjection, self).__init__()
        self._embedding_dimensions = embedding_dimensions
        self.layernorm = nn.LayerNorm(embedding_dimensions)

    @staticmethod
    def create_positional_encoding(seq_count, embedding_dimensions, batch_size, device):
        position = torch.arange(seq_count, device=device).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, embedding_dimensions, 2, device=device).float()
            * (-math.log(10000.0) / embedding_dimensions)
        )
        pos_embedding = torch.zeros(seq_count, embedding_dimensions, device=device)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        pos_embedding = pos_embedding.unsqueeze(0).expand(batch_size, -1, -1)

        return pos_embedding

    def forward(self, x):
        batch_size, seq_count, _ = x.size()
        positional_encoding = self.create_positional_encoding(
            seq_count, self._embedding_dimensions, batch_size, device=x.device
        )
        return self.layernorm(x + positional_encoding)


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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.embedding_with_projection = EmbeddingWithProjection(query_dim)
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
        dec_output = self.embedding_with_projection(video_clips)
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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.embedding_with_projection = EmbeddingWithProjection(query_dim)
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
        dec_output = self.embedding_with_projection(music_clips)
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output)

        return dec_output


class VideoConverter:
    def __init__(self, clip_model_type: str = "ViT-B/32") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(clip_model_type, device=self.device)

    def __call__(
        self,
        video_segments: torch.Tensor,
        minibatch_size: int = 16,
    ) -> torch.Tensor:
        b, s, f, h, w, c = video_segments.shape
        flat_frames = video_segments.view(-1, h, w, c).permute(0, 3, 1, 2).float()
        all_embeddings = []

        for i in range(0, flat_frames.size(0), minibatch_size):
            batch = flat_frames[i : i + minibatch_size].to(self.device)
            batch_resized = F.interpolate(batch, size=(224, 224), mode='bicubic', align_corners=False)
            
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(self.device).view(1, 3, 1, 1)
            batch_normalized = (batch_resized / 255.0 - mean) / std
            
            with torch.no_grad():
                features = self.model.encode_image(batch_normalized)
                all_embeddings.append(features.cpu())

        frame_embeddings = torch.cat(all_embeddings, dim=0).view(b, s, f, -1)
        video_embeddings = frame_embeddings.mean(dim=2)

        return video_embeddings


class MusicConverter:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        self._clap_processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")

    def __call__(
        self, music_segments: torch.Tensor, sampling_rate: int = 48000
    ) -> torch.Tensor:
        music_segments_shape = music_segments.shape
        flat_audio = music_segments.squeeze(-2).view(-1, music_segments_shape[3]).cpu()
        audio_list = [x.numpy() for x in flat_audio]

        inputs = self._clap_processor(audio=audio_list, return_tensors="pt", sampling_rate=sampling_rate, padding=True)
        with torch.no_grad():
            audio_features = self._clap_model.get_audio_features(**inputs)

        return audio_features.pooler_output.view(music_segments_shape[0], music_segments_shape[1], -1).to(self.device)


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

        self.output_projection = nn.Linear(self.embed_dim, query_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        self.attention_dropout(attn_probs)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(
            1, 2
        )

    def combine_heads(self, x):
        batch_size, _, seq_length, _ = x.size()
        return (
            x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        )

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
    ):

        if key is None:
            key = query
        if value is None:
            value = key

        key_len = key.size(1)
        value_len = value.size(1)

        assert key_len == value_len, "Key and value must have same sequence length"

        query = self.split_heads(self.W_q(query))
        key = self.split_heads(self.W_k(key))
        value = self.split_heads(self.W_v(value))

        attn_output = self.scaled_dot_product_attention(query, key, value, None)
        output = self.W_o(self.combine_heads(attn_output))

        output = self.output_projection(output)
        output = self.output_dropout(output)

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
        mlp_ratio: int = 4,
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

        mlp_hidden = query_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(query_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, query_dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_output))
        x = self.norm2(x + self.mlp(x))
        return x
