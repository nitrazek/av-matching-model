import torch
import pytest
import math
from unittest.mock import MagicMock, patch
from PIL import Image

# Assuming your classes are in a file named models.py
from scripts.src.models.base_model import (
    EmbeddingWithProjection,
    MultiHeadAttention,
    DecoderLayer,
    VideoTransformer,
    MusicTransformer,
    VideoConverter,
    MusicConverter,
)

## --- Configuration & Fixtures ---


@pytest.fixture
def dummy_batch():
    return torch.randn(2, 10, 512)  # [batch, seq_len, dim]


## --- 1. Embedding & Positional Encoding Tests ---


def test_positional_encoding_logic():
    seq_len, dim, batch = 5, 16, 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_enc = EmbeddingWithProjection.create_positional_encoding(
        seq_count=seq_len,
        embedding_dimensions=dim,
        batch_size=batch,
        device=device
    )

    assert pos_enc.shape == (batch, seq_len, dim)
    # Check if sine/cosine pattern exists (sin(0) should be 0)
    assert torch.allclose(pos_enc[0, 0, 0], torch.tensor(0.0), atol=1e-5)
    # Check if values are within expected bounds
    assert pos_enc.max() <= 1.0
    assert pos_enc.min() >= -1.0


def test_embedding_projection_forward(dummy_batch):
    dim = 512
    model = EmbeddingWithProjection(dim)
    output = model(dummy_batch)

    assert output.shape == dummy_batch.shape
    # Ensure LayerNorm was applied (mean should be close to 0)
    assert torch.abs(output.mean()).item() < 0.1


## --- 2. Attention & Decoder Layer Tests ---


def test_multi_head_attention_shapes(dummy_batch):
    dim = 512
    heads = 8
    mha = MultiHeadAttention(query_dim=dim, num_heads=heads)
    output = mha(dummy_batch)

    assert output.shape == dummy_batch.shape


def test_decoder_layer_residual_connection(dummy_batch):
    dim = 512
    layer = DecoderLayer(query_dim=dim)
    output = layer(dummy_batch)

    assert output.shape == dummy_batch.shape
    # Check that output is not identical to input (parameters changed it)
    assert not torch.equal(output, dummy_batch)


## --- 3. Full Transformer Model Tests ---


@pytest.mark.parametrize("ModelClass", [VideoTransformer, MusicTransformer])
def test_transformers_full_forward(ModelClass, dummy_batch):
    dim = 512
    model = ModelClass(num_layers=2, query_dim=dim)
    output = model(dummy_batch)

    assert output.shape == (2, 10, 512)


## --- 4. Converter Mock Tests ---


@patch("clip.load")
def test_video_converter_mock(mock_clip_load):
    # Setup mock clip
    mock_model = MagicMock()
    mock_model.encode_image.return_value = torch.randn(1, 512)
    mock_preprocess = MagicMock(return_value=torch.randn(3, 224, 224))
    mock_clip_load.return_value = (mock_model, mock_preprocess)

    converter = VideoConverter()
    # Create dummy nested list of PIL images
    dummy_videos = [[Image.new("RGB", (10, 10)), Image.new("RGB", (10, 10))]]

    result = converter(dummy_videos, segment_length=1.0)

    assert isinstance(result, list)
    assert len(result) == 1  # 1 video
    assert len(result[0]) == 2  # 2 frames


@patch("transformers.ClapModel.from_pretrained")
@patch("transformers.AutoProcessor.from_pretrained")
def test_music_converter_segmentation(mock_proc, mock_model):
    # Testing the static-like segmentation logic
    converter = MusicConverter()
    audio = torch.randn(1000)
    sr = 100
    seg_len = 2.0  # 200 samples per segment

    segments = converter(audio, sr, seg_len)

    assert len(segments) == 5  # 1000 / 200
    assert len(segments[0]) == 200


## --- 5. Error & Edge Case Handling ---


def test_mha_invalid_heads():
    # embed_dim 512 not divisible by 7
    with pytest.raises(AssertionError):
        MultiHeadAttention(query_dim=512, embed_dim=512, num_heads=7)


def test_mha_mismatched_kv():
    mha = MultiHeadAttention(query_dim=512)
    q = torch.randn(1, 10, 512)
    k = torch.randn(1, 5, 512)  # Length 5
    v = torch.randn(1, 8, 512)  # Length 8 (mismatch)

    with pytest.raises(
        AssertionError, match="Key and value must have same sequence length"
    ):
        mha(q, k, v)
