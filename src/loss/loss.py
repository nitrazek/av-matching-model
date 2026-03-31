import torch
import torch.nn as nn


def infonce_loss(music_emb, video_emb, temperature=0.07):
    """
    Symmetric Contrastive Loss (InfoNCE).

    Args:
        video_emb ([batch, embed_dim])
        music_emb ([batch, embed_dim])
    """

    music_emb = nn.functional.normalize(music_emb, dim=-1)
    video_emb = nn.functional.normalize(video_emb, dim=-1)

    logits = torch.matmul(video_emb, music_emb.T) / temperature
    labels = torch.arange(logits.size(0)).to(video_emb.device)

    loss_music = nn.functional.cross_entropy(logits.T, labels)
    loss_video = nn.functional.cross_entropy(logits, labels)

    return (loss_music + loss_video) / 2
