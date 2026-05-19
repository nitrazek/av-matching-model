import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F


@torch.no_grad()
def get_accuracy(dataloader: DataLoader, music_transformer, video_transformer, device):
    """
    Computes bidirectional Recall@1 and Recall@5 accuracy for cross-modal retrieval.

    Args:
        dataloader: PyTorch DataLoader yielding (video_features, music_features)
        music_transformer: The music embedding network
        video_transformer: The video embedding network
        device: torch.device (cuda or cpu)
    Returns:
        dict: A dictionary containing R@1 and R@5 for both V2M and M2V directions,
              as well as their averages.
    """
    music_transformer.eval()
    video_transformer.eval()

    all_video_embs = []
    all_music_embs = []

    # 1. Extract and pool embeddings across the entire evaluation dataset
    for video_feat, music_feat in dataloader:
        video_feat, music_feat = video_feat.to(device), music_feat.to(device)

        # Re-create masks matching your training pipeline
        v_mask = (video_feat.abs().sum(dim=-1) != 0).float().unsqueeze(-1)
        m_mask = (music_feat.abs().sum(dim=-1) != 0).float().unsqueeze(-1)

        # Forward passes
        v_out = video_transformer(video_feat)
        m_out = music_transformer(music_feat)

        # Global average pooling using sequence masks
        v_emb = (v_out * v_mask).sum(dim=1) / v_mask.sum(dim=1).clamp(min=1)
        m_emb = (m_out * m_mask).sum(dim=1) / m_mask.sum(dim=1).clamp(min=1)

        # Normalize to project onto unit hypersphere (crucial for accurate cosine similarity)
        all_video_embs.append(F.normalize(v_emb, dim=-1))
        all_music_embs.append(F.normalize(m_emb, dim=-1))

    all_video_embs = torch.cat(all_video_embs, dim=0)
    all_music_embs = torch.cat(all_music_embs, dim=0)

    num_samples = all_video_embs.size(0)
    if num_samples == 0:
        return {
            "v2m_r1": 0.0,
            "v2m_r5": 0.0,
            "m2v_r1": 0.0,
            "m2v_r5": 0.0,
            "mean_recall": 0.0,
        }

    sim_matrix = torch.matmul(all_video_embs, all_music_embs.T)
    targets = torch.arange(num_samples, device=device)

    _, v2m_indices = torch.sort(sim_matrix, descending=True, dim=1)

    v2m_r1 = (v2m_indices[:, 0] == targets).float().mean().item()
    v2m_r5 = (
        (v2m_indices[:, :5] == targets.unsqueeze(1)).any(dim=1).float().mean().item()
    )

    _, m2v_indices = torch.sort(sim_matrix.T, descending=True, dim=1)

    m2v_r1 = (m2v_indices[:, 0] == targets).float().mean().item()
    m2v_r5 = (
        (m2v_indices[:, :5] == targets.unsqueeze(1)).any(dim=1).float().mean().item()
    )
    metrics = {
        "v2m_R1": v2m_r1,
        "v2m_R5": v2m_r5,
        "m2v_R1": m2v_r1,
        "m2v_R5": m2v_r5,
        "mean_R1": (v2m_r1 + m2v_r1) / 2.0,
        "mean_R5": (v2m_r5 + m2v_r5) / 2.0,
    }

    return metrics
