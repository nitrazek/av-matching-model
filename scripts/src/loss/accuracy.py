import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def get_accuracy(dataloader: DataLoader, music_transformer, video_transformer, device):
    music_transformer.eval()
    video_transformer.eval()

    all_music = []
    all_video = []

    for video_feat, music_feat in dataloader:
        video_feat, music_feat = video_feat.to(device), music_feat.to(device)

        v_mask = (video_feat.abs().sum(dim=-1) != 0).float().unsqueeze(-1)
        m_mask = (music_feat.abs().sum(dim=-1) != 0).float().unsqueeze(-1)

        v_out = video_transformer(video_feat)
        m_out = music_transformer(music_feat)

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

    return {"R_1": r1, "R_5": r5}
