import torch
import torch.nn.functional as F


def build_pairs(mos: torch.Tensor, margin: float = 0.1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Build pairs of samples based on MOS scores."""
    idx_pos, idx_neg, labels = [], [], []
    for i in range(len(mos)):
        diff = mos[i] - mos
        pos = torch.where(diff > margin)[0]
        neg = torch.where(diff < -margin)[0]
        for j in pos:
            idx_pos.append(i)
            idx_neg.append(j)
            labels.append(1.0)
        for j in neg:
            idx_pos.append(i)
            idx_neg.append(j)
            labels.append(0.0)
    if len(labels) == 0:
        return torch.tensor(idx_pos, dtype=torch.long), torch.tensor(idx_neg, dtype=torch.long), None
    return (
        torch.tensor(idx_pos, dtype=torch.long),
        torch.tensor(idx_neg, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float),
    )


def ranknet_loss(scores: torch.Tensor, mos: torch.Tensor, sigma: float = 1.0, margin: float = 0.1) -> torch.Tensor:
    """scores: predictor 出力 (B,).

    mos: GT MOS (B,)
    戻り値: scalar
    """
    i_idx, j_idx, y = build_pairs(mos, margin=margin)
    if y is None:
        return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
    device = scores.device
    i_idx = i_idx.to(device)
    j_idx = j_idx.to(device)
    y = y.to(device=device, dtype=scores.dtype)
    # Compute logits (difference of scores scaled by sigma)
    logits = sigma * (scores[i_idx] - scores[j_idx])
    # Use BCEWithLogits for stability (applies sigmoid internally)
    return F.binary_cross_entropy_with_logits(logits, y)


def mosdiff_loss(scores: torch.Tensor, mos: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    """scores: predictor 出力 (B,).

    mos: GT MOS (B,)
    戻り値: scalar
    """
    diff_gt = mos.unsqueeze(1) - mos.unsqueeze(0)
    diff_pred = scores.unsqueeze(1) - scores.unsqueeze(0)

    loss_mat = (diff_gt - diff_pred).abs() - margin
    loss_mat = torch.clamp(loss_mat, min=0)

    # use each unordered pair only once (upper-triangular, i<j)
    triu_mask = torch.triu(torch.ones_like(loss_mat, dtype=torch.bool), diagonal=1)
    loss = loss_mat[triu_mask].mean()
    return loss
