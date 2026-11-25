import torch
import torch.nn as nn
import torch.nn.functional as F


class PCCLoss(nn.Module):
    """Pearson correlation loss: 1 - PCC."""
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1).to(pred.device)
        if pred.numel() != target.numel():
            raise RuntimeError(
                f"BFactorLoss shape mismatch: pred={tuple(pred.shape)}, "
                f"target={tuple(target.shape)}"
            )

        p = pred - pred.mean()
        t = target - target.mean()
        num = (p * t).sum()
        den = torch.sqrt((p ** 2).sum() * (t ** 2).sum() + 1e-8)
        return 1.0 - num / den


class RankLoss(nn.Module):
    """Sample residue pairs and preserve their relative B-factor ordering."""
    def __init__(self, num_pairs=512, margin=0.1):
        super().__init__()
        self.num_pairs = num_pairs
        self.margin = margin

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1).to(pred.device)
        length = pred.shape[0]
        if length < 2:
            return torch.tensor(0.0, device=pred.device)

        n_pairs = min(self.num_pairs, length * (length - 1) // 2)
        idx_i = torch.randint(0, length, (n_pairs,), device=pred.device)
        idx_j = torch.randint(0, length, (n_pairs,), device=pred.device)

        mask = idx_i != idx_j
        idx_i = idx_i[mask]
        idx_j = idx_j[mask]
        if idx_i.numel() == 0:
            return torch.tensor(0.0, device=pred.device)

        target_diff = target[idx_i] - target[idx_j]
        pred_diff = pred[idx_i] - pred[idx_j]
        sign = torch.sign(target_diff)

        valid = sign != 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        loss = F.relu(self.margin - sign[valid] * pred_diff[valid])
        return loss.mean()


class SmoothnessLoss(nn.Module):
    """Match adjacent-residue roughness between prediction and target."""
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1).to(pred.device)
        if pred.shape[0] < 2:
            return torch.tensor(0.0, device=pred.device)

        pred_diff = (pred[1:] - pred[:-1]) ** 2
        target_diff = (target[1:] - target[:-1]) ** 2
        return F.mse_loss(pred_diff, target_diff)


class VarianceMatchingLoss(nn.Module):
    """Prevent overly flat predictions by matching target standard deviation."""
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1).to(pred.device)
        if pred.numel() < 2:
            return torch.tensor(0.0, device=pred.device)

        pred_std = torch.sqrt(torch.var(pred, unbiased=False) + 1e-8)
        target_std = torch.sqrt(torch.var(target, unbiased=False) + 1e-8)
        return (pred_std - target_std).pow(2)


class DiffLoss(nn.Module):
    """Match local residue-to-residue B-factor changes."""
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1).to(pred.device)
        if pred.numel() < 3:
            return torch.tensor(0.0, device=pred.device)

        pred_diff = pred[1:] - pred[:-1]
        target_diff = target[1:] - target[:-1]

        mse = F.mse_loss(pred_diff, target_diff)
        p = pred_diff - pred_diff.mean()
        t = target_diff - target_diff.mean()
        den = torch.sqrt((p ** 2).sum() * (t ** 2).sum() + 1e-8)
        pcc_loss = 1.0 - (p * t).sum() / den
        return 0.5 * mse + 0.5 * pcc_loss


class CombinedLoss(nn.Module):
    """MSE + PCC + rank + smoothness + variance matching."""
    def __init__(self, config):
        super().__init__()
        self.mse = nn.MSELoss()
        self.pcc = PCCLoss()
        self.rank = RankLoss()
        self.smooth = SmoothnessLoss()
        self.var = VarianceMatchingLoss()
        self.diff = DiffLoss()

        self.w_mse = config.loss_mse_weight
        self.w_pcc = config.loss_pcc_weight
        self.w_rank = config.loss_rank_weight
        self.w_smooth = config.loss_smooth_weight
        self.w_var = getattr(config, 'loss_var_weight', 0.0)
        self.w_diff = getattr(config, 'loss_diff_weight', 0.0)

        
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1).to(pred.device)
        if pred.numel() != target.numel():
            raise RuntimeError(
                f"CombinedLoss shape mismatch: pred={tuple(pred.shape)}, "
                f"target={tuple(target.shape)}"
            )

        l_mse = self.mse(pred, target)
        l_pcc = self.pcc(pred, target)
        l_rank = self.rank(pred, target)
        l_smooth = self.smooth(pred, target)
        l_var = self.var(pred, target)
        l_diff = self.diff(pred, target)

        total = (
            self.w_mse * l_mse
            + self.w_pcc * l_pcc
            + self.w_rank * l_rank
            + self.w_smooth * l_smooth
            + self.w_var * l_var
            + self.w_diff * l_diff
        )

        return total, {
            'mse': l_mse.item(),
            'pcc': l_pcc.item(),
            'rank': l_rank.item(),
            'smooth': l_smooth.item(),
            'var': l_var.item(),
            'diff': l_diff.item(),
            'total': total.item(),
        }
