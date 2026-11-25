# evaluate_v3.py
"""
Evaluate BFactorPredictor V3 on the dataset_v3 test split.

Outputs:
- summary_v3.json
- per_protein_results_v3.json
- bottom_low_pcc_v3.csv
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
from scipy import stats

from config_v6 import Config
from dataset_v6 import get_dataloaders
from loss_v6 import CombinedLoss
from model_v3 import BFactorPredictor


def move_to_device(data, device):
    out = {}
    for key, value in data.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out


def to_numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def compute_metrics_np(pred, target):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    n = min(len(pred), len(target))
    pred = pred[:n]
    target = target[:n]

    mask = np.isfinite(pred) & np.isfinite(target)
    pred = pred[mask]
    target = target[mask]
    n = len(pred)

    if n == 0:
        return {"mse": 0.0, "mae": 0.0, "rmse": 0.0, "pcc": 0.0, "scc": 0.0}

    diff = pred - target
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))

    if n >= 2 and np.std(pred) > 1e-12 and np.std(target) > 1e-12:
        pcc = float(stats.pearsonr(pred, target)[0])
        scc = float(stats.spearmanr(pred, target)[0])
    else:
        pcc = 0.0
        scc = 0.0

    return {"mse": mse, "mae": mae, "rmse": rmse, "pcc": pcc, "scc": scc}


def array_stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "range": 0.0}

    min_value = float(np.min(values))
    max_value = float(np.max(values))
    return {
        "min": min_value,
        "max": max_value,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "range": max_value - min_value,
    }


def get_first_scalar(data, keys):
    for key in keys:
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().view(-1)[0].item())
        if isinstance(value, (list, tuple)):
            return float(value[0])
        return float(value)
    return None


def denormalize_pred(pred_norm_np, target_norm_np, target_raw_np, data):
    mean = get_first_scalar(data, ("bf_mean", "mean", "b_mean"))
    std = get_first_scalar(data, ("bf_std", "std", "b_std"))
    if mean is not None and std is not None and abs(std) > 1e-12:
        return pred_norm_np * std + mean

    norm_params = data.get("norm_params")
    if isinstance(norm_params, dict):
        mean = norm_params.get("mean", norm_params.get("bf_mean"))
        std = norm_params.get("std", norm_params.get("bf_std"))
        if mean is not None and std is not None and abs(float(std)) > 1e-12:
            return pred_norm_np * float(std) + float(mean)

    if target_raw_np is not None and len(target_raw_np) == len(target_norm_np):
        x = np.asarray(target_norm_np, dtype=np.float64).reshape(-1)
        y = np.asarray(target_raw_np, dtype=np.float64).reshape(-1)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if len(x) >= 2 and np.var(x) > 1e-12:
            scale = float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
            offset = float(np.mean(y) - scale * np.mean(x))
            return pred_norm_np * scale + offset

    return pred_norm_np.copy()


def get_pdb_id(data):
    pdb_id = data.get("pdb_id", data.get("name", "unknown"))
    if isinstance(pdb_id, (list, tuple)):
        pdb_id = pdb_id[0]
    return str(pdb_id)


def get_num_residues(data, fallback):
    value = data.get("num_residues", data.get("length", fallback))
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().view(-1)[0].item())
    if isinstance(value, (list, tuple)):
        return int(value[0])
    return int(value)


def write_low_pcc_diagnostics(per_protein, output_dir, bottom_n=100):
    rows = sorted(per_protein, key=lambda x: x.get("pcc_norm", x.get("pcc", 0.0)))[:bottom_n]
    out_path = os.path.join(output_dir, "bottom_low_pcc_v3.csv")

    fieldnames = [
        "rank",
        "name",
        "length",
        "pcc_norm",
        "scc_norm",
        "mse_norm",
        "mae_norm",
        "rmse_norm",
        "pcc_raw",
        "scc_raw",
        "mse_raw",
        "mae_raw",
        "rmse_raw",
        "pred_norm_min",
        "pred_norm_max",
        "pred_norm_mean",
        "pred_norm_std",
        "pred_norm_range",
        "target_norm_min",
        "target_norm_max",
        "target_norm_mean",
        "target_norm_std",
        "target_norm_range",
        "pred_raw_min",
        "pred_raw_max",
        "pred_raw_mean",
        "pred_raw_std",
        "pred_raw_range",
        "target_raw_min",
        "target_raw_max",
        "target_raw_mean",
        "target_raw_std",
        "target_raw_range",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rank, row_data in enumerate(rows, start=1):
            row = {
                "rank": rank,
                "name": row_data.get("name", "unknown"),
                "length": row_data.get("length", 0),
                "pcc_norm": row_data.get("pcc_norm", row_data.get("pcc", 0.0)),
                "scc_norm": row_data.get("scc_norm", row_data.get("scc", 0.0)),
                "mse_norm": row_data.get("mse_norm", 0.0),
                "mae_norm": row_data.get("mae_norm", 0.0),
                "rmse_norm": row_data.get("rmse_norm", 0.0),
                "pcc_raw": row_data.get("pcc", 0.0),
                "scc_raw": row_data.get("scc", 0.0),
                "mse_raw": row_data.get("mse", 0.0),
                "mae_raw": row_data.get("mae", 0.0),
                "rmse_raw": row_data.get("rmse", 0.0),
            }

            for prefix in ("pred_norm", "target_norm", "pred_raw", "target_raw"):
                stats_dict = array_stats(row_data.get(prefix, []))
                for stat_name, value in stats_dict.items():
                    row[f"{prefix}_{stat_name}"] = value

            writer.writerow(row)

    return out_path


@torch.no_grad()
def evaluate_model_v3(checkpoint_path: str, config: Config = None, output_dir: str = None):
    config = config or Config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    config.device = str(device)

    if output_dir is None:
        output_dir = os.path.join(config.log_dir, "evaluation_v3")
    os.makedirs(output_dir, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "config" in ckpt and isinstance(ckpt["config"], dict):
        for key, value in ckpt["config"].items():
            if hasattr(config, key) and key != "device":
                setattr(config, key, value)

    _, _, test_loader = get_dataloaders(config)

    model = BFactorPredictor(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    criterion = CombinedLoss(config)

    per_protein = []
    total = {"loss": 0.0, "mse": 0.0, "mae": 0.0, "rmse": 0.0, "pcc": 0.0, "scc": 0.0}
    total_norm = {"mse": 0.0, "mae": 0.0, "rmse": 0.0, "pcc": 0.0, "scc": 0.0}
    count = 0

    for data in test_loader:
        data = move_to_device(data, device)

        target_norm = data["bfactors"].view(-1)
        pred_norm = model(data).view(-1)

        n = min(pred_norm.numel(), target_norm.numel())
        pred_norm = pred_norm[:n]
        target_norm = target_norm[:n]

        loss, _ = criterion(pred_norm, target_norm)

        pred_norm_np = to_numpy(pred_norm).reshape(-1)
        target_norm_np = to_numpy(target_norm).reshape(-1)

        if "bfactors_raw" in data:
            target_raw_np = to_numpy(data["bfactors_raw"]).reshape(-1)[:n]
        else:
            target_raw_np = target_norm_np.copy()

        pred_raw_np = denormalize_pred(pred_norm_np, target_norm_np, target_raw_np, data)
        pred_raw_np = np.asarray(pred_raw_np).reshape(-1)[:n]

        m = compute_metrics_np(pred_raw_np, target_raw_np)
        m_norm = compute_metrics_np(pred_norm_np, target_norm_np)

        num_res = get_num_residues(data, fallback=n)

        per_protein.append({
            "name": get_pdb_id(data),
            "length": int(num_res),
            "pcc": m["pcc"],
            "scc": m["scc"],
            "mse": m["mse"],
            "mae": m["mae"],
            "rmse": m["rmse"],
            "pcc_norm": m_norm["pcc"],
            "scc_norm": m_norm["scc"],
            "mse_norm": m_norm["mse"],
            "mae_norm": m_norm["mae"],
            "rmse_norm": m_norm["rmse"],
            "pred_raw": pred_raw_np.tolist(),
            "target_raw": target_raw_np.tolist(),
            "pred_norm": pred_norm_np.tolist(),
            "target_norm": target_norm_np.tolist(),
        })

        total["loss"] += float(loss.item())
        for key in ("mse", "mae", "rmse", "pcc", "scc"):
            total[key] += float(m[key])
            total_norm[key] += float(m_norm[key])
        count += 1

    denom = max(count, 1)
    summary = {
        "n_proteins": count,
        "loss_mean": total["loss"] / denom,
        "mse_mean": total["mse"] / denom,
        "mae_mean": total["mae"] / denom,
        "rmse_mean": total["rmse"] / denom,
        "pcc_mean": total["pcc"] / denom,
        "scc_mean": total["scc"] / denom,
        "mse_norm_mean": total_norm["mse"] / denom,
        "mae_norm_mean": total_norm["mae"] / denom,
        "rmse_norm_mean": total_norm["rmse"] / denom,
        "pcc_norm_mean": total_norm["pcc"] / denom,
        "scc_norm_mean": total_norm["scc"] / denom,
        "best_epoch": int(ckpt.get("epoch", -1)),
        "best_val_pcc": float(ckpt.get("val_pcc", 0.0)),
        "checkpoint": checkpoint_path,
    }

    summary_path = os.path.join(output_dir, "summary_v3.json")
    per_protein_path = os.path.join(output_dir, "per_protein_results_v3.json")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(per_protein_path, "w", encoding="utf-8") as f:
        json.dump(per_protein, f, indent=2)

    diag_path = write_low_pcc_diagnostics(per_protein, output_dir, bottom_n=100)

    print("\n[V3 Evaluate] Summary")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print(f"Saved summary     : {summary_path}")
    print(f"Saved per-protein : {per_protein_path}")
    print(f"Saved diagnostics : {diag_path}")

    return summary, per_protein


def main():
    parser = argparse.ArgumentParser(description="Evaluate V3 model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = Config()
    if args.device:
        config.device = args.device

    evaluate_model_v3(args.checkpoint, config, args.output_dir)


if __name__ == "__main__":
    main()
