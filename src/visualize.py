# visualize_v6.py
"""
Visualization for V6 evaluation outputs.

Reads evaluate_v6.py output directory and saves:
- evaluation_summary_v6.png
- top_bottom_profiles_v6.png
- low_pcc_profiles_v6.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def get_metric(row, norm_key, raw_key=None, default=0.0):
    if norm_key in row:
        return row[norm_key]
    if raw_key is not None and raw_key in row:
        return row[raw_key]
    return default


def get_name(row):
    return row.get("name", row.get("pdb_id", "unknown"))


def get_length(row):
    return int(row.get("length", row.get("num_residues", 0)))


def get_profile_arrays(row, prefer_norm=False):
    if prefer_norm and "pred_norm" in row and "target_norm" in row:
        return np.asarray(row["pred_norm"]), np.asarray(row["target_norm"]), "Normalized B-factor"
    if "pred_raw" in row and "target_raw" in row:
        return np.asarray(row["pred_raw"]), np.asarray(row["target_raw"]), "Raw B-factor"
    if "pred_norm" in row and "target_norm" in row:
        return np.asarray(row["pred_norm"]), np.asarray(row["target_norm"]), "Normalized B-factor"
    return None, None, "B-factor"


def load_eval(eval_dir):
    candidates = [
        os.path.join(eval_dir, "per_protein_results_v3.json"),
        os.path.join(eval_dir, "per_protein_results_v6.json"),
        os.path.join(eval_dir, "ensemble_per_protein_v3.json"),
        os.path.join(eval_dir, "ensemble_per_protein_v6.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f), path
    raise FileNotFoundError(
        "No per-protein JSON found. Tried:\n" + "\n".join(candidates)
    )


def plot_eval_summary(per_protein, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    pccs = np.asarray([get_metric(x, "pcc_norm", "pcc") for x in per_protein], dtype=float)
    sccs = np.asarray([get_metric(x, "scc_norm", "scc") for x in per_protein], dtype=float)
    rmses = np.asarray([get_metric(x, "rmse_norm", "rmse") for x in per_protein], dtype=float)
    lengths = np.asarray([get_length(x) for x in per_protein], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.hist(pccs, bins=30, alpha=0.78, color="#3B82F6")
    ax.axvline(np.mean(pccs), color="black", linestyle="--", linewidth=1, label=f"mean={np.mean(pccs):.3f}")
    ax.axvline(np.median(pccs), color="#DC2626", linestyle=":", linewidth=1, label=f"median={np.median(pccs):.3f}")
    ax.set_title("PCC Distribution")
    ax.set_xlabel("PCC")
    ax.set_ylabel("Proteins")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.hist(sccs, bins=30, alpha=0.78, color="#F59E0B")
    ax.axvline(np.mean(sccs), color="black", linestyle="--", linewidth=1, label=f"mean={np.mean(sccs):.3f}")
    ax.set_title("SCC Distribution")
    ax.set_xlabel("SCC")
    ax.set_ylabel("Proteins")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.scatter(lengths, pccs, s=14, alpha=0.58, color="#10B981", edgecolors="none")
    ax.set_title("PCC vs Protein Length")
    ax.set_xlabel("Length")
    ax.set_ylabel("PCC")

    ax = axes[1, 1]
    ax.hist(rmses, bins=30, alpha=0.78, color="#6366F1")
    ax.set_title("RMSE Distribution")
    ax.set_xlabel("RMSE")
    ax.set_ylabel("Proteins")

    fig.suptitle(f"V6 Evaluation Summary | n={len(per_protein)} | mean PCC={np.mean(pccs):.4f}", y=1.02)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "evaluation_summary_v6.png")
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_profiles(rows, output_dir, filename, title, prefer_norm=False):
    os.makedirs(output_dir, exist_ok=True)
    if not rows:
        return

    fig, axes = plt.subplots(len(rows), 1, figsize=(14, 3.0 * len(rows)))
    if len(rows) == 1:
        axes = [axes]

    for ax, row in zip(axes, rows):
        pred, target, ylabel = get_profile_arrays(row, prefer_norm=prefer_norm)
        if pred is None or target is None:
            ax.text(0.5, 0.5, "No profile arrays found", ha="center", va="center")
            continue

        n = min(len(pred), len(target))
        x = np.arange(1, n + 1)
        pcc = get_metric(row, "pcc_norm", "pcc")
        scc = get_metric(row, "scc_norm", "scc")

        ax.plot(x, target[:n], label="True", linewidth=1.15, color="#111827")
        ax.plot(x, pred[:n], label="Pred", linewidth=1.15, color="#2563EB", alpha=0.9)
        ax.set_title(f"{get_name(row)} | L={get_length(row)} | PCC={pcc:.3f} | SCC={scc:.3f}")
        ax.set_xlabel("Residue index")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(title, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_top_bottom_profiles(per_protein, output_dir, n=3, prefer_norm=False):
    ranked = sorted(per_protein, key=lambda x: get_metric(x, "pcc_norm", "pcc"), reverse=True)
    rows = ranked[:n] + ranked[-n:]
    plot_profiles(
        rows,
        output_dir,
        "top_bottom_profiles_v6.png",
        f"Top {n} and Bottom {n} V6 Protein Profiles",
        prefer_norm=prefer_norm,
    )


def plot_low_pcc_profiles(per_protein, output_dir, n=10, prefer_norm=False):
    rows = sorted(per_protein, key=lambda x: get_metric(x, "pcc_norm", "pcc"))[:n]
    plot_profiles(
        rows,
        output_dir,
        "low_pcc_profiles_v6.png",
        f"Lowest {n} V6 Protein Profiles",
        prefer_norm=prefer_norm,
    )


def main():
    parser = argparse.ArgumentParser(description="Visualization for V6 evaluation")
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n", type=int, default=3, help="Top/bottom profile count")
    parser.add_argument("--low_n", type=int, default=10, help="Low-PCC profile count")
    parser.add_argument("--prefer_norm", action="store_true", help="Plot normalized profiles instead of raw when available")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.eval_dir, "figures_v6")
    per_protein, source_path = load_eval(args.eval_dir)
    print(f"Loaded: {source_path}")
    print(f"Proteins: {len(per_protein)}")

    plot_eval_summary(per_protein, output_dir)
    plot_top_bottom_profiles(per_protein, output_dir, n=args.n, prefer_norm=args.prefer_norm)
    plot_low_pcc_profiles(per_protein, output_dir, n=args.low_n, prefer_norm=args.prefer_norm)


if __name__ == "__main__":
    main()
