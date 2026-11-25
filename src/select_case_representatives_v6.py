import argparse
import os

import pandas as pd


DEFAULT_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select representative proteins for direct-site, neighborhood-loop, and distal hotspot cases."
    )
    parser.add_argument(
        "--case_rank_csv",
        required=True,
        help="Case ranking CSV from evaluate_binding_site_hotspots_v6.py",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Output representative case table. Default: <binding_type>_representative_cases_v6.csv",
    )
    parser.add_argument(
        "--top_n_per_type",
        type=int,
        default=5,
        help="How many candidates to retain for each case type.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.case_rank_csv, low_memory=False, encoding="utf-8-sig")
    if "binding_type" not in df.columns:
        df["binding_type"] = "unknown"

    required = {"case_type_suggestion", "case_priority_score", "pdb_chain"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"case_rank_csv missing required columns: {missing}")

    rows = []
    preferred_order = [
        "direct_site_hotspot",
        "site_neighborhood_hotspot",
        "distal_hotspot",
    ]

    for case_type in preferred_order:
        sub = df[df["case_type_suggestion"].astype(str).eq(case_type)].copy()
        if sub.empty:
            continue
        sub["case_priority_score"] = pd.to_numeric(sub["case_priority_score"], errors="coerce")
        sub = sub.sort_values(
            ["case_priority_score", "near_site_auprc", "near_site_top10_overlap_rate"],
            ascending=[False, False, False],
            kind="stable",
        )
        sub = sub.head(args.top_n_per_type).copy()
        sub.insert(0, "representative_case_type", case_type)
        rows.append(sub)

    if not rows:
        raise ValueError("No representative cases were selected.")

    out = pd.concat(rows, ignore_index=True, sort=False)
    binding_type = str(out["binding_type"].dropna().iloc[0]) if "binding_type" in out.columns and not out.empty else "binding"
    output_csv = args.output_csv or os.path.join(DEFAULT_BASE_DIR, f"{binding_type}_representative_cases_v6.csv")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"Saved representative cases: {output_csv}")
    print(out[[
        "representative_case_type",
        "pdb_chain",
        "case_priority_score",
        "top1_hotspot_min_site_dist",
        "top10_min_hotspot_site_dist",
        "near_site_auprc",
        "near_site_top10_overlap_rate",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
