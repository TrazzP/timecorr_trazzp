#!/usr/bin/env python3
"""
Reduction technique comparison at factors==700 (method from rfun).

Generates two figures:
1) Reduction_Technique_Raw.png      → accuracy by level (PCA, IncrementalPCA, FactorAnalysis, TruncatedSVD)
2) Reduction_Technique_Deltas.png   → Δ accuracy vs PCA by level (if PCA present)
"""

from pathlib import Path
import re
import math
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
INPUT_ROOT  = Path("/app/Cluster_Data")
CONDS       = ["paragraph", "word", "intact", "rest"]
OUTPUT_DIR  = Path("/app/Graphs/Method_Value_By_Level/Reduction_Techniques")
REQUIRE_FACTORS = "700"

CANON_METHODS = ["PCA", "IncrementalPCA", "FactorAnalysis", "TruncatedSVD"]
BASE_METHOD   = "PCA"

ALIAS_MAP = {
    "pca": "PCA",
    "incrementalpca": "IncrementalPCA",
    "incremental_pca": "IncrementalPCA",
    "ipca": "IncrementalPCA",
    "factoranalysis": "FactorAnalysis",
    "fa": "FactorAnalysis",
    "truncatedsvd": "TruncatedSVD",
    "truncsvd": "TruncatedSVD",
    "svd": "TruncatedSVD",
}

def parse_stem(stem: str):
    parts = stem.split("_")
    if len(parts) < 8:
        raise ValueError(f"Unexpected filename stem: {stem}")
    cond, factors, level, reps, cfun, rfun, width = parts[:7]
    wp = "_".join(parts[7:])
    return cond, factors, level, reps, cfun, rfun, width, wp

def normalize_method(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "", name.strip().lower())
    return ALIAS_MAP.get(s, name.strip())

def ci95(series: pd.Series):
    n = series.count()
    m = series.mean()
    if n < 2:
        return m, m
    sd = series.std(ddof=1)
    se = sd / math.sqrt(n)
    d = 1.96 * se
    return m - d, m + d

def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    rfun_raw_seen = []

    for cond in CONDS:
        folder = INPUT_ROOT / cond
        if not folder.is_dir():
            continue
        for fp in folder.glob("*.csv"):
            try:
                c, factors, level_from_name, reps, cfun, rfun, width, wp = parse_stem(fp.stem)
            except ValueError:
                continue
            if factors != REQUIRE_FACTORS:
                continue
            try:
                df = pd.read_csv(fp)
            except Exception:
                continue
            if "accuracy" not in df.columns or "level" not in df.columns:
                continue
            df["level"] = pd.to_numeric(df["level"], errors="coerce")
            df = df.dropna(subset=["level", "accuracy"])
            rfun_raw_seen.append(rfun)
            df["cond"] = c
            df["factors"] = factors
            df["reps"] = reps
            df["method"] = normalize_method(rfun)
            df["width"] = width
            df["wp"] = wp
            rows.append(df[["cond","factors","reps","method","width","wp","level","accuracy"]])

    if not rows:
        print("No usable data with factors==700.")
        return 0

    all_rows = pd.concat(rows, ignore_index=True)
    print("Observed rfun (raw):")
    print(pd.Series(rfun_raw_seen).value_counts())
    print("\nObserved methods (normalized):")
    print(all_rows["method"].value_counts())

    raw_rows = all_rows[all_rows["method"].isin(CANON_METHODS)].copy()
    if raw_rows.empty:
        print("No canonical reduction methods found — check rfun names above.")
        return 0

    key_cols = ["cond","factors","reps","width","wp","method","level"]
    per_key = raw_rows.groupby(key_cols, as_index=False)["accuracy"].mean()

    # ---- RAW plot ----
    raw_summary = (
        per_key.groupby(["method","level"])["accuracy"]
               .agg(["mean","std","count"]).reset_index()
               .rename(columns={"mean":"mean_acc","std":"std_acc"})
    )
    ci_map = per_key.groupby(["method","level"])["accuracy"].apply(ci95).to_dict()
    raw_summary["ci_lower"] = raw_summary.apply(lambda r: ci_map[(r["method"], r["level"])][0], axis=1)
    raw_summary["ci_upper"] = raw_summary.apply(lambda r: ci_map[(r["method"], r["level"])][1], axis=1)

    plt.figure(figsize=(6,5))
    for method in CANON_METHODS:
        sub = raw_summary[raw_summary["method"] == method].sort_values("level")
        if sub.empty:
            continue
        plt.plot(sub["level"], sub["mean_acc"], marker="o", label=method)
        plt.fill_between(sub["level"], sub["ci_lower"], sub["ci_upper"], alpha=0.2)
    plt.xlabel("Higher-order correlation level", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Reduction Technique Comparison (Raw Accuracy)", fontsize=15.5)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_raw = OUTPUT_DIR / "Reduction_Technique_Raw.png"
    plt.savefig(out_raw, dpi=200)
    plt.close()
    print(f"→ Saved: {out_raw}")

    # ---- Δ vs PCA plot ----
    if (per_key["method"] == BASE_METHOD).any():
        pivot_index = ["cond","factors","reps","width","wp","level"]
        wide = per_key.pivot_table(index=pivot_index, columns="method", values="accuracy", aggfunc="mean")

        if BASE_METHOD not in wide.columns:
            print("Baseline PCA not found; skipping Δ plot.")
            return 0

        delta_frames = []
        for method in [m for m in CANON_METHODS if m != BASE_METHOD and m in wide.columns]:
            sub = wide[[BASE_METHOD, method]].dropna()
            if sub.empty:
                continue
            sub = sub.reset_index()
            sub["delta"] = sub[method] - sub[BASE_METHOD]
            sub["method"] = method
            delta_frames.append(sub)
        if not delta_frames:
            print("No overlap with PCA; skipping Δ plot.")
            return 0

        deltas = pd.concat(delta_frames, ignore_index=True)
        deltas_summary = (
            deltas.groupby(["method","level"])["delta"]
                  .agg(["mean","std","count"]).reset_index()
                  .rename(columns={"mean":"mean_delta","std":"std_delta"})
        )
        dci = deltas.groupby(["method","level"])["delta"].apply(ci95).to_dict()
        deltas_summary["ci_lower"] = deltas_summary.apply(lambda r: dci[(r["method"], r["level"])][0], axis=1)
        deltas_summary["ci_upper"] = deltas_summary.apply(lambda r: dci[(r["method"], r["level"])][1], axis=1)

        plt.figure(figsize=(6,5))
        all_levels = sorted(deltas_summary["level"].unique())
        plt.plot(all_levels, [0.0]*len(all_levels), label="PCA (baseline)", linestyle="--")
        for method in ["IncrementalPCA","FactorAnalysis","TruncatedSVD"]:
            sub = deltas_summary[deltas_summary["method"] == method].sort_values("level")
            if sub.empty:
                continue
            plt.plot(sub["level"], sub["mean_delta"], marker="o", label=method)
            plt.fill_between(sub["level"], sub["ci_lower"], sub["ci_upper"], alpha=0.2)
        plt.xlabel("Higher-order correlation level", fontsize=14)
        plt.ylabel("Accuracy Δ vs PCA", fontsize=14)
        plt.title("Reduction Technique Δ Accuracy vs PCA", fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_delta = OUTPUT_DIR / "Reduction_Technique_Deltas.png"
        plt.savefig(out_delta, dpi=200)
        plt.close()
        print(f"→ Saved: {out_delta}")
    else:
        print("PCA not detected — generated only the raw accuracy plot.")

    print("Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
