#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------------------------------------------------------
# CONFIGURATION: adjust these if your directory structure differs.
# -----------------------------------------------------------------------------
INPUT_ROOT    = Path("/app/Cluster_Data")
OUTPUT_ROOT   = Path("/app/Graphs/comparisons")
INTACT_DIR    = INPUT_ROOT / "intact"
CONDS         = ["rest", "word", "paragraph", "intact"]
TOPK          = 10
# Any rfun values here will be skipped
EXCLUDE_RFUN  = {"locallylinearembedding"}
# -----------------------------------------------------------------------------

def parse_params_from_stem(stem: str):
    parts = stem.split("_")
    if len(parts) < 8 or parts[0] != "intact":
        raise ValueError(f"Unexpected filename stem: {stem}")
    factors, level, reps, cfun, rfun, width = parts[1:7]
    wp = "_".join(parts[7:])
    return factors, level, reps, cfun, rfun, width, wp

def load_mean_series(cond, factors, level, reps, cfun, rfun, width, wp):
    fname = f"{cond}_{factors}_{level}_{reps}_{cfun}_{rfun}_{width}_{wp}.csv"
    path = INPUT_ROOT / cond / fname
    if not path.exists():
        raise FileNotFoundError
    df = pd.read_csv(path)
    return df.groupby("level")["accuracy"].mean().sort_index()

def score_curve(factors, level, reps, cfun, rfun, width, wp):
    intact = load_mean_series("intact", factors, level, reps, cfun, rfun, width, wp)
    others = []
    for cond in ("rest", "word", "paragraph"):
        try:
            s = load_mean_series(cond, factors, level, reps, cfun, rfun, width, wp)
            others.append(s)
        except FileNotFoundError:
            continue
    if not others:
        raise ValueError("Not enough conditions")
    mean_intact = intact.mean()
    mean_others = pd.concat(others, axis=1).mean(axis=1).mean()
    return mean_intact - mean_others

def plot_comparison(stem, params, out_dir):
    factors, level, reps, cfun, rfun, width, wp = params
    series = {}
    for cond in CONDS:
        try:
            series[cond] = load_mean_series(cond, factors, level, reps, cfun, rfun, width, wp)
        except FileNotFoundError:
            pass

    plt.figure(figsize=(8,5))
    for cond, s in series.items():
        plt.plot(s.index, s.values, marker="o", label=cond.capitalize())

    plt.xlabel("Level")
    plt.ylabel("Accuracy")

    base = f"compare_{factors}_{level}_{reps}_{cfun}_{rfun}_{width}_{wp}"
    plt.title(base)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    sub = out_dir / rfun
    sub.mkdir(parents=True, exist_ok=True)
    plt.savefig(sub / f"{base}.png")
    plt.close()

def main():
    if not INTACT_DIR.is_dir():
        print(f"Error: '{INTACT_DIR}' not found.", file=sys.stderr)
        sys.exit(1)

    items = []
    for csv_path in INTACT_DIR.glob("*.csv"):
        stem = csv_path.stem
        try:
            params = parse_params_from_stem(stem)
        except ValueError:
            continue

        # unpack to check rfun exclusion
        _, _, _, _, rfun, _, _ = params
        if rfun.lower() in EXCLUDE_RFUN:
            # skip any curves using the problematic method
            continue

        try:
            sc = score_curve(*params)
            items.append((stem, params, sc))
        except Exception:
            continue

    if not items:
        print("No valid curves found.", file=sys.stderr)
        sys.exit(0)

    # select top K by highest intact-vs-others difference
    items.sort(key=lambda x: x[2], reverse=True)
    topk = items[:TOPK]

    print(f"Top {TOPK} curves by (mean_intact − mean_others):")
    for rank, (stem, _, score) in enumerate(topk, 1):
        print(f"{rank:2d}. {stem} → {score:.4f}")

    out_dir = OUTPUT_ROOT / "Top10"
    for stem, params, _ in topk:
        plot_comparison(stem, params, out_dir)

    print(f"\nPlots saved under {out_dir}/<rfun>/compare_*.png")

if __name__ == "__main__":
    main()