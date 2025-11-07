#!/usr/bin/env python3
"""
Kernel width comparison at factors==700.

Generates two figures saved in:
  /app/Graphs/Method_Value_By_Level/Kernel_Width/
    ├── Kernel_Width_All.png          (all widths 5..50, no CI)
    └── Kernel_Width_KeyWidths.png    (min, middle, max widths with 95% CI)

Assumes files in /app/Cluster_Data/{paragraph,word,intact,rest} named:
  <cond>_<factors>_<level>_<reps>_<cfun>_<rfun>_<width>_<wp...>.csv
and each CSV contains columns: level, accuracy
"""

from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---------------- CONFIG ----------------
INPUT_ROOT  = Path("/app/Cluster_Data")
CONDS       = ["paragraph", "word", "intact", "rest"]
OUTPUT_DIR  = Path("/app/Graphs/Method_Value_By_Level/Kernel_Width")
REQUIRE_FACTORS = "700"

# ---------------- HELPERS ----------------
def parse_stem(stem: str):
    """Return tuple: (cond, factors, level_from_name, reps, cfun, rfun, width, wp)"""
    parts = stem.split("_")
    if len(parts) < 8:
        raise ValueError(f"Unexpected filename stem: {stem}")
    cond, factors, level_name, reps, cfun, rfun, width = parts[:7]
    wp = "_".join(parts[7:])
    return cond, factors, level_name, reps, cfun, rfun, width, wp

def ci95(series: pd.Series):
    n = series.count()
    m = series.mean()
    if n < 2:
        return m, m
    sd = series.std(ddof=1)
    se = sd / math.sqrt(n)
    d = 1.96 * se
    return m - d, m + d

# ---------------- MAIN ----------------
def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    scanned = 0
    kept = 0

    for cond in CONDS:
        folder = INPUT_ROOT / cond
        if not folder.is_dir():
            continue
        for fp in folder.glob("*.csv"):
            scanned += 1
            try:
                c, factors, level_from_name, reps, cfun, rfun, width_str, wp = parse_stem(fp.stem)
            except ValueError:
                continue
            if factors != REQUIRE_FACTORS:
                continue
            try:
                df = pd.read_csv(fp)
            except Exception:
                continue
            if "level" not in df.columns or "accuracy" not in df.columns:
                continue

            # clean
            df = df.copy()
            df["level"] = pd.to_numeric(df["level"], errors="coerce")
            df = df.dropna(subset=["level", "accuracy"])

            # ensure integer width from filename
            try:
                width_val = int(str(width_str).lstrip("0") or "0")
            except Exception:
                # fallback if width is non-numeric
                try:
                    width_val = int(float(width_str))
                except Exception:
                    continue

            df["cond"] = c
            df["factors"] = factors
            df["reps"] = reps
            df["rfun"] = rfun
            df["wp"] = wp
            df["width"] = width_val

            rows.append(df[["cond","factors","reps","rfun","wp","width","level","accuracy"]])
            kept += 1

    if not rows:
        print("No usable data with factors==700.")
        return 0

    all_rows = pd.concat(rows, ignore_index=True)

    # Aggregate per unique key to avoid over-weighting duplicates:
    # Key holds all params EXCEPT accuracy; we average accuracy within key.
    key_cols = ["cond","factors","reps","rfun","wp","width","level"]
    per_key = all_rows.groupby(key_cols, as_index=False)["accuracy"].mean()

    # ---- Plot 1: All widths, no CI ----
    summary_all = (
        per_key.groupby(["width","level"])["accuracy"]
               .agg(mean_acc="mean")
               .reset_index()
               .sort_values(["width","level"])
    )

    # Sort widths and levels
    widths_sorted = sorted(summary_all["width"].unique())
    levels_sorted = sorted(summary_all["level"].unique())
    
    # Create a colormap (magma from cold→hot)
    cmap = plt.cm.get_cmap("magma", len(widths_sorted))
    norm = mcolors.Normalize(vmin=min(widths_sorted), vmax=max(widths_sorted))


    plt.figure(figsize=(6, 5))
    for w in widths_sorted:
        sub = summary_all[summary_all["width"] == w]
        if sub.empty:
            continue
        color = cmap(norm(w))
        plt.plot(sub["level"], sub["mean_acc"], color=color, marker="o", label=f"w={w}")
    plt.xlabel("Higher-order correlation level", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Kernel Width Comparison (All Widths)", fontsize=16)
    plt.grid(True, alpha=0.3)
    # With many lines, a full legend is okay—if it’s too busy, comment the next line.
    plt.legend(ncol=2, title="Width")
    plt.tight_layout()
    out_all = OUTPUT_DIR / "Kernel_Width_All.png"
    plt.savefig(out_all, dpi=200)
    plt.close()
    print(f"→ Saved: {out_all}")

    # ---- Plot 2: Key widths (min, middle, max) with 95% CI ----
    # Determine min, max, and 'middle' (lower median if even count)
    if len(widths_sorted) >= 3:
        min_w = widths_sorted[0]
        max_w = widths_sorted[-1]
        mid_idx = (len(widths_sorted) - 1) // 2  # lower median for even length
        mid_w = widths_sorted[mid_idx]
        key_widths = [min_w, mid_w, max_w]
    else:
        # Fallback: whatever exists
        key_widths = widths_sorted

    # Compute CI per (width, level)
    ci_map = per_key.groupby(["width","level"])["accuracy"].apply(ci95).to_dict()
    summary_ci = (
        per_key[per_key["width"].isin(key_widths)]
        .groupby(["width","level"])["accuracy"]
        .agg(count="count", mean_acc="mean", std_acc="std")
        .reset_index()
    )
    summary_ci["ci_lower"] = summary_ci.apply(lambda r: ci_map[(r["width"], r["level"])][0], axis=1)
    summary_ci["ci_upper"] = summary_ci.apply(lambda r: ci_map[(r["width"], r["level"])][1], axis=1)

    plt.figure(figsize=(6, 5))
    for w in key_widths:
        sub = summary_ci[summary_ci["width"] == w].sort_values("level")
        if sub.empty:
            continue
        plt.plot(sub["level"], sub["mean_acc"], marker="o", label=f"w={w}")
        plt.fill_between(sub["level"], sub["ci_lower"], sub["ci_upper"], alpha=0.2)
    plt.xlabel("Higher-order correlation level", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title(f"Important Kernel Widths", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Width")
    plt.tight_layout()
    out_key = OUTPUT_DIR / "Kernel_Width_KeyWidths.png"
    plt.savefig(out_key, dpi=200)
    plt.close()
    print(f"→ Saved: {out_key}")

    print("Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
