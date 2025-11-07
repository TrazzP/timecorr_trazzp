#!/usr/bin/env python3
"""
Window-profile (wp) comparison at factors==700.

Generates two figures saved in:
  /app/Graphs/Method_Value_By_Level/Window_Profiles/
    ├── WP_Raw.png
    └── WP_Deltas.png  (if Gaussian present)

Behavior:
- Reads /app/Cluster_Data/{paragraph,word,intact,rest}/*.csv
  stem: <cond>_<factors>_<level>_<reps>_<cfun>_<rfun>_<width>_<wp...>.csv
- Filters factors == 700
- Compares wp ∈ {Gaussian, Laplace, MexicanHat} across levels
- Controls for other params via pairing on (cond, reps, rfun, width, level, factors)
"""

from pathlib import Path
import re
import math
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
INPUT_ROOT  = Path("/app/Cluster_Data")
CONDS       = ["paragraph", "word", "intact", "rest"]
OUTPUT_DIR  = Path("/app/Graphs/Method_Value_By_Level/Window_Profiles")
REQUIRE_FACTORS = "700"

CANON_WP    = ["Gaussian", "Laplace", "MexicanHat"]
BASE_WP     = "Gaussian"

# Normalize wp (lowercase, remove non-alnum, alias)
WP_ALIAS = {
    "gaussian": "Gaussian",
    "laplace": "Laplace",
    "laplacian": "Laplace",
    "mexicanhat": "MexicanHat",
    "mexican_hat": "MexicanHat",
    "mexican": "MexicanHat",
    "mexicanhats": "MexicanHat",
}

def parse_stem(stem: str):
    parts = stem.split("_")
    if len(parts) < 8:
        raise ValueError(f"Unexpected filename stem: {stem}")
    cond, factors, level, reps, cfun, rfun, width = parts[:7]
    wp = "_".join(parts[7:])
    return cond, factors, level, reps, cfun, rfun, width, wp

def norm_wp(name: str) -> str:
    s = name.strip()
    key = re.sub(r"[^a-z0-9]+", "", s.lower())
    return WP_ALIAS.get(key, s)

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
    wp_raw_seen = []

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

            df = df.copy()
            df["level"] = pd.to_numeric(df["level"], errors="coerce")
            df = df.dropna(subset=["level", "accuracy"])

            wp_raw_seen.append(wp)
            df["cond"] = c
            df["factors"] = factors
            df["reps"] = reps
            df["rfun"] = rfun
            df["width"] = width
            df["wp_norm"] = norm_wp(wp)

            rows.append(df[["cond","factors","reps","rfun","width","wp_norm","level","accuracy"]])

    if not rows:
        print("No usable data with factors==700.")
        return 0

    all_rows = pd.concat(rows, ignore_index=True)
    print("Observed wp (raw):")
    print(pd.Series(wp_raw_seen).value_counts())
    print("\nObserved wp (normalized):")
    print(all_rows["wp_norm"].value_counts())

    # Keep only canonical WPs
    raw_rows = all_rows[all_rows["wp_norm"].isin(CANON_WP)].copy()
    if raw_rows.empty:
        print("No canonical wp values found — check names above and extend WP_ALIAS if needed.")
        return 0

    # Aggregate to avoid duplicate overweighting, then summarize by wp, level
    key_cols = ["cond","factors","reps","rfun","width","wp_norm","level"]
    per_key = raw_rows.groupby(key_cols, as_index=False)["accuracy"].mean()

    # ---- RAW plot ----
    raw_summary = (
        per_key.groupby(["wp_norm","level"])["accuracy"]
               .agg(["mean","std","count"]).reset_index()
               .rename(columns={"mean":"mean_acc","std":"std_acc"})
    )
    ci_map = per_key.groupby(["wp_norm","level"])["accuracy"].apply(ci95).to_dict()
    raw_summary["ci_lower"] = raw_summary.apply(lambda r: ci_map[(r["wp_norm"], r["level"])][0], axis=1)
    raw_summary["ci_upper"] = raw_summary.apply(lambda r: ci_map[(r["wp_norm"], r["level"])][1], axis=1)

    plt.figure(figsize=(6,5))
    for wp_name in CANON_WP:
        sub = raw_summary[raw_summary["wp_norm"] == wp_name].sort_values("level")
        if sub.empty:
            continue
        plt.plot(sub["level"], sub["mean_acc"], marker="o", label=wp_name)
        plt.fill_between(sub["level"], sub["ci_lower"], sub["ci_upper"], alpha=0.2)
    plt.xlabel("Higher-order correlation level", fontsize=14)
    plt.ylabel("Accuracy", fontsize= 14)
    plt.title("Window Profile Comparison (Raw Accuracy)", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(title="wp")
    plt.tight_layout()
    out_raw = OUTPUT_DIR / "WP_Raw.png"
    plt.savefig(out_raw, dpi=200)
    plt.close()
    print(f"→ Saved: {out_raw}")

    # ---- Δ vs Gaussian plot ----
    if (per_key["wp_norm"] == BASE_WP).any():
        pivot_index = ["cond","factors","reps","rfun","width","level"]
        wide = per_key.pivot_table(index=pivot_index, columns="wp_norm", values="accuracy", aggfunc="mean")
        if BASE_WP not in wide.columns:
            print("Baseline Gaussian not found; skipping Δ plot.")
            return 0

        delta_frames = []
        for wp_name in [w for w in CANON_WP if w != BASE_WP and w in wide.columns]:
            sub = wide[[BASE_WP, wp_name]].dropna()
            if sub.empty:
                continue
            sub = sub.reset_index()
            sub["delta"] = sub[wp_name] - sub[BASE_WP]
            sub["wp_norm"] = wp_name
            delta_frames.append(sub)

        if not delta_frames:
            print("No overlap with Gaussian; skipping Δ plot.")
            return 0

        deltas = pd.concat(delta_frames, ignore_index=True)
        deltas_summary = (
            deltas.groupby(["wp_norm","level"])["delta"]
                  .agg(["mean","std","count"]).reset_index()
                  .rename(columns={"mean":"mean_delta","std":"std_delta"})
        )
        dci = deltas.groupby(["wp_norm","level"])["delta"].apply(ci95).to_dict()
        deltas_summary["ci_lower"] = deltas_summary.apply(lambda r: dci[(r["wp_norm"], r["level"])][0], axis=1)
        deltas_summary["ci_upper"] = deltas_summary.apply(lambda r: dci[(r["wp_norm"], r["level"])][1], axis=1)

        plt.figure(figsize=(6,5))
        all_levels = sorted(deltas_summary["level"].unique())
        plt.plot(all_levels, [0.0]*len(all_levels), label=f"{BASE_WP} (baseline)", linestyle="--")
        for wp_name in [w for w in CANON_WP if w != BASE_WP]:
            sub = deltas_summary[deltas_summary["wp_norm"] == wp_name].sort_values("level")
            if sub.empty:
                continue
            # Plot the line and capture the line color
            line, = plt.plot(sub["level"], sub["mean_delta"], marker="o", label=wp_name)
            color = line.get_color()  # automatically chosen color from the cycle

            # Use the same color for fill_between
            plt.fill_between(sub["level"], sub["ci_lower"], sub["ci_upper"],
                     color=color, alpha=0.2)
        plt.xlabel("Higher-order correlation level", fontsize=14)
        plt.ylabel(f"Accuracy Δ vs {BASE_WP}", fontsize=14)
        plt.title("Window Profile Δ Accuracy vs Gaussian", fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(title="wp")
        plt.tight_layout()
        out_delta = OUTPUT_DIR / "WP_Deltas.png"
        plt.savefig(out_delta, dpi=200)
        plt.close()
        print(f"→ Saved: {out_delta}")
    else:
        print("Gaussian not detected — generated only the raw accuracy plot.")

    print("Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
