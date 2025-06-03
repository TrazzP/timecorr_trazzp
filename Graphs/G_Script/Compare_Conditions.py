#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------------------------------------------------------
# CONFIGURATION: adjust these if your directory structure differs.
# -----------------------------------------------------------------------------
INPUT_ROOT   = Path("/app/Cluster_Data")
OUTPUT_ROOT  = Path("/app/Graphs/comparisons")
CONDS        = ["paragraph", "word", "intact", "rest"]
INTACT_DIR   = INPUT_ROOT / "intact"
# -----------------------------------------------------------------------------

def parse_params_from_stem(stem: str):
    """
    Stem format: <cond>_<factors>_<level>_<reps>_<cfun>_<rfun>_<width>_<wp>
    We assume cond == "intact" here; return the 7‐tuple (factors, level, reps,
    cfun, rfun, width, wp).  Raises ValueError if the stem doesn’t match.
    """
    parts = stem.split("_")
    if len(parts) < 8 or parts[0] != "intact":
        raise ValueError(f"Unexpected filename stem: {stem}")
    # Reconstruct wp if it contained underscores:
    factors = parts[1]
    level   = parts[2]
    reps    = parts[3]
    cfun    = parts[4]
    rfun    = parts[5]
    width   = parts[6]
    wp      = "_".join(parts[7:])
    return factors, level, reps, cfun, rfun, width, wp

def load_mean_series(cond, factors, level, reps, cfun, rfun, width, wp):
    """
    Try to load CSV at:
      /app/Cluster_Data/{cond}/{cond}_{factors}_{level}_{reps}_{cfun}_{rfun}_{width}_{wp}.csv
    Compute and return a pandas.Series indexed by level (as in the CSV) containing
    mean(accuracy) per level.  Raises FileNotFoundError if the file is missing.
    """
    fname = f"{cond}_{factors}_{level}_{reps}_{cfun}_{rfun}_{width}_{wp}.csv"
    csv_path = INPUT_ROOT / cond / fname
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")
    df = pd.read_csv(csv_path)
    # group by the CSV’s "level" column (which may or may not match our 'level' variable)
    return df.groupby("level")["accuracy"].mean().sort_index()

def main():
    # 1) Gather all stems under intact/
    if not INTACT_DIR.is_dir():
        print(f"Error: '{INTACT_DIR}' does not exist or is not a directory.")
        sys.exit(1)

    # Collect a set of unique parameter‐tuples by scanning filenames:
    tuples = []
    for csv_path in INTACT_DIR.glob("*.csv"):
        stem = csv_path.stem
        try:
            params = parse_params_from_stem(stem)
        except ValueError:
            print(f"Skipping unexpected file in intact/: {csv_path.name}")
            continue
        tuples.append((stem, params))

    if not tuples:
        print("No valid CSVs found in intact/.")
        sys.exit(0)

    # 2) For each (stem, params), attempt to load & plot across all four conditions
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for stem, (factors, level, reps, cfun, rfun, width, wp) in tuples:
        all_series = {}
        for cond in CONDS:
            try:
                series = load_mean_series(cond, factors, level, reps, cfun, rfun, width, wp)
                all_series[cond] = series
            except FileNotFoundError:
                # Simply skip any missing condition
                continue

        if not all_series:
            # Somehow even intact was missing (unlikely), so skip
            print(f"  → No data found for '{stem}' under any condition; skipping.")
            continue

        # 3) Plot all available conditions on one figure
        plt.figure(figsize=(8, 5))
        for cond, series in all_series.items():
            plt.plot(series.index, series.values, marker="o", label=cond.capitalize())

        plt.xlabel("Level")
        plt.ylabel("Accuracy")
        plt.title(f"Compare: {stem}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # 4) Save under /app/Graphs/comparisons/compare_<stem>.png
        out_name = f"compare_{stem}.png"
        out_fp   = OUTPUT_ROOT / out_name
        plt.savefig(out_fp)
        plt.close()
        print(f"  → Saved {out_fp}")

if __name__ == "__main__":
    main()
