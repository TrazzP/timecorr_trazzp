#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ----------------------------------------------------------------------------------
# Search in Cluster_Data to find file
# ----------------------------------------------------------------------------------
def find_csv(cond, factors, level, reps, cfun, rfun, width, wp, root_dir="Cluster_Data"):
    """Construct the expected filename and return its Path under root_dir/cond."""
    root = Path(root_dir) / cond
    fname = f"{cond}_{factors}_{level}_{reps}_{cfun}_{rfun}_{width}_{wp}.csv"
    fp = root / fname
    if not fp.exists():
        raise FileNotFoundError(f"Could not find {fp}")
    return fp

# ----------------------------------------------------------------------------------
# Load File using pandas
# ----------------------------------------------------------------------------------
def load_data(fp):
    """Load CSV into a DataFrame."""
    return pd.read_csv(fp)

# ----------------------------------------------------------------------------------
# Create graph and save to initial_graphs/$cond/$factors/$rfun
# Title should be same as csv file it pulls from.
# ----------------------------------------------------------------------------------
def plot_average_accuracy(df: pd.DataFrame, title: str, out_path: Path):
    """Compute mean accuracy across folds and plot vs. level."""
    mean_acc = df.groupby("level")["accuracy"].mean().sort_index()

    y_min = max(0, mean_acc.min() - 0.05)  # don't go below 0
    y_max = mean_acc.max() + 0.05

    plt.figure(figsize=(8, 5))
    plt.plot(mean_acc.index, mean_acc.values, marker="o", label="Average Accuracy")

    plt.xlabel("Level")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # set y limits and ticks
    plt.ylim(y_min, y_max)
    plt.yticks(np.arange(0, y_max + 0.05, 0.05))  # only ticks spaced by ≥0.05

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ----------------------------------------------------------------------------------
# Run function for programmatic calls
# ----------------------------------------------------------------------------------
def run(cond, factors, level, reps, cfun, rfun, width, wp,
        input_root="/app/Cluster_Data", output_root="/app/Graphs/Single_Conditions"):
    """
    Find the CSV, load it, plot accuracy vs. level by fold,
    and save under output_root/cond/factors/rfun.
    Returns the path to the saved figure.
    """
    # locate and load
    csv_path = find_csv(cond, factors, level, reps, cfun, rfun, width, wp, root_dir=input_root)
    df = load_data(csv_path)

    # prepare output
    title = csv_path.stem
    out_dir = Path(output_root) / str(factors) / cond / rfun
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / f"{title}.png"

    # plot & save
    plot_average_accuracy(df, title, out_fp)
    return out_fp

# ----------------------------------------------------------------------------------
# CLI entrypoint
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    # expect exactly 8 args: cond, factors, level, reps, cfun, rfun, width, wp
    if len(sys.argv) != 9:
        print(f"Usage: {sys.argv[0]} <cond> <factors> <level> <reps> <cfun> <rfun> <width> <wp>")
        sys.exit(1)
    # parse & run
    cond, factors, level, reps, cfun, rfun, width, wp = (
        sys.argv[1],
        int(sys.argv[2]),
        sys.argv[3],
        sys.argv[4],
        sys.argv[5],
        sys.argv[6],
        int(sys.argv[7]),
        sys.argv[8],
    )
    out_path = run(cond, factors, level, reps, cfun, rfun, width, wp)
    print(f"Plot saved to {out_path}")