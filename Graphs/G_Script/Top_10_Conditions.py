#!/usr/bin/env python3
import sys
import pandas as pd
from pathlib import Path
import importlib.util

# -----------------------------------------------------------------------------
# Dynamically import `run` from One_Script.py
# -----------------------------------------------------------------------------
script_file = Path(__file__).parent / "One_Script.py"
spec = importlib.util.spec_from_file_location("one_script", script_file)
one_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(one_mod)
run = one_mod.run

# -----------------------------------------------------------------------------
# Directories and Exclusions
# -----------------------------------------------------------------------------
DATA_ROOT = Path("/app/Cluster_Data")
OUTPUT_ROOT = Path("/app/Graphs/Top10")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Specify which rfun (or other parts) to exclude
EXCLUDE_RFUN = {"LocallyLinearEmbedding"}
# EXCLUDE_CFUN = {...}
# EXCLUDE_COND = {...}

# -----------------------------------------------------------------------------
# Gather average accuracies, skipping excluded
# -----------------------------------------------------------------------------
results = []  # list of tuples (avg_accuracy, params_tuple)
for csv_path in DATA_ROOT.rglob("*.csv"):
    parts = csv_path.stem.split("_")
    if len(parts) < 8:
        continue
    cond, factors_str, level, reps, cfun, rfun, width_str = parts[:7]
    wp = "_".join(parts[7:])

    # Skip excluded functions or conditions
    if rfun in EXCLUDE_RFUN:
        continue
    # if cfun in EXCLUDE_CFUN: continue
    # if cond in EXCLUDE_COND: continue

    # Parse numeric
    try:
        factors = int(factors_str)
        width = int(width_str)
    except ValueError:
        continue

    # Compute average accuracy
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        continue
    if "accuracy" not in df.columns:
        continue
    avg_acc = df["accuracy"].mean()

    params = (cond, factors, level, reps, cfun, rfun, width, wp)
    results.append((avg_acc, params))

if not results:
    print("No data found after applying exclusions.")
    sys.exit(0)

# -----------------------------------------------------------------------------
# Select top 10 by average accuracy
# -----------------------------------------------------------------------------
top10 = sorted(results, key=lambda x: x[0], reverse=True)[:10]

# -----------------------------------------------------------------------------
# Generate plots for top 10
# -----------------------------------------------------------------------------
print("Generating Top 10 graphs in:", OUTPUT_ROOT)
for avg_acc, params in top10:
    cond, factors, level, reps, cfun, rfun, width, wp = params
    print(f"Plotting: {cond}_{factors}_{level}_{reps}_{cfun}_{rfun}_{width}_{wp} -> avg={avg_acc:.4f}")
    try:
        out_fp = run(cond, factors, level, reps, cfun, rfun, width, wp,
                     output_root=str(OUTPUT_ROOT))
        print(f"  Saved to {out_fp}")
    except Exception as e:
        print(f"  Failed to plot {params}: {e}")
