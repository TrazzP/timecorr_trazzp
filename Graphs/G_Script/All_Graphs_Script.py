#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path

# Path to the script that generates a single plot
SCRIPT = Path(__file__).parent / "One_Script.py"

# Base directory containing all CSVs
DATA_ROOT = Path("/app/Cluster_Data")
# Base directory for generated graphs
OUTPUT_ROOT = Path("/app/Graphs/initial_graphs")

if not SCRIPT.is_file():
    print(f"Error: could not find plot script at {SCRIPT}")
    sys.exit(1)

# Iterate through every CSV in Cluster_Data
for csv_path in DATA_ROOT.rglob("*.csv"):
    stem = csv_path.stem
    parts = stem.split("_")
    # Need at least 8 parts: cond, factors, level, reps, cfun, rfun, width, and wp (which may include underscores)
    if len(parts) < 8:
        print(f"Skipping unexpected filename: {csv_path.name}")
        continue

    cond = parts[0]
    factors = parts[1]
    level = parts[2]
    reps = parts[3]
    cfun = parts[4]
    rfun = parts[5]
    width = parts[6]
    # join all remaining parts as wp to account for underscores in wp
    wp = "_".join(parts[7:])

    # Determine expected output file
    out_fp = OUTPUT_ROOT / cond / factors / rfun / f"{stem}.png"
    if out_fp.is_file():
        print(f"Skipping {stem}: output already exists at {out_fp}")
        continue

    cmd = [
        sys.executable,
        str(SCRIPT),
        cond,
        factors,
        level,
        reps,
        cfun,
        rfun,
        width,
        wp,
    ]
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error for {csv_path.name}: {e.stderr.strip()}")
