#!/usr/bin/env python3
import sys
import os

# ----------------------------------------------------------------------------------
# Command-Line Argument Parsing
# ----------------------------------------------------------------------------------
cond    = sys.argv[1]
factors = int(sys.argv[2])
level   = int(sys.argv[3])
reps    = int(sys.argv[4])
cfun    = sys.argv[5]
rfun    = sys.argv[6]
width   = int(sys.argv[7])
wp      = sys.argv[8]

# ----------------------------------------------------------------------------------
# Ensure timecorr is on PYTHONPATH before import
# ----------------------------------------------------------------------------------
cluster = True
if cluster:
    sys.path.append('/mnt/beegfs/hellgate/home/tp183485/timecorr_trazzp')
else:
    sys.path.append('/app')

import numpy as np
import pandas as pd
import timecorr as tc
from timecorr.helpers import isfc, wisfc, mean_combine, corrmean_combine

# ----------------------------------------------------------------------------------
# Paths & Skip-If-Done Guard
# ----------------------------------------------------------------------------------
data_dir    = os.path.join(os.path.dirname(__file__), '..', 'data', 'inital_data')
results_dir = '/mnt/beegfs/hellgate/home/tp183485/timecorr_trazzp/Cluster_Data'
os.makedirs(results_dir, exist_ok=True)

filename  = f"{cond}_{factors}_{level}_{reps}_{cfun}_{rfun}_{width}_{wp}.csv"
save_path = os.path.join(results_dir, filename)

if os.path.isfile(save_path):
    print(f"⚠️  Skipping '{filename}' — output already exists.")
    sys.exit(0)

# ----------------------------------------------------------------------------------
# Load & (Optionally) Truncate Data
# ----------------------------------------------------------------------------------
file_path = os.path.join(data_dir, f'pieman_data_{factors}_{cond}.npy')
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Unable to locate data file at: {file_path}")

data = np.load(file_path, allow_pickle=True)

debug = False
if debug:
    data = data[:10]
    print(f"[DEBUG] Data truncated to {len(data)} samples")

# ----------------------------------------------------------------------------------
# Weight Functions Definition
# ----------------------------------------------------------------------------------
weight_defs = {
    'laplace':     {'weights': tc.laplace_weights,  'params': {'scale': width}},
    'delta':       {'weights': tc.eye_weights,      'params': tc.eye_params},
    'gaussian':    {'weights': tc.gaussian_weights, 'params': {'var': width}},
    'mexican_hat': {'weights': tc.mexican_hat_weights, 'params': {'sigma': width}},
}
weights_param = weight_defs.get(wp, eval(wp))

# ----------------------------------------------------------------------------------
# Run Weighted Timepoint Decoding
# ----------------------------------------------------------------------------------
iter_results = tc.helpers.weighted_timepoint_decoder(
    data,
    nfolds=2,
    optimize_levels=list(range(0, level + 1)),
    level=level,
    combine=corrmean_combine,
    cfun=eval(cfun),
    rfun=rfun,
    weights_fun=weights_param['weights'],
    weights_params=weights_param['params']
)
iter_results['iteration'] = reps

# ----------------------------------------------------------------------------------
# Persist Results
# ----------------------------------------------------------------------------------
iter_results.to_csv(save_path, index=False)
print(f"✨ Experiment complete. Results archived at: {save_path}")