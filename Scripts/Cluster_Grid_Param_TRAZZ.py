import sys
import os
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------
# Command-Line Argument Parsing
# ----------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------
# Execution Context Detection (Local vs. Cluster)
# ----------------------------------------------------------------------------------
cluster = True
if cluster:
    sys.path.append('/mnt/beegfs/hellgate/home/tp183485/timecorr_trazzp')
else:
    sys.path.append('/app')


import timecorr as tc
from timecorr.helpers import isfc, wisfc, mean_combine, corrmean_combine
# ----------------------------------------------------------------------------------
# Debug Mode Configuration
# ----------------------------------------------------------------------------------
debug = False
DEBUG_SAMPLE_SIZE = 10  # Number of samples to retain when debugging

# ----------------------------------------------------------------------------------
# Directory Setup
# ----------------------------------------------------------------------------------
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'initial_data')

results_dir = os.path.join('/mnt/beegfs/hellgate/home/tp183485/timecorr_trazzp/Cluster_Data', cond)
os.makedirs(results_dir, exist_ok=True)
# ----------------------------------------------------------------------------------
# Data Loading & Optional Debug Truncation
# ----------------------------------------------------------------------------------
file_path = os.path.join(data_dir, f'pieman_data_{factors}_{cond}.npy')
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Unable to locate data file at: {file_path}")

data = np.load(file_path, allow_pickle=True)

if debug:
    original_count = data.shape[0]
    data = data[:DEBUG_SAMPLE_SIZE]
    print(f"[DEBUG] Data truncated from {original_count} to {data.shape[0]} samples")

# ----------------------------------------------------------------------------------
# Weight Functions Definition
# ----------------------------------------------------------------------------------
laplace = {'name': 'Laplace', 'weights': tc.laplace_weights, 'params': {'scale': width}}
delta   = {'name': '$\\delta$',   'weights': tc.eye_weights,     'params': tc.eye_params}
gaussian= {'name': 'Gaussian',    'weights': tc.gaussian_weights,'params': {'var': width}}
mexican_hat = {'name': 'Mexican Hat', 'weights': tc.mexican_hat_weights, 'params': {'sigma': width}}
weights_param = eval(wp)

# ----------------------------------------------------------------------------------
# Weighted Timepoint Decoding
# ----------------------------------------------------------------------------------
iter_results = tc.helpers.weighted_timepoint_decoder(
    data,
    nfolds=2,
    optimize_levels=list(range(0, int(level) + 1)),
    level=int(level),
    combine=corrmean_combine,
    cfun=eval(cfun),
    rfun=rfun,
    weights_fun=weights_param['weights'],
    weights_params=weights_param['params']
)
iter_results['iteration'] = int(reps)

# ----------------------------------------------------------------------------------
# Results Persistence
# ----------------------------------------------------------------------------------
filename = f"{cond}_{factors}_{level}_{reps}_{cfun}_{rfun}_{width}_{wp}.csv"
save_file = os.path.join(results_dir, filename)

if not os.path.isfile(save_file):
    iter_results.to_csv(save_file, index=False)
else:
    existing = pd.read_csv(save_file)
    updated = pd.concat([existing, iter_results], ignore_index=True)
    updated.to_csv(save_file, index=False)

print(f"✨ Experiment complete. Results archived at: {save_file}")


# ----------------------------------------------------------------------------------
# Example Usage
# ----------------------------------------------------------------------------------

# python3 Cluster_Grid_Param_TRAZZ.py intact 100 10 10 isfc PCA 5 gaussian
