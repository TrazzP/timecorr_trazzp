import sys
import os
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------
# Command-Line Argument Parsing
# ----------------------------------------------------------------------------------
cond, factors, level, reps, cfun, rfun, width, wp, iteration = (
    sys.argv[1],
    int(sys.argv[2]),
    sys.argv[3],
    sys.argv[4],
    sys.argv[5],
    sys.argv[6],
    int(sys.argv[7]),
    sys.argv[8],
    int(sys.argv[9]),
    
)

# ----------------------------------------------------------------------------------
# Execution Context Detection (Local vs. Cluster) and Directory Setup
# ----------------------------------------------------------------------------------
cluster = False
if cluster:
    sys.path.append('/mnt/beegfs/hellgate/home/tp183485/timecorr_trazzp')
    #Change this to True if want to put files into the 10 iteration file
    if False:
        base_dir = os.path.join('/mnt/beegfs/hellgate/home/tp183485/timecorr_trazzp/Cluster_Data', '10_Iterations')
    else:
        base_dir = os.path.join('/mnt/beegfs/hellgate/home/tp183485/timecorr_trazzp/Cluster_Data')
    
    results_dir = os.path.join(base_dir, cond)
else:
    sys.path.append('/app')
    results_dir = os.path.join('/app/Cluster_Data/Local_Machine', cond)
    


data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'initial_data')
os.makedirs(results_dir, exist_ok=True)

import timecorr as tc
from timecorr.helpers import isfc, wisfc, mean_combine, corrmean_combine
# ----------------------------------------------------------------------------------
# Debug Mode Configuration
# ----------------------------------------------------------------------------------
debug = False
DEBUG_SAMPLE_SIZE = 10  # Number of samples to retain when debugging

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

np.random.seed(1337 + iteration)  # different fold splits each iteration

iter_results = tc.helpers.weighted_timepoint_decoder(
    data,
    nfolds=2,  # keep standard 2-fold CV
    optimize_levels=list(range(0, int(level) + 1)),
    level=int(level),
    combine=lambda x: np.asarray(corrmean_combine(x)),  # ensures ndarray output
    cfun=eval(cfun),
    rfun=rfun,
    weights_fun=weights_param["weights"],
    weights_params=weights_param["params"],
    opt_init="random",
)

iter_results["iteration"] = iteration
iter_results["reps_arg"] = int(reps)

final_df = iter_results

# ----------------------------------------------------------------------------------
# Results Persistence
# ----------------------------------------------------------------------------------
filename = f"{cond}_{factors}_{level}_{reps}_{cfun}_{rfun}_{width}_{wp}_{iteration}.csv"
save_file = os.path.join(results_dir, filename)

final_df.to_csv(save_file, index=False)


print(f" Experiment complete. Results at: {save_file}")


# ----------------------------------------------------------------------------------
# Example Usage
# ----------------------------------------------------------------------------------

# python3 Cluster_Grid_Param_TRAZZ.py intact 100 10 10 isfc PCA 5 gaussian 1
