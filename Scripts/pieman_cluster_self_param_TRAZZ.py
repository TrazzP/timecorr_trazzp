import sys
sys.path.append('/app')
import os
import numpy as np
import pandas as pd
import timecorr as tc
from timecorr.helpers import isfc, wisfc, mean_combine, corrmean_combine

# Read command-line arguments
cond = sys.argv[1]
factors = int(sys.argv[2])
level = sys.argv[3]
reps = sys.argv[4]
cfun = sys.argv[5]
rfun = sys.argv[6]
width = int(sys.argv[7])
wp = sys.argv[8]

# Change this later so I can check if running on local machine or on cluster
debug = False

# Set up directories
result_name = 'level_analysis_optimized_param_search'
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data/inital_data')  # Set 'inital_data' as the base directory

# Results directory with debug flag handling
results_dir = os.path.join(data_dir, result_name, f"{cfun}_{rfun}_{wp}_{width}{'_debug' if debug else ''}")
os.makedirs(results_dir, exist_ok=True)

# Define weight functions
laplace = {'name': 'Laplace', 'weights': tc.laplace_weights, 'params': {'scale': width}}
delta = {'name': '$\delta$', 'weights': tc.eye_weights, 'params': tc.eye_params}
gaussian = {'name': 'Gaussian', 'weights': tc.gaussian_weights, 'params': {'var': width}}
mexican_hat = {'name': 'Mexican hat', 'weights': tc.mexican_hat_weights, 'params': {'sigma': width}}
weights_param = eval(wp)

# Load the data
file_path = os.path.join(data_dir, f'pieman_data_{factors}_{cond}.npy')
if os.path.exists(file_path):
    data = np.load(file_path, allow_pickle=True)
else:
    raise FileNotFoundError(f"Data file {file_path} not found!")

# Prepare the results container
append_iter = pd.DataFrame()

# Perform the weighted timepoint decoding
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

# Print the iteration results
iter_results['iteration'] = int(reps)

# Save the results to a CSV file
save_file = os.path.join(results_dir, f'{cond}.csv')

# If file exists, append; otherwise, create new
if not os.path.isfile(save_file):
    iter_results.to_csv(save_file)
else:
    append_iter = pd.read_csv(save_file, index_col=0)
    append_iter = append_iter.append(iter_results)
    append_iter.to_csv(save_file)

print(f"Results saved to: {save_file}")

# Example usage: 
# python3 pieman_cluster_param_search.py intact 100 10 10 isfc PCA 5 gaussian
