import sys
sys.path.append('/app')


import timecorr as tc
from timecorr.helpers import isfc, wisfc, mean_combine, corrmean_combine
from scipy.io import loadmat
import numpy as np
import os
import pandas as pd


cond = sys.argv[1]
level = sys.argv[2]
reps = sys.argv[3]
cfun = sys.argv[4]
rfun = sys.argv[5]
width = int(sys.argv[6])
wp = sys.argv[7]


#Change this later so I can check if running on local machine or on cluster
debug = False

result_name = 'level_analysis_optimized_param_search'
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')  # Set 'data' as the base directory

if debug:
    results_dir = os.path.join(data_dir, result_name, f"{sys.argv[4]}_{sys.argv[5]}_{sys.argv[7]}_{sys.argv[6]}_debug")
else:
    results_dir = os.path.join(data_dir, result_name, f"{sys.argv[4]}_{sys.argv[5]}_{sys.argv[7]}_{sys.argv[6]}")

try:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
except OSError as err:
    print(err)



laplace = {'name': 'Laplace', 'weights': tc.laplace_weights, 'params': {'scale': width}}
delta = {'name': '$\delta$', 'weights': tc.eye_weights, 'params': tc.eye_params}
gaussian = {'name': 'Gaussian', 'weights': tc.gaussian_weights, 'params': {'var': width}}
mexican_hat = {'name': 'Mexican hat', 'weights': tc.mexican_hat_weights, 'params': {'sigma': width}}

factors = 100

if factors == 100:
    pieman_name = 'pieman_ica100.mat'
else:
    pieman_name = 'pieman_data.mat'

pieman_data = loadmat(os.path.join(data_dir, pieman_name))
pieman_conds = ['intact', 'paragraph', 'word', 'rest']

weights_paramter = eval(wp)

#
# if debug:
#     data = []
#     conds = []
#     for c in pieman_conds:
#         next_data = list(map(lambda i: pieman_data[c][:, i][0][:30, :70], np.arange(4)))
#         data.extend(next_data)
#         conds.extend([c]*len(next_data))
#     del pieman_data
#

if debug:
    data = []
    conds = []
    for c in pieman_conds:
        print(c)
        if c == 'paragraph':
            if factors == 700:
                next_data = list(map(lambda i: pieman_data[c][:, i][0][:30, :70], np.where(np.arange(pieman_data[c].shape[1]) != 3)[0]))
            else:
                next_data = list(map(lambda i: pieman_data[c][:, i][0][:30, :70], np.where(np.arange(pieman_data[c].shape[1]) != 0)[0]))
        else:
            next_data = list(map(lambda i: pieman_data[c][:, i][0][:30, :70], np.arange(pieman_data[c].shape[1])))
        data.extend(next_data)
        conds.extend([c]*len(next_data))
    del pieman_data


else:

    data = []
    conds = []
    for c in pieman_conds:
        print(c)
        if c == 'paragraph':
            if factors == 700:
                next_data = list(map(lambda i: pieman_data[c][:, i][0], np.where(np.arange(pieman_data[c].shape[1]) != 3)[0]))
            else:
                next_data = list(map(lambda i: pieman_data[c][:, i][0], np.where(np.arange(pieman_data[c].shape[1]) != 0)[0]))
        else:
            next_data = list(map(lambda i: pieman_data[c][:, i][0], np.arange(pieman_data[c].shape[1])))
        data.extend(next_data)
        conds.extend([c]*len(next_data))
    del pieman_data


data = np.array(data, dtype=object)
conds = np.array(conds)
append_iter = pd.DataFrame()

iter_results = tc.helpers.weighted_timepoint_decoder(np.array(data[conds == cond]), nfolds=2, optimize_levels=list(range(0,int(level)+1)), level=int(level),
                                    combine=corrmean_combine,
                                    cfun=eval(cfun),
                                    rfun=rfun,
                                    weights_fun=weights_paramter['weights'],
                                    weights_params=weights_paramter['params'])

print(iter_results)
iter_results['iteration'] = int(reps)


save_file = os.path.join(results_dir, cond)


if not os.path.isfile(save_file + '.csv'):
      iter_results.to_csv(save_file + '.csv')
else:
    append_iter = pd.read_csv(save_file + '.csv', index_col=0)
    append_iter = append_iter._append(iter_results)
    append_iter.to_csv(save_file + '.csv')


#Example Script: python3 pieman_cluster_param_search.py intact 10 10 isfc PCA 5 gaussian