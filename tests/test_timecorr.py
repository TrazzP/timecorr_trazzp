
import numpy as np
import pandas as pd

import timecorr as tc
from timecorr.timecorr import timecorr
from timecorr.simulate import simulate_data
from timecorr.helpers import isfc, gaussian_weights, gaussian_params
from timecorr.braintools import brain_reduce, brain_format

#TODO: need *real* tests-- e.g. generate a small dataset and verify that we actually get the correct answers

#gaussian_params = {'var': 1000}
data_list= np.random.randn(10,3)
pandas_dataframe= pd.DataFrame(np.random.randint(low=0, high=10, size=(2, 2)))
numpy_array= np.array([[5, 9], [10, 7]])
numpy_array_list= np.array([[8,2],[4,6]]).tolist()

sim_1 = simulate_data(S=1, T=30, K=30, set_random_seed=100)
sim_3 = simulate_data(S=3, T=30, K=30, set_random_seed=100)

width = 10
laplace = {'name': 'Laplace', 'weights': tc.laplace_weights, 'params': {'scale': width}}


def test_reduce_shape():
    dyna_corrs_reduced_1 = timecorr(sim_1, rfun='PCA',
                                    weights_function=laplace['weights'], weights_params=laplace['params'])

    dyna_corrs_reduced_3 = timecorr(sim_3, rfun='PCA',
                                    weights_function=laplace['weights'], weights_params=laplace['params'])
    assert np.shape(dyna_corrs_reduced_1) == np.shape(sim_1)
    assert np.shape(dyna_corrs_reduced_3) == np.shape(sim_3)

def test_nans():
    sim_3[0][0] = np.nan
    dyna_corrs_reduced_3 = timecorr(sim_3, rfun='PCA',
                                    weights_function=laplace['weights'], weights_params=laplace['params'])

    assert np.shape(dyna_corrs_reduced_3) == np.shape(sim_3)


def test_include_timepoints_all():
    dyna_corrs_reduced_3 = timecorr(sim_3, rfun='PCA',
                                    weights_function=laplace['weights'], weights_params=laplace['params'],
                                    include_timepoints='all')

    assert np.shape(dyna_corrs_reduced_3) == np.shape(sim_3)


def test_include_timepoints_pre():
    dyna_corrs_reduced_3 = timecorr(sim_3, rfun='PCA',
                                    weights_function=laplace['weights'], weights_params=laplace['params'],
                                    include_timepoints='pre')

    assert np.shape(dyna_corrs_reduced_3) == np.shape(sim_3)


def test_include_timepoints_post():
    dyna_corrs_reduced_3 = timecorr(sim_3, rfun='PCA',
                                    weights_function=laplace['weights'], weights_params=laplace['params'],
                                    include_timepoints='post')

    assert np.shape(dyna_corrs_reduced_3) == np.shape(sim_3)

def test_exclude_timepoints_pos():
    dyna_corrs_reduced_3 = timecorr(sim_3, rfun='PCA',
                                    weights_function=laplace['weights'], weights_params=laplace['params'],
                                    exclude_timepoints=3)

    assert np.shape(dyna_corrs_reduced_3) == np.shape(sim_3)


def test_exclude_timepoints_neg():
    dyna_corrs_reduced_3 = timecorr(sim_3, rfun='PCA',
                                    weights_function=laplace['weights'], weights_params=laplace['params'],
                                    exclude_timepoints=-3)

    assert np.shape(dyna_corrs_reduced_3) == np.shape(sim_3)


def test_timecorr():
    # Format different types of input data
    data_dl = brain_format.format_data(data_list)
    data_pdf = brain_format.format_data(pandas_dataframe)
    data_npa = brain_format.format_data(numpy_array)
    data_npl = brain_format.format_data(numpy_array_list)

    # Check all outputs are lists
    assert isinstance(data_dl, list)
    assert isinstance(data_pdf, list)
    assert isinstance(data_npa, list)
    assert isinstance(data_npl, list)

    # Check shapes of first elements in each formatted list
    assert data_dl[0].shape[0] > 0, "data_dl first element has zero rows"
    assert data_pdf[0].shape[0] > 0, "data_pdf first element has zero rows"
    assert data_npa[0].shape[0] > 0, "data_npa first element has zero rows"
    assert data_npl[0].shape[0] > 0, "data_npl first element has zero rows"

    # Test gaussian_weights function
    dl_tester = gaussian_weights(data_dl[0].shape[0], params=gaussian_params)
    pdf_tester = gaussian_weights(data_pdf[0].shape[0], params=gaussian_params)
    npa_tester = gaussian_weights(data_npa[0].shape[0], params=gaussian_params)
    npl_tester = gaussian_weights(data_npl[0].shape[0], params=gaussian_params)
    
    # Verify gaussian_weights outputs
    assert isinstance(dl_tester, np.ndarray), "dl_tester is not a numpy array"
    assert dl_tester.shape == (data_dl[0].shape[0], data_dl[0].shape[0]), "dl_tester shape mismatch"
    assert npa_tester.shape == (data_npa[0].shape[0], data_npa[0].shape[0]), "npa_tester shape mismatch"

    # Verify no errors in handling other input formats
    assert pdf_tester.shape == (data_pdf[0].shape[0], data_pdf[0].shape[0]), "pdf_tester shape mismatch"
    assert npl_tester.shape == (data_npl[0].shape[0], data_npl[0].shape[0]), "npl_tester shape mismatch"


#unsure how to test 'across' mode

corrs = timecorr(numpy_array, weights_function=gaussian_weights, weights_params=gaussian_params, cfun=isfc)
#assert()
#assert len(corrs.get_time_data()[0]) == len(numpy_array)
