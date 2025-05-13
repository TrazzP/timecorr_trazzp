import numpy as np
import pandas as pd
import pytest

from timecorr.braintools.brain_format import format_data
from timecorr.braintools.brain_reduce import reduce as reducer

#Format Data Tests
def test_np_array():
    data = np.random.rand(100,10)
    assert isinstance(format_data(data), list)
    assert isinstance(format_data(data)[0], np.ndarray)


def test_df():
    data = pd.DataFrame(np.random.rand(100,10))
    assert isinstance(format_data(data), list)
    assert isinstance(format_data(data)[0], np.ndarray)
    

global size
size = 40
#Reduce Function Tests
data = [np.random.multivariate_normal(np.zeros(4), np.eye(4), size=size) for i in range(2)]
reduced_data_2d = reducer(data, reduce='PCA', ndims=2)
reduced_data_1d = reducer(data,reduce='PCA',ndims=1)


def test_reduce_is_list():
    reduced_data_3d = reducer(data)
    assert type(reduced_data_3d) is list


def test_reduce_is_array():
    reduced_data_3d = reducer(data, ndims=3)
    assert isinstance(reduced_data_3d[0],np.ndarray)


def test_reduce_dims_3d():
    reduced_data_3d = reducer(data, ndims=3)
    assert reduced_data_3d[0].shape==(size,3)


def test_reduce_dims_2d():
    reduced_data_2d = reducer(data, ndims=2)
    assert reduced_data_2d[0].shape==(size,2)


def test_reduce_dims_1d():
    reduced_data_1d = reducer(data, ndims=1)
    assert reduced_data_1d[0].shape==(size,1)

def test_reduce_PCA():
    reduced_data_3d = reducer(data, reduce='PCA', ndims=3)
    assert reduced_data_3d[0].shape==(size,3)


def test_reduce_IncrementalPCA():
    reduced_data_3d = reducer(data, reduce='IncrementalPCA', ndims=3)
    assert reduced_data_3d[0].shape==(size,3)


def test_reduce_SparsePCA():
    reduced_data_3d = reducer(data, reduce='SparsePCA', ndims=3)
    assert reduced_data_3d[0].shape==(size,3)


def test_reduce_MiniBatchSparsePCA():
    reduced_data_3d = reducer(data, reduce='MiniBatchSparsePCA', ndims=3)
    assert reduced_data_3d[0].shape==(size,3)


def test_reduce_KernelPCA():
    reduced_data_3d = reducer(data, reduce='KernelPCA', ndims=3)
    assert reduced_data_3d[0].shape==(size,3)


def test_reduce_FastICA():
    reduced_data_3d = reducer(data, reduce='FastICA', ndims=3)
    assert reduced_data_3d[0].shape==(size,3)


def test_reduce_FactorAnalysis():
    reduced_data_3d = reducer(data, reduce='FactorAnalysis', ndims=3)
    assert reduced_data_3d[0].shape==(size,3)


def test_reduce_TruncatedSVD():
    reduced_data_3d = reducer(data, reduce='TruncatedSVD', ndims=3)
    assert reduced_data_3d[0].shape==(size,3)


def test_reduce_DictionaryLearning():
    reduced_data_3d = reducer(data, reduce='DictionaryLearning', ndims=3)
    assert reduced_data_3d[0].shape==(size,3)


def test_reduce_MiniBatchDictionaryLearning():
    reduced_data_3d = reducer(data, reduce='MiniBatchDictionaryLearning', ndims=3)
    assert reduced_data_3d[0].shape==(size,3)


def test_reduce_TSNE():
    reduced_data_3d = reducer(data, reduce='TSNE', ndims=3)
    assert reduced_data_3d[0].shape==(size,3)


def test_reduce_Isomap():
    reduced_data_3d = reducer(data, reduce='Isomap', ndims=3)
    assert reduced_data_3d[0].shape==(size,3)


def test_reduce_SpectralEmbedding():
    reduced_data_3d = reducer(data, reduce='SpectralEmbedding', ndims=3)
    assert reduced_data_3d[0].shape==(size,3)


def test_reduce_LocallyLinearEmbedding():
    reduced_data_3d = reducer(data, reduce='LocallyLinearEmbedding', ndims=3)
    assert reduced_data_3d[0].shape==(size,3)


def test_reduce_MDS():
    reduced_data_3d = reducer(data, reduce='MDS', ndims=3)
    assert reduced_data_3d[0].shape==(size,3)


def test_reduce_UMAP():
    reduced_data_3d = reducer(data, reduce='UMAP', ndims=3)
    assert reduced_data_3d[0].shape==(size,3)


def test_reduce_params_UMAP():
    from umap.umap_ import UMAP
    data1 = np.random.rand(20, 10)
    params = {'n_neighbors': 5, 'n_components': 2, 'metric': 'correlation', 'random_state': 1234}
    # testing override of n_dims by n_components. Should raise UserWarning due to conflict
    hyp_data = reducer(data1, reduce={'model': 'UMAP', 'params': params}, ndims=3)
    umap_data = UMAP(**params).fit_transform(data1)
    # Asserting that an AssertionError is raised when comparing the arrays (n_components and ndims not the same)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(hyp_data, umap_data)
