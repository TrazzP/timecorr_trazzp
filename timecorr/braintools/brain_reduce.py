#!/usr/bin/env python

import warnings
import numpy as np
from sklearn.decomposition import PCA, FastICA, IncrementalPCA, KernelPCA, FactorAnalysis, TruncatedSVD, SparsePCA, MiniBatchSparsePCA, DictionaryLearning, MiniBatchDictionaryLearning
from sklearn.manifold import TSNE, MDS, SpectralEmbedding, LocallyLinearEmbedding, Isomap
from umap.umap_ import UMAP
from .brain_format import format_data as formatter
from .brain_helpers import memoize

# dictionary of models
models = {
    'PCA': PCA,
    'IncrementalPCA': IncrementalPCA,
    'SparsePCA': SparsePCA,
    'MiniBatchSparsePCA': MiniBatchSparsePCA,
    'KernelPCA': KernelPCA,
    'FastICA': FastICA,
    'FactorAnalysis': FactorAnalysis,
    'TruncatedSVD': TruncatedSVD,
    'DictionaryLearning': DictionaryLearning,
    'MiniBatchDictionaryLearning': MiniBatchDictionaryLearning,
    'TSNE': TSNE,
    'Isomap': Isomap,
    'SpectralEmbedding': SpectralEmbedding,
    'LocallyLinearEmbedding': LocallyLinearEmbedding,
    'MDS': MDS,
    'UMAP': UMAP
}

# main function
@memoize
def reduce(x, reduce='IncrementalPCA', ndims=None, format_data=True):
    """
    Reduces dimensionality of an array, or list of arrays
    """

    # Parse reduce argument
    if isinstance(reduce, (str, np.bytes_)):
        model_name = reduce
        model_params = { 'n_components': ndims }
    elif isinstance(reduce, dict):
        try:
            model_name = reduce['model']
            model_params = reduce['params']
        except KeyError:
            raise ValueError("If passing a dictionary, pass keys 'model' and 'params'.")
    else:
        model_name = reduce

    # Validate model
    try:
        if isinstance(model_name, (str, np.bytes_)):
            model_cls = models[model_name]
        else:
            model_cls = model_name
            getattr(model_cls, 'fit_transform')
            getattr(model_cls, 'n_components')
    except (KeyError, AttributeError):
        raise ValueError('Unsupported reduction model.')

    # Sync n_components and ndims
    if 'n_components' in model_params:
        if ndims is not None and ndims != model_params['n_components']:
            warnings.warn('Unequal dims vs n_components: using ndims.')
            model_params['n_components'] = ndims
    else:
        model_params['n_components'] = ndims

    # Format data
    if format_data:
        x = formatter(x, ppca=True)

    # Early exit if no reduction needed
    if model_params['n_components'] is None or all(i.shape[1] <= model_params['n_components'] for i in x):
        return x

    stacked_x = np.vstack(x)
    if stacked_x.shape[0] <= model_params['n_components']:
        warnings.warn('Rows <= ndims: returning zeros.')
        return [np.zeros((stacked_x.shape[0], model_params['n_components']))]

    # --- Algorithm-specific parameter tweaks ---
    name = model_name.lower()
    # TSNE: allow >3 dims via exact method, speedups
    if name == 'tsne':
        n_comp = model_params.get('n_components')
        if n_comp and n_comp > 3:
            model_params['method'] = 'exact'
        # reduce default iterations for speed
        model_params.setdefault('n_iter', 300)
        model_params.setdefault('init', 'pca')
        model_params.setdefault('learning_rate', 'auto')

    # DictionaryLearning: limit iterations, loosen tolerance
    if name == 'dictionarylearning':
        model_params.setdefault('max_iter', 300)
        model_params.setdefault('tol', 1e-2)

    # SparsePCA: limit iterations, loosen tolerance
    if name == 'sparsepca':
        model_params.setdefault('max_iter', 300)
        model_params.setdefault('tol', 1e-2)

    # initialize and reduce
    model = model_cls(**model_params)
    x_reduced = reduce_list(x, model)
    return x_reduced

# sub functions
def reduce_list(x, model):
    split = np.cumsum([len(xi) for xi in x])[:-1]
    x_r = np.vsplit(model.fit_transform(np.vstack(x)), split)
    if len(x) > 1:
        return [xi for xi in x_r]
    else:
        return [x_r[0]]
