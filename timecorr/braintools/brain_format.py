import warnings

import numpy as np
import pandas as pd
from .brain_helpers import PPCA


def format_data(x, ppca=True):
    """
    Formats data into a list of numpy arrays

    This function is useful to identify rows of your array that contain missing
    data or nans.  The returned indices can be used to remove the rows with
    missing data, or label the missing data points that are interpolated
    using PPCA.

    Parameters
    ----------

    x : numpy array, dataframe, string or (mixed) list
        The data to convert
        
    ppca : bool
        Performs PPCA to fill in missing values (default: True)

    Returns
    ----------
    data : list of numpy arrays
        A list of formatted arrays
    """


    # Ensure input is a list
    if not isinstance(x, list):
        x = [x]

    # Validate and convert input data
    formatted_data = []
    for xi in x:
        if isinstance(xi, np.ndarray):
            formatted_data.append(xi)
        elif isinstance(xi, pd.DataFrame):
            formatted_data.append(xi.values)
        elif isinstance(xi, list):
            formatted_data.append(np.array(xi))
        else:
            raise TypeError("Input must be a numpy array, pandas DataFrame, "
                            "or a list of these types.")

    for i in range(len(formatted_data)):
        if formatted_data[i].ndim == 1:
            formatted_data[i] = formatted_data[i].reshape(-1, 1)

    # Handle missing data with PPCA
    if ppca:
        stacked_data = np.vstack(formatted_data)
        if np.isnan(stacked_data).any():
            warnings.warn('Missing data detected. Performing PPCA to fill missing values.')
            formatted_data = fill_missing(formatted_data)

    return formatted_data


def fill_missing(x):

    # ppca if missing data
    m = PPCA()
    m.fit(data=np.vstack(x))
    x_pca = m.transform()

    # if the whole row is missing, return nans
    all_missing = [idx for idx, a in enumerate(np.vstack(x)) if all([type(b)==np.nan for b in a])]
    if len(all_missing)>0:
        for i in all_missing:
            x_pca[i, :] = np.nan

    # get the original lists back
    if len(x)>1:
        x_split = np.cumsum([i.shape[0] for i in x][:-1])
        return list(np.split(x_pca, x_split, axis=0))
    else:
        return [x_pca]
