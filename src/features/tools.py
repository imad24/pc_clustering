import numpy as np
import pandas as pd
import math
import statsmodels.api as sm
from scipy.stats import chisquare

def get_features_by_type(df):
    """Returns the the list of features, numeric features and categorical features separately
    
    Arguments:
        df {Dataframe} -- the pandas dataframe used for an estimator
    
    Returns:
        tuple of lists -- the complete features list, numeric features, categorical features
    """

    features_list = list(sorted(df.columns))
    numeric = (list(df.columns.to_series().groupby(df.dtypes).groups[np.dtype('float64')]))
    categorical = [f for f in features_list if f not in numeric]
    
    return features_list, numeric, categorical


def get_significance(dist,f_exp=None):
    # ddof = (dist.shape[0]-1)
    chisq , p = chisquare(dist,f_exp=f_exp,ddof=1)
    vc = v_cramer(chisq,n = np.sum(dist), k= len(dist))

    return chisq, p, vc

def v_cramer(chisq,n,k,r=2):
    if k==1: return
    return math.sqrt((chisq/n)  / min(k-1,r-1) )
