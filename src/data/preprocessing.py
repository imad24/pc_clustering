import itertools
import logging
import math
from datetime import datetime

import click
import numpy as np
import pandas as pd
from scipy.stats import mstats
from sklearn.preprocessing import (LabelBinarizer, LabelEncoder, MinMaxScaler,
                                   OneHotEncoder, StandardScaler, RobustScaler)

import settings


def filter_by_season(df, season):
    seasons_df = load_file("product_season")[["Key_lvl2", "Sales Season"]].drop_duplicates().set_index("Key_lvl2")
    sdf = df.join(seasons_df)
    season_index = sdf[sdf["Sales Season"] == season].index.tolist()
    return df.loc[season_index]


def get_scaled_series(data):
    """Returns a standard scaled dataframe
    
    Arguments:
        data {Dataframe} -- Pandas dataframe
    
    Returns:
        Dataframe -- Scaled Dataframe
        StandardScaler -- the standard scaler used
    """
    std_scaler = StandardScaler(with_mean=True, with_std=True)
    df = pd.DataFrame(std_scaler.fit_transform(data.T).T, index=data.index, columns=data.columns)
    return df, std_scaler



def trim_series(data):
    """Trims (removes complete zeros from each side) the series along the dataset
    
    Arguments:
        data {Pandas Dataframe} -- a dataframe with only the series values
    
    Returns:
        dataframe -- returns a trimmed dataframe
    """

    tail = 0
    head = 0
    # drop first column if zeros
    while (data.iloc[:, 0] == 0).all():
        data.drop(data.columns[0], axis=1, inplace=True)
        head += 1
    # drop last columns if zeros
    while (data.iloc[:, -1] == 0).all():
        data.drop(data.columns[-1], axis=1, inplace=True)
        tail += 1
    return data


def range_from_origin(data, range_, offset=2):
    """Shifts the timeseries values to origin ie makes the first non zero value as the first one and counts "range_" values ahead
    
    Arguments:
        data {Dataframe} -- Pandas datafrale
        range_ {int} -- number of values to take into account
        offset {int} -- offset of zero values in the begining of the series {default:2}
    
    Raises:
        error -- prints the index of the timeseries raising the error
    
    Returns:
        Dataframe -- returns a dataframe with the shifted data
    """

    N = data.shape[0]
    centered = np.zeros((N, range_))
    i = 0
    for index, row in data.iterrows():
        try:
            f = row.nonzero()[0][0]
            r = np.resize(row[f:f + range_].values, ((1, range_)))
            centered[i] = r
            i += 1
        except Exception as error:
            print(index)
            raise error

    centered_df = pd.DataFrame(centered, index=data.index, columns=range(offset, range_ + offset))

    centered_df = centered_df.loc[~(centered_df == 0).all(axis=1)]
    for i in range(offset)[::-1]:
        centered_df.insert(0, i, 0.0)
 
    return centered_df


def remove_tails(data, t=15):
    """remove the timeseries having at least "t" zero values
    
    Arguments:
        data {Dataframe} -- Pandas dataframe
    
    Keyword Arguments:
        t {int} -- the threshold number of zeros to consider to remove a series (default: {15})
    
    Returns:
        Dataframe -- Cleaned timeseries
    """

    mask = (data.iloc[:, -t:] == 0).all(axis=1)
    df = data[~mask]
    print("Series With %d trailing zeros are removed" % t)
    print("Removed: %d , Remaining: %s" % (mask.astype(int).sum(), data.shape[0]))
    return df
    

def remove_heads(data, t=15):
    mask = (data.iloc[:, :t] == 0).all(axis=1)
    df = data[~mask]
    print("Series With more than %d zeros are removed" % t)
    print("Removed: %d , Remaining: %s" % (mask.astype(int).sum(), data.shape[0]))
    return df


def smooth_series(data, window, method="average"):
    """Apply a moving average or mean with window of size "window"
    
    Arguments:
        data {Dataframe} -- Pandas dataframe
        window {int} -- size of window to apply
    
    Keyword Arguments:
        method {str} -- the method applied to smooth the data (default: {"average"})
    
    Returns:
        Dataframe -- the new dataframe
    """
    if method == "average":
        rolled_df = data.rolling(window=window, axis=1, center=True, win_type=None).mean()
    elif method == "median":
        rolled_df = data.rolling(window=window, axis=1, center=True, win_type=None).median()
    else:
        raise ValueError("Unknow method name")

    return rolled_df.dropna(axis=1)


def winsore_data(data, top=0.05, bottom=0.05):
    """Applies a winsorizing on data
    Winsorizing is to set all outliers to a specified percentile of the data; for example,
    a 90% winsorization would see all data below the 5th percentile set to
    the 5th percentile, and data above the 95th percentile set to the 95th percentile

    
    Arguments:
        data {Dataframe} -- Pandas datagframe
    
    Keyword Arguments:
        top {float} -- upper qunatile to consider (default: {0.05})
        bottom {float} -- lower quantile to consider (default: {0.05})
    
    Returns:
        Dataframe -- Winsorized dataframe
    """

    df = data.apply(mstats.winsorize, limits=(bottom, top), axis=1)
    return df


def remove_rare(data, t=5):
    """Remove the series with less than "t" values
    
    Arguments:
        data {Dataframe} -- Pandas dataframe
    
    Keyword Arguments:
        t {int} -- Minimum number of values to consider (default: {5})
    
    Returns:
        Dataframe -- Cleaned dataframe
    """

    mask = (data.where(data == 0, other=1.).sum(axis=1) <= t)
    return data[~mask]


def data_with_headers(series, data, raw_df):
    """Add headers to data (only timeseries)
    
    Arguments:
        series {Numpy array} -- Numpy 2D array containing only timeseries values
        data {[type]} -- [description]
        raw_df {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    headers = raw_df[settings.row_headers[::-1]].loc[data.index]
    product_df_full = pd.DataFrame(series, columns=data.columns, index=data.index)
    for label, column in headers.iteritems():
        product_df_full.insert(0, label, column)
    return product_df_full


def display_df(data, head=5):
    """Displays shape and dataframe head
    
    Arguments:
        data {Dataframe} -- Pandas dataframe
    
    Keyword Arguments:
        head {int} -- number of rows to display (default: {5})
    """
    from IPython.display import display as dp
    print(data.shape)
    if head > 0:
        dp(data.head(head))
    else:
        dp(data)


def translate_df(df, columns, dic_path=settings.raw_path + "dictionnary.npy"):
    """Translates specified columns in dataframe using a numpy dictionnary
    
    Arguments:
        df {Dataframe} -- Pandas dataframe
        columns {list} -- List of columns to translate
    
    Returns:
        Dataframe -- the dataframe with ONLY translated columns
    """

    try:
        tdf = df.copy()
        dico = np.load(dic_path).item()
        tans = df[columns].applymap(lambda x: dico[x])
        for index, col in tans.iteritems():
            if index in df.columns:
                tdf[index] = col
        return tdf
    except Exception as ex:
        print("Error when translating: ", ex)



def save_file(data, filename, type_="I", version=None, index=False):
    """save a dataframe into a .csv file
    
    Arguments:
        data {Dataframe} -- a Pandas dataframe
        filename {str} -- the file name
    
    Keyword Arguments:
        type_ {str} -- The data folder: (I)nterim, (P)rocessed, (R):Raw or (M)odel (default: {"I"})
        version {int} -- the file version (default: {1})
        index {bool} -- either the include the index or not (default: {False})
    """
    # logger = settings.get_logger(__name__)
    try:
        folder = {
            "R": settings.raw_path,
            "I": settings.interim_path,
            "P": settings.processed_path,
            "M": settings.models_path,
            "RP": settings.reports_path
        }.get(type_, settings.interim_path)

        fullname = "%s_%s_v%d.csv" % (settings.PREFIX, filename, version) if version else "%s_%s.csv" % (settings.PREFIX, filename)
        data.to_csv(folder + fullname, sep=";", encoding="utf-8", index=index)
    except Exception:
        # logger.error(err)
        raise


def load_file(filename, type_="I", version=None, sep=";", ext="csv", index=None, **kwargs):
    """Loads a csv or txt file into a dataframe
    
    Arguments:
        filename {string} -- the filename to load
    
    Keyword Arguments:
        type_ {str} -- The data folder: (I)nterim, (P)rocessed, (R):Raw or (M)odel (default: {"I"})
        version {int} -- The file version specified when saved (default: {1})
        sep {str} -- the separator in the file (default: {";"})
        ext {str} -- the extension of the file (default: {"csv"})
        Index {list} -- the columns to set as index to the dataframe
    
    Returns:
        Dataframe -- returns a pandas dataframe
    """
    # logger = settings.get_logger(__name__)
    try:
        folder = {
            "R": settings.raw_path,
            "I": settings.interim_path,
            "P": settings.processed_path,
            "M": settings.models_path,
            "RP": settings.reports_path
        }.get(type_, settings.interim_path)

        fullname = "%s_%s_v%d.%s" % (settings.PREFIX, filename, version, ext) if version else "%s_%s.%s" % (settings.PREFIX, filename, ext)
        df = pd.read_csv(folder + fullname, sep=";", encoding="utf-8", **kwargs)
        if index is not None:
            df.set_index(index, inplace=True)

        return df
    except Exception:
        # logger.error(err)
        raise




        