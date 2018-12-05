import numpy as np
import pandas as pd
import math
import statsmodels.api as sm
from scipy.stats import chisquare


import settings
import itertools
from sklearn.preprocessing import (LabelBinarizer, LabelEncoder, MinMaxScaler,
                                   OneHotEncoder, StandardScaler, RobustScaler)




def get_encoders(le_name,ohe_name,scaler_name):
    le_encoder = np.load(settings.models_path + le_name + '.npy').item()
    ohe_encoder = np.load(settings.models_path + ohe_name + '.npy').item()
    scaler = np.load(settings.models_path + scaler_name + '.npy').item()

    return le_encoder,ohe_encoder,scaler


def create_encoder(df, le_name = None, ohe_name = None, scaler_name=None, categorical_features=None, numeric_features=None):
    """Creates and stores a categorical encoder of a given dataframe
    
    Arguments:
        df {Dataframe} -- The Pandas Dataframe to encode
    
    Keyword Arguments:
        categorical_features {list} -- The list of categorical features to consider (default: {None})
        numeric_features {list} -- The list of non categorical features to ignore (default: {None})
    
    Returns:
        tuple(dict,dict,OneHotEncoder) -- Return the encoders used in every columns as a dictionnary
    """


    if (categorical_features is None):
        categorical_features = sorted(df.drop(numeric_features,axis=1).columns)
    le_dict = {}
    ohe_dict = {}
    scalers = {}
    for index, col in df[categorical_features].sort_index(axis=1).iteritems():
        if (numeric_features is not None) and (index in numeric_features):
            continue
        if index not in categorical_features:
            continue
        le = LabelEncoder().fit(col)
        le_dict[index] = le
        #TODO: What the hell is cateogries
        ohe = OneHotEncoder(categories="auto").fit(le.transform(col).reshape((-1, 1)))
        ohe_dict[index] = ohe

    labeled_df = df[categorical_features].sort_index(axis=1).apply(lambda x: le_dict[x.name].transform(x))
    ohe_encoder = OneHotEncoder(categories="auto").fit(labeled_df)

    # add numeric features
    if len(numeric_features)==0:
        numeric_features = (list(df.columns.to_series().groupby(df.dtypes).groups[np.dtype('float64')]))
    for f in numeric_features:
        values = df[[f]].values
        scaler = MinMaxScaler().fit(values)
        scalers[f] = scaler


    # if le_name is not None:
    #     np.save(settings.models_path + le_name + '.npy', le_dict)
    # if ohe_name is not None:
    #     np.save(settings.models_path + ohe_name + '.npy', ohe_encoder)
    # if scaler_name is not None:
    #     np.save(settings.models_path + scaler_name + '.npy', scalers)
    
    return labeled_df, le_dict, ohe_encoder, scalers, categorical_features, numeric_features


def model_encode(df,model):
    non_categorical = model.non_categorical
    le_encoder = model.le_encoder
    ohe_encoder = model.ohe_encoder
    scaler = model.scaler
    categorical = model.categorical


    features = [["%s_%s" % (f_name, c) for c in f_encoder.classes_] for f_name, f_encoder in le_encoder.items() if f_name in categorical]
    columns = list(itertools.chain.from_iterable(features))

    labeled_df = df[categorical].sort_index(axis=1).apply(lambda x: le_encoder[x.name].transform(x))
    encoded_df = pd.DataFrame(ohe_encoder.transform(labeled_df).toarray(), columns=columns, index=df.index)

    # add numeric features
    if len(non_categorical)==0:
        non_categorical = (list(df.columns.to_series().groupby(df.dtypes).groups[np.dtype('float64')]))
    for f in non_categorical:
        values = df[[f]].values
        scaled = scaler[f].fit_transform(values)
        encoded_df[f] = scaled

    return encoded_df



def encode(df, non_categorical=[], le_encoder=None, ohe_encoder=None, scaler=None, features_list = None):
    """Encodes a given dataframe into a one hot format using a given encoder
    
    Arguments:
        df {Dataframe} -- Pandas dataframe to encode
    
    Keyword Arguments:
        non_categorical {list} -- list of non categorical features (add them at the end of the returned dataframe) (default: {[]})
        le_encoder {dict} -- a dictionnary of label encoders created previously (default: {None})
        ohe_encoder {OneHotEncoder} --  a OneHotEncoder created previously to encode the data (default: {None})
    
    Returns:
        [Dataframe] -- Returns a one hot encoded dataframe
    """
    if(le_encoder is None):
        le_encoder = np.load(settings.models_path + 'prd_le.npy').item()
        ohe_encoder = np.load(settings.models_path + 'prd_ohe.npy').item()
        scaler = np.load(settings.models_path + 'prd_scaler.npy').item()
    if features_list is None:
        features = [["%s_%s" % (f_name, c) for c in f_encoder.classes_] for f_name, f_encoder in le_encoder.items()]
        columns = list(itertools.chain.from_iterable(features))
        categorical = list(le_encoder.keys())
    else:
        features = [["%s_%s" % (f_name, c) for c in f_encoder.classes_] for f_name, f_encoder in le_encoder.items() if f_name in features_list]
        columns = list(itertools.chain.from_iterable(features))
        categorical = features_list
    labeled_df = df[categorical].sort_index(axis=1).apply(lambda x: le_encoder[x.name].transform(x))
    encoded_df = pd.DataFrame(ohe_encoder.transform(labeled_df).toarray(), columns=columns, index=df.index)

    # add numeric features
    if len(non_categorical)==0:
        non_categorical = (list(df.columns.to_series().groupby(df.dtypes).groups[np.dtype('float64')]))
    for f in non_categorical:
        values = df[[f]].values
        scaled = scaler[f].fit_transform(values)
        encoded_df[f] = scaled
    return encoded_df

def get_features_by_type(df):
    """Returns the the list of features, numeric features and categorical features separately
    
    Arguments:
        df {Dataframe} -- the pandas dataframe used for an estimator
    
    Returns:
        tuple of lists -- the complete features list, numeric features, categorical features
    """

    features_list = list(sorted(df.columns))
    numeric = list(df._get_numeric_data().columns)
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



def MASE(training_series, testing_series, prediction_series,m=1):

    T = training_series.shape[0]
    diff=[]
    for i in range(m, T):
        value = training_series[i] - training_series[i - m]
        diff.append(value)
    d = np.abs(diff).sum()/(T-m)
    errors = np.abs(testing_series - prediction_series )

    return errors.mean()/d
    
