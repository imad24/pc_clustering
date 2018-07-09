# -*- coding: utf-8 -*-
import sys 
import os
import click
# add the 'src' directory as one where we can import modules
root_dir = os.path.join(os.getcwd(),os.pardir,os.pardir)
src_dir = os.path.join(os.getcwd(), os.pardir,os.pardir, 'src')
if src_dir not in sys.path: sys.path.append(src_dir)


from dotenv import find_dotenv, load_dotenv
 
import math
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from scipy.stats import mstats
from sklearn.preprocessing import StandardScaler, MinMaxScaler


load_dotenv(find_dotenv())


# add the 'src' directory as one where we can import modules

# logger = logging.getLogger(__name__)
# logger.info('making final data set from raw data')

subfolder = os.getenv("SUBFOLDER")
PREFIX = os.getenv("PREFIX")
raw_path = os.path.join(root_dir,"data\\raw\\",subfolder)
interim_path = os.path.join(root_dir,"data\\interim\\",subfolder) 
processed_path = os.path.join(root_dir,"data\\processed\\",subfolder) 

reports_path = os.path.join(root_dir,"reports\\",subfolder)
models_path = os.path.join(root_dir,"models\\",subfolder)
row_headers = ["Product"]


@click.command()
def main():
    """ Contains all the functions of data preprocessing
    """

    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()


def load_data(filename):
    """lksdjfdkls
    
    Arguments:
        filename {dj} -- kidsjf
    
    Returns:
        sdkjf -- dsklfj

    """

    
    n_row_headers = len(row_headers)

    product_raw_df = pd.read_csv(interim_path + filename , sep = ";", encoding = 'utf-8', header = 0)

    cols = product_raw_df.columns.values
    cols[:n_row_headers]  = row_headers
    product_raw_df.columns =cols

    return product_raw_df


def trim_series(data):
    """Trims (removes complete zeros from each side) the series along the dataset 
    
    Arguments:
        data {Pandas Dataframe} -- a dataframe with only the series values
    
    Returns:
        dataframe -- returns a trimmed dataframe 
    """

    tail = 0
    head = 0
    #drop first column if zeros
    while (data.iloc[:,0] == 0).all():
        data.drop(data.columns[0], axis=1, inplace=True) 
        head +=1
    #drop last columns if zeros
    while (data.iloc[:,-1] == 0).all():
        data.drop(data.columns[-1], axis=1, inplace=True)
        tail += 1
    return data


def range_from_origin(data,range_,offset=2):
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
    centered = np.zeros((N,range_))
    i=0
    for index,row in data.iterrows():
        try:
            f = row.nonzero()[0][0]
            r = np.resize(row[f:f+range_].values,((1,range_)))
            centered[i] = r
            i+=1
        except Exception as error:
            print(index)
            raise error

    centered_df = pd.DataFrame(centered,index = data.index,columns=range(offset,range_+offset))

    centered_df = centered_df.loc[~(centered_df==0).all(axis = 1)]
    for i in range(offset)[::-1]:
        centered_df.insert(0,i,0.0)

    return centered_df

def remove_tails(data,t = 15):
    """remove the timeseries having at least "t" zero values 
    
    Arguments:
        data {Dataframe} -- Pandas dataframe
    
    Keyword Arguments:
        t {int} -- the threshold number of zeros to consider to remove a series (default: {15})
    
    Returns:
        Dataframe -- Cleaned timeseries
    """

    mask = (data.iloc[:,-t:]==0).all(axis=1)
    df  =  data[~mask]
    print("Series With %d trailing zeros are removed"%t)
    print("Removed: %d , Remaining: %s"%(mask.astype(int).sum(),data.shape[0]))
    return df

def remove_heads(data,t = 15):
    mask = (data.iloc[:,:t] == 0).all(axis=1)
    df  =  data[~mask]
    print("Series With more than %d zeros are removed"%t)
    print("Removed: %d , Remaining: %s"%(mask.astype(int).sum(),data.shape[0]))
    return df


def smooth_series(data,window,method="average"):
    """Apply a moving average or mean with window of size "window"
    
    Arguments:
        data {Dataframe} -- Pandas dataframe
        window {int} -- size of window to apply
    
    Keyword Arguments:
        method {str} -- the method applied to smooth the data (default: {"average"})
    
    Returns:
        Dataframe -- the new dataframe
    """
    if method =="average":
        rolled_df = data.rolling(window=window,axis=1,center = True,win_type=None).mean()
    elif method == "median":
        rolled_df = data.rolling(window=window,axis=1,center = True,win_type=None).median()
    else:
        raise "Unknow method name"

    return rolled_df.dropna(axis = 1)


def winsore_data(data,top=0.05,bottom=0.05):
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

    df = data.apply(mstats.winsorize,limits = (bottom,top),axis=1)
    return df


def remove_rare(data,t = 5):
    """Remove the series with less than "t" values
    
    Arguments:
        data {Dataframe} -- Pandas dataframe
    
    Keyword Arguments:
        t {int} -- Minimum number of values to consider (default: {5})
    
    Returns:
        Dataframe -- Cleaned dataframe
    """

    mask =(data.where(data==0,other=1.).sum(axis=1)<=t)
    return data[~mask]


# def get_scaled_series(data):
#     """Returns a standard scaled dataframe
    
#     Arguments:
#         data {Dataframe} -- Pandas dataframe
    
#     Returns:
#         Dataframe -- Scaled Dataframe
#         StandardScaler -- the standard scaler used
#     """

#     d = data.as_matrix().astype(float)
#     std_scaler = StandardScaler(with_mean=True, with_std=True).fit(d.T)
#     X_z = std_scaler.transform(data.T).T
#     return X_z,std_scaler


def get_scaled_series(data):
    """Returns a standard scaled dataframe
    
    Arguments:
        data {Dataframe} -- Pandas dataframe
    
    Returns:
        Dataframe -- Scaled Dataframe
        StandardScaler -- the standard scaler used
    """
    std_scaler = StandardScaler(with_mean=True, with_std=True)
    df = pd.DataFrame(std_scaler.fit_transform(data),index = data.index,columns=data.columns)
    return df,std_scaler

def data_with_headers(series,data,raw_df):
    """Add headers to data (only timeseries)
    
    Arguments:
        series {Numpy array} -- Numpy 2D array containing only timeseries values
        data {[type]} -- [description]
        raw_df {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    headers = raw_df[row_headers[::-1]].loc[data.index]
    product_df_full = pd.DataFrame(series, columns = data.columns,index=data.index)
    for label ,column in headers.iteritems():
        product_df_full.insert(0,label,column)
    return product_df_full

def display(data,head=5):
    """Displays shape and dataframe head
    
    Arguments:
        data {Dataframe} -- Pandas dataframe
    
    Keyword Arguments:
        head {int} -- number of rows to display (default: {5})
    """
    from IPython.display import display as dp 
    print(data.shape)
    if head>0:
        dp(data.head(head))
    else:
        dp(data)

def translate_df(df,columns,dic_path=raw_path +"dictionnary.npy"):
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
        tans = df[columns].applymap(lambda x:dico[x])
        for index,col in tans.iteritems():
            if index in df.columns: tdf[index] = col
        return tdf
    except Exception as ex:
        print("Error when translating: ",ex)



def save_file(data,filename,type_="I",version = None,index=False):
    """save a dataframe into a .csv file
    
    Arguments:
        data {Dataframe} -- a Pandas dataframe
        filename {str} -- the file name
    
    Keyword Arguments:
        type_ {str} -- The data folder: (I)nterim, (P)rocessed, (R):Raw or (M)odel (default: {"I"})
        version {int} -- the file version (default: {1})
        index {bool} -- either the include the index or not (default: {False})
    """

    folder  = {
        "R" : raw_path,
        "I" : interim_path,
        "P" : processed_path,
        "M" : models_path
    }.get(type_,interim_path)

    fullname = "%s_%s_v%d.csv"%(PREFIX,filename,version) if version else "%s_%s.csv"%(PREFIX,filename)
    data.to_csv(folder+fullname, sep=";", encoding = "utf-8",index = index)


def load_file(filename,type_="I",version=None,sep=";", ext="csv",index =None):
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

    folder  = {
        "R" : raw_path,
        "I" : interim_path,
        "P" : processed_path,
        "M" : models_path
    }.get(type_,interim_path)
    fullname = "%s_%s_v%d.%s"%(PREFIX,filename,version,ext) if version else "%s_%s.%s"%(PREFIX,filename,ext)
    df = pd.read_csv(folder+fullname,sep=";",encoding="utf-8")
    if index is not None: df.set_index(index,inplace=True)

    return df



from sklearn.preprocessing import OneHotEncoder,LabelBinarizer,LabelEncoder
import itertools


def create_encoder(df,categorical_features= None,non_categorical=None):
    """Creates and stores a categorical encoder of a given dataframe
    
    Arguments:
        df {Dataframe} -- The Pandas Dataframe to encode
    
    Keyword Arguments:
        categorical_features {list} -- The list of categorical features to consider (default: {None})
        non_categorical {list} -- The list of non categorical features to ignore (default: {None})
    
    Returns:
        tuple(dict,dict,OneHotEncoder) -- Return the encoders used in every columns as a dictionnary
    """

    if (categorical_features is None):
        categorical_features = df.columns
    le_dict = {}
    ohe_dict = {}
    for index,col in df[categorical_features].sort_index(axis=1).iteritems():
        if (non_categorical is not None) and (index in non_categorical): continue
        if index not in categorical_features: continue
        le = LabelEncoder().fit(col)
        le_dict[index] = le
        ohe = OneHotEncoder().fit(le.transform(col).reshape((-1,1)))
        ohe_dict[index] = ohe     
    
    labeled_df = df[categorical_features].sort_index(axis=1).apply(lambda x: le_dict[x.name].transform(x))
    ohe_encoder  = OneHotEncoder().fit(labeled_df)
    
    np.save(models_path+'le_encoder', le_dict)
    np.save(models_path+'ohe_encoder', ohe_encoder)
    print("encoders saved")
    return labeled_df,le_dict,ohe_encoder

def encode(df,non_categorical=[],le_encoder = None,ohe_encoder=None):
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
        le_encoder = np.load(models_path+'le_encoder.npy').item()
        ohe_encoder = np.load(models_path+'ohe_encoder.npy').item()
    features =[ ["%s_%s"%(f_name,c) for c in f_encoder.classes_] for f_name,f_encoder in le_encoder.items()]
    columns = list(itertools.chain.from_iterable(features))
    categorical = list(le_encoder.keys())
    labeled_df = df[categorical].sort_index(axis = 1).apply(lambda x: le_encoder[x.name].transform(x))
    encoded_df = pd.DataFrame(ohe_encoder.transform(labeled_df).toarray(), columns = columns,index=df.index)

    #add numeric features
    for f in non_categorical:
        encoded_df[f] = df[f]


    return encoded_df