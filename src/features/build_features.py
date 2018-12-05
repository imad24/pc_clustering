import click
import pandas as pd
import math
import numpy as np
import re
import itertools
from datetime import datetime,date

from sklearn.preprocessing import OneHotEncoder,LabelBinarizer,LabelEncoder,MinMaxScaler

import logging

from data.preprocessing import load_file,save_file,translate_df
from features.tools import create_encoder
import settings

@click.command()
def main():
    """ Build features for classification and prediction models
    """
    try:
        logger = settings.get_logger(__name__)
        logger.info("*** Build features for classification and prediction models ***")

        logger.info("Load raw sales file...")
        file_name = "p2_raw"
        sales = load_file(file_name, index='Product')

        #product description
        logger.info("Load product description file...")
        file_name = "product_7cerf.txt"
        products = pd.read_csv(settings.raw_path+file_name, sep='\t',encoding="utf8")
        products = products.drop_duplicates(["Key_lvl2","Description"]).set_index(["Key_lvl2"])
        products.index.names = ["Product"]

        sales_desc = products.join(sales,how="inner")[products.columns]
        unbalanced = ["Key_lvl1","Description","Key_lvl7","Product Status"]
        sales_desc.drop(unbalanced,axis=1,inplace=True)

        en_sales_desc = translate_df(sales_desc,columns=["Key_lvl3","Key_lvl4","Key_lvl5","Key_lvl6"])

        keep_features = ["Key_lvl3","Color","Size","Launch Date","Age Group","Sales Season","Tag Price"]
        dataframe  = en_sales_desc[keep_features].copy()

        # add number of clients
        logger.info("Load clients count  file...")
        p2c = load_file("p2c1_count",index="Product").astype(np.float64)
        p2c.columns = ["Nstore"]
        # store_counts = prp.load_file("store_counts",index="Product")

        #add number of clients by week
        logger.info("Load number of clients by week...")
        p2cc = load_file("p1cc",index="Product").iloc[:,:5]
        dataframe = dataframe.join(p2cc,how="left").fillna(0)

        dataframe = dataframe.join(p2c,how="left").fillna(0)
        dataframe["Missing"] = 0

        features_list = ["Color","Size","Ldate","Age Group","Person","Pname","Ptype","Tprice","Currency","Sales Season"]+list(p2cc.columns)
        
        
        raw_df = dataframe[dataframe.Nstore!=0.].copy()
        logger.info("Feature engineering...")
        numeric_features = ["Tprice"] + list(p2cc.columns)#Nstore

        features = extract_features(raw_df, non_categorical =numeric_features)
        features_df = features[features_list]


        filename = "clf_features"
        logger.info("==> Saving features file to <<%s>> ..."%filename)
        save_file(features_df,filename,type_="P",index = True)

        logger.info("Dataset  %s  succefully made !"%(features_df.shape[0]))

        logger.info("Creating encoders...")
        categorical_features = ["Color","Size","Age Group","Ldate","Person","Pname","Ptype","Currency","Sales Season"]
        create_encoder(features_df,le_name="prd_le", ohe_name="prd_ohe", scaler_name="prd_scaler", categorical_features=categorical_features,numeric_features=numeric_features)
        logger.info("Encoders created...")
    except Exception as err:
        logger.error(err)
    

def extract_features(rdf, non_categorical):
    data_frame = rdf.copy()
    data_frame["Person"] = data_frame["Key_lvl3"].map(lambda x: GetInfo(x,0))
    data_frame["Pname"] = data_frame["Key_lvl3"].map(lambda x: GetInfo(x,1))
    data_frame["Ptype"] = data_frame["Key_lvl3"].map(lambda x: GetInfo(x,2))
    data_frame["Price"] = data_frame["Key_lvl3"].map(lambda x: GetInfo(x,3))

    # season = rdf["Sales Season"].min()
    # data_frame["Launch Date"] = data_frame["Launch Date"].map(lambda x: date_to_week(x,season)).astype(str)
    data_frame["Ldate"] = data_frame["Launch Date"].map(lambda x: get_week_number(x)).astype(str)
    data_frame.drop(["Key_lvl3"],axis=1,inplace  = True)

    data_frame["Tprice"] = data_frame["Tag Price"] 
    # data_frame["Age Group"] = data_frame["Age Group"].map(lambda x: _redefine_age(x))

    for num_fearture in non_categorical:
        values = data_frame[num_fearture].values.astype(np.float64).reshape((-1,1))
        scaled = values# MinMaxScaler().fit_transform(values)
        data_frame[num_fearture] = scaled.astype(np.float64)

    data_frame["Currency"] = data_frame["Price"].map(lambda x: re.findall(r'[^\d\.]',x)[0] if (len(re.findall(r'[^\d\.]',x))>0) else "Y")
    #missing values
    data_frame.Person.fillna("Female")

    data_frame.Pname.fillna("One-Piece Pants Inside")

    #reduce colors:
    data_frame.pipe(_reduce_colors)


    # data_frame["Currency"] ="bella ciao"
    return data_frame

def _reduce_size(s):
    return {
        "":""

    }[s]

def _reduce_colors(df):
    df.loc[df.Color.str.contains("Grey"),"Color"]="Grey"
    df.loc[df.Color.str.contains("Gray"),"Color"]="Grey"

    df.loc[df.Color.str.contains("Blue"),"Color"]="Blue"
    df.loc[df.Color.str.contains("Cyan"),"Color"]="Blue"
    df.loc[df.Color.str.contains("Navy"),"Color"]="Blue"

    df.loc[df.Color.str.contains("Red"),"Color"]="Red"

    df.loc[df.Color.str.contains("No Color"),"Color"]="Other"

    df.loc[df.Color.str.contains("Green"),"Color"]="Green"

    df.loc[df.Color.str.contains("Pink"),"Color"]="Pink"
    df.loc[df.Color.str.contains("Purple"),"Color"]="Pink"
    df.loc[df.Color.str.contains("Rose"),"Color"]="Pink"
    df.loc[df.Color.str.contains("Pink"),"Color"]="Pink"

    df.loc[df.Color.str.contains("Brown"),"Color"]="Brown"
    df.loc[df.Color.str.contains("Cameo"),"Color"]="Brown"
    df.loc[df.Color.str.contains("Coffee"),"Color"]="Brown"
    df.loc[df.Color.str.contains("Sheer Beige"),"Color"]="Brown"


    df.loc[df.Color.str.contains("Black"),"Color"]="Black"
    return df

def _discretize_client(nb_client):
    if nb_client>1000: return 7
    if nb_client>500: return 6
    if nb_client>150: return 5
    if nb_client>50: return 4
    if nb_client>10: return 3
    if nb_client>5: return 2
    return 1

def _get_price(s):
    """Get the price from the key_lvl3 using a regex
    
    Arguments:
        s {Key} -- The Key_lvl3

    Returns:
        str -- Returns the price tag + the currrency (etiher $ or (Y)uan)
    """

    try:
        regex = r"^[^\d\$]*(\$?\s?\d{1,3}\.?\d{0,2}\D{0,5}$)"
        matches  = re.findall(regex,s)
        price = matches[0].replace(" ","").upper().replace("RMB","YUAN").replace("YUAN","Y").replace("%","Y").strip()
        return price
    except Exception as ex:
        raise ex
    
def _first_week_of_season(season,year):
    """Returns the first week number of a given season
    
    Arguments:
        season {str} -- the season
        year {int} -- the year to considerate
    
    Returns:
        int -- the week number
    """

    return {
        "Autumn":date(year,9,21).isocalendar()[1],
        "Winter":date(year,12,21).isocalendar()[1],
        "Spring":date(year,3,21).isocalendar()[1],
        "Summer":date(year,6,21).isocalendar()[1]
    }[season]

def date_to_week(d,season):
    try:
        the_date = datetime.strptime(d,"%d/%m/%Y")
        first_week = _first_week_of_season(season,the_date.year)
        if(d=="01/01/1900"): return 1
        week_number = the_date.isocalendar()[1]
        return (week_number - first_week)+1
    except Exception as err:
        print(err)
        return d

def get_week_number(d):
    the_date = datetime.strptime(d,"%d/%m/%Y")
    return the_date.isocalendar()[1]

def GetInfo(key3,order,sep = " -"):
    try:
        splits = key3.split(sep)
        if len(splits)<4:
            if order == 3: res = _get_price(key3).strip()
            if order == 2: res = "Thin" 
            if order == 0: res="Female" 
            if order == 1: res = "One-piece pants inside"
        else:
            if order == 3: res =  _get_price(key3).strip()
            else: res =  splits[order].strip()
        
        return str(_redefine_group(res)).title()
    except Exception:
        print("An error occured (%d,%s)"%(order,key3))
        return None

def _redefine_group(key):
    key = key.title()
    dico = {
        "Boy":"Boys",
        "Pregnant Woman" : "Pregnant",
        "Pregnant Women"  : "Pregnant",
        "Women" : "Female",
        "Male" : "Men"
    }
    return dico[key] if key in dico else key

def _redefine_age(age):
    dico ={
        "4-6":"Young",
        "7-9":"Young",
        "10-15":"Young",
        "18-28":"Adult",
        "29-38":"Adult",
        "39-48":"Senior"
    }
    return dico[age]
    
if __name__ == '__main__':
    main()