import logging
import os

import click
import numpy as np
import pandas as pd

import settings
from data.preprocessing import load_file, save_file, filter_by_season, get_scaled_series, encode
from features import tools
from sklearn.externals import joblib

features_df = load_file("clf_features",type_="P",index = "Product")
features_df.Ldate = features_df.Ldate.apply(lambda x:str(x))

encoded_df = encode(features_df,non_categorical = ["Tprice","Nstore"])
predictor = joblib.load(settings.models_path+'regressor_std.pkl')

row_headers = settings.row_headers

@click.command()
@click.option('--version', type=int)
def main():

    
    return


def predict_std(x_index):
    x = encoded_df.loc[[x_index]]
    y_pred = predictor.predict(x)

    return y_pred

if __name__ == '__main__':
    row_headers = settings.row_headers
    # pylint: disable=no-value-for-parameter
    main()