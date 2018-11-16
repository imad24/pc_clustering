import logging
import os

import click
import numpy as np
import pandas as pd


import settings
from data.preprocessing import load_file, save_file, filter_by_season, get_scaled_series
from features import tools

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


@click.command()
@click.option('--version', type=int)
def main(version):
    """ Train an estimator to predict std of product quantities
    """
    logger = settings.get_logger(__name__)
    logger.info("*** Train the estimator model ***")

    try:
        version = 99
        # Load files
        logger.info("Loading data file...")
        clean_df = load_file("p2_clean", type_="P", version=1).set_index(row_headers)
        # cleaned features
        df = load_file('clf_features', type_="P", index="Product")
        assert df is not None

        df.Ldate = df.Ldate.apply(lambda x: str(x))
        # df = df[df.Nstore != 0]

        # get the list of features
        logger.info("Preparing data...")
        _, numeric, _ = tools.get_features_by_type(df)
        data = df.join(clean_df, how="inner")
        features = df.columns

        # prepare data
        logger.info("Encoding data...")
        X_data = data[features]

        series = data.drop(features, axis=1)
        y_data = np.array(series.std(axis=1)).T

        X = tools.encode(X_data, non_categorical=numeric)
        y = y_data

        logger.info("Training regressor...")
        regressor = RandomForestRegressor(n_estimators=100, max_depth=12)
        regressor.fit(X, y)

        filename = "regressor_std_v%d.pkl" % version
        logger.info("Saving regressor model to << %s >>..." % filename)
        path = os.path.join(settings.models_path, filename)
        joblib.dump(regressor, path)

    except Exception as err:
        print(err)


if __name__ == '__main__':
    row_headers = settings.row_headers
    # pylint: disable=no-value-for-parameter
    main()
