REM import the data 
python "data/import_data.py"

REM make the ready to train dataset
python "data/make_dataset.py" 1

REM further feature engineering
python "features/build_features.py"

REM train models
python "models/train_model_clustering.py" Autumn --version=99
python "models/train_model_classifier.py" Autumn --version=99
python "models/train_model_estimator.py" --version=99