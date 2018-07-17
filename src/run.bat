python "data/import_data.py"
python "data/make_dataset.py" 1
python "features/build_features.py"
python "models/train_model_clustering.py" Autumn --version=99
python "models/train_model_classifier.py" Autumn --version=99
python "models/train_model_estimator.py" --version=99