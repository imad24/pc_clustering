python "data/import_data.py"
python "data/make_dataset.py" 1
python "features/build_features.py"
python "models/train_model_clustering.py" Autumn --version=99