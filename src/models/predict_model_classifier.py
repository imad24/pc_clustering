import settings
from features import tools
from sklearn.externals import joblib
import models.classifierModel


# @click.command()
# @click.option('--version', type=int)
def main():
    return classifier


def load_model(name):
    return joblib.load(settings.models_path + name + '.pkl')

def predict(X):
    y_pred = classifier.model.predict(X)
    y_pred_proba = classifier.model.predict_proba(X)

    return y_pred,y_pred_proba

def preprocess(X_data,y_data):
    X = tools.model_encode(X_data, model=classifier)
    y = y_data

    return X,y


def train_model(X,y):
    classifier.model.fit(X,y)

classifier = load_model("classifier_Autumn_v1")


if __name__ == '__main__':
    row_headers = settings.row_headers
    # pylint: disable=no-value-for-parameter
    main()

