from flask import Flask, jsonify
from flask_api import status
import random

from data import import_data

app = Flask(__name__)

@app.route("/api/train")
def train_model():
    if (import_data.import_data()): 
        return jsonify("model training"), status.HTTP_200_OK
    else:
        return "The files are not ready to launch the training", status.HTTP_204_NO_CONTENT



if __name__ == "__main__":
	app.run(debug=True)