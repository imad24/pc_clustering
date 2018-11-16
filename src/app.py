from flask import Flask, jsonify
import random

from data import import_data

app = Flask(__name__)

@app.route("/api/train")
def train_model():
    import_data.import_data() 
    return jsonify("model training"), 404




if __name__ == "__main__":
	app.run(debug=True)