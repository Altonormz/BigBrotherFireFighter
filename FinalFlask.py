from io import BytesIO

from flask import Flask, request, jsonify
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import preprocess_input
from PIL import Image
import requests


app = Flask(__name__)

model = load_model('/Users/danlellouche/Downloads/my_model.h5')

# Dictionnaire des labels
labels_dict = {0: 'default', 1: 'fire', 2: 'smoke'}

@app.route('/', methods=['GET'])
def home():
    """
    Home endpoint of the Churn Prediction API.
    """
    return "Welcome to the Hackathon Fire Detection API! Use the /predict_fire?image_url= route to make predictions."

@app.route('/predict_fire', methods=['GET'])
def predict():
    image_url = request.args.get('image_url')

    # Téléchargez l'image à partir de l'URL
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)


    pred_vec = model.predict(img_array)
    pred_classes = np.argsort(pred_vec)[0][::-1]  # Obtenez les indices triés en ordre décroissant
    predictions = [{'label': labels_dict[class_index], 'confidence': round(float(pred_vec[0][class_index]), 3)} for class_index in pred_classes]

    return jsonify(predictions)

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=8080, debug=True)
