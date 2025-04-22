from flask import Flask, request, jsonify
from flask_cors import CORS
from preprocessing import Predict_Next_Words,Predict_Next_Word
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import os
from chat import get_response 
import requests

app = Flask(__name__)
CORS(app)


# Load the GrU  model
# Load the model and tokenizer
url2 = 'https://drive.google.com/uc?id=1cr6HNDs-xY78dcCUkXnZvo-KSTiXc6zg'
r = requests.get(url2, allow_redirects=True)
open('Medical_Research.h5', 'wb').write(r.content)
from tensorflow.keras.models import load_model
model2 = load_model('Medical_Research.h5')

url1 = 'https://drive.google.com/uc?id=1DDAuQUM2imMxEjb0Uo45RDh5RAqm0_ue'
rr = requests.get(url1, allow_redirects=True)
open('Next_word_predictionModel.h5', 'wb').write(rr.content)
from tensorflow.keras.models import load_model
model = load_model('Next_word_predictionModel.h5')
# model2 = load_model('Medical_Research.h5')

tokenizer = pickle.load(open('token.pkl', 'rb'))

@app.route("/", methods=['GET'])
def index():
    return "Flask API is running on Vercel ✅"
# Define the API endpoint
@app.route('/predict', methods=['POST'])
def predict():


    # Get the uploaded file from the request
    text=request.form["text"]
    print(text)
 

    predicted_text = Predict_Next_Words(
        model=model,tokenizer=tokenizer,text=text,k=3 )

    response = {'predicted_text': predicted_text}
    print(response)
    return jsonify(response)
  

@app.route('/medical', methods=['POST'])
def medical():


    # Get the uploaded file from the request
    text=request.form["text"]
    print(text)
    # Save the uploaded file to disk
    #file_path = os.path.join(
     #   app.config['UPLOAD_FOLDER'], uploaded_file.filename)
    #uploaded_file.save(file_path)

    predicted_text = Predict_Next_Word(
        model2=model2,tokenizer=tokenizer,text=text,k=3 )

    response = {'predicted_text': predicted_text}
    print(response)
    return jsonify(response)




@app.route('/chatbot', methods=['POST'])
def chatbot():
    text =request.get_json().get("message")

    #TODO: check if text vaild 
    response = get_response(text)
    message={"answer":response}
    return jsonify(message)

# Define the main function
if __name__ == '__main__':
    #app.config['UPLOAD_FOLDER'] = './uploads'
    app.run(debug=True)




