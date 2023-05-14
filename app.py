from flask import Flask, render_template, request,jsonify
import numpy as np
import keras.models
import re
import sys 
import os
import base64
global graph, model

from keras.models import model_from_json

# opening and store file in a variable

json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded Model from disk")
loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


def read_text_file(file_path):
    #request.get_json()
    file = np.loadtxt(file_path)
    file = file.reshape(2,80,3,1)
    return file


app = Flask(__name__)


@app.route('/')
def index_view():
    return ('index.html')

@app.route('/predict',methods=['POST'])
def predict():
     x_axis = request.form.get('dado')
     return jsonify({'placement':x_axis})




if __name__ == '__main__':
    app.run()
    
    
    '''
    @app.route('/predict',methods=['POST'])
    def predict():
         file = request.files['file']
         classes_x = read_text_file(file)
         class_prediction = np.argmax(loaded_model.predict(classes_x),axis=1)
     result = class_prediction
     if result.max() == 0:
        return jsonify({'placement':('Andando')})
     if result.max() == 1:
        return jsonify({'placement':'Correndo'})
     if result.max() == 2:
        return jsonify({'placement':'Subindo Escadas'})
     if result.max() == 3:
        return jsonify({'placement':'Descendo Escadas'})
     if result.max() == 4:
        return jsonify({'placement':'Em Pé'})
     if result.max() == 5:

       '''

