from flask import Flask,request,jsonify,make_response
import numpy as np
import pandas as pd
import keras.models
from keras.models import model_from_json
from itens import identificacao,movimento
import json
#from werkzeug.utils import secure_filename
#import re
#import sys 
#import os
#global graph, model
#from werkzeug.utils import secure_filename
#ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)

# opening and store file in a variable

json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded Model from disk")
loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


def read_text_file(file):
    #request.get_json()
    file = np.asarray(file)
    file = file.reshape(2,80,3,1)
    return file



@app.route('/',methods=['GET'])
def index_view():
    return jsonify(identificacao)

@app.route('/predict',methods=['POST'])
def predict():
     file = request.files['file']
     #load_file = np.loadtxt(file,delimiter=',')
     #load_file = load_file.reshape(2,80,3,1)
     print(load_file.ndim)
     class_prediction = np.argmax(loaded_model.predict(load_file),axis=1)
     return('ola')   
'''
     result = class_prediction
     print(class_prediction)
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
        return jsonify({'placement':'Deitado'})
'''

if __name__ == '__main__':
    app.run(debug=True)
    
    
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
        return jsonify({'placement':'Deitado'})
       

#%%
vetor = np.random.rand(480)
file = vetor.reshape(2,80,3,1)
np.savetxt('vetor17.txt', vetor)

'''
