from distutils.log import debug
from fileinput import filename
from flask import *  
import numpy as np
import pandas as pd
import keras.models
from keras.models import model_from_json
import json
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

app = Flask(__name__)

# opening and store file in a variable

json_file = open('model.json','r')

load=json_file.read()

json_file.close()

load_model =model_from_json(load)
load_model.load_weights('model.h5')
load_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print('Modelo carregdo com sucesso!!!')


def read_text_file(file):
    #request.get_json()
    file = np.asarray(file)
    file = file.reshape(1,80,3,1)
    return file



@app.route('/',methods=['GET'])
def index_view():
    return jsonify('resultado')

@app.route('/predict',methods=['POST'])
def predict():
    f = request.files['file']
    print('Arquivo',f.filename,'salvo com sucesso !!!')
    f.save(f.filename)
    file = open(f.filename)
    lines = file.readlines()
    data_shaped = read_text_file(lines)
    data_numpy = np.asarray(data_shaped)
    data_numpy = np.asarray(data_numpy, dtype = float)
    print(data_numpy)
    print(data_numpy.ndim)
    class_predict = np.argmax(load_model.predict(data_numpy),axis=1)
    #print(class_prediction)
    print(class_predict)
    if class_predict.max() == 0:
       return jsonify({'placement':('Andando')})
    if class_predict.max() == 1:
       return jsonify({'placement':'Correndo'})
    if class_predict.max() == 2:
       return jsonify({'placement':'Subindo Escadas'})
    if class_predict.max() == 3:
       return jsonify({'placement':'Descendo Escadas'})
    if class_predict.max() == 4:
       return jsonify({'placement':'Em Pé'})
    if class_predict.max() == 5:
       return jsonify({'placement':'Deitado'})
   





app.run(host='0.0.0.0',port=3000)

'''
    load_file = np.loadtxt(file,delimiter=',')
    dataShaped = read_text_file(load_file)
    class_prediction = np.argmax(load_model.predict(dataShaped),axis=1)
    class_prediction = {"campo":class_prediction}
    encodedNumpyData = json.dumps(class_prediction, cls=NumpyArrayEncoder)
    resultado.append(encodedNumpyData)
    return resultado
'''

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
'''       

#%%
'''
vetor = np.random.rand(240)
#file = vetor.reshape(1,80,3,1)
np.savetxt('vetor22.txt', vetor)
'''


