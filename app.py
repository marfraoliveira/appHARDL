from flask import Flask,request,jsonify,make_response
import numpy as np
import pandas as pd
import keras.models
from keras.models import model_from_json
from itens import identificacao,movimento
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
    return jsonify(identificacao)

@app.route('/predict',methods=['POST'])
def predict():
    file = request.files['file']
    load_file = np.loadtxt(file,delimiter=',')
    dataShaped = read_text_file(load_file)
    print(dataShaped)
    class_prediction = np.argmax(load_model.predict(dataShaped),axis=1)
    class_prediction = {"campo":class_prediction}
    encodedNumpyData = json.dumps(class_prediction, cls=NumpyArrayEncoder)
    #return(encodedNumpyData)
    return('encodedNumpyData')

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
    app.run(debug=True,port=8000)
    
    
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
vetor = np.random.rand(2400000)
#file = vetor.reshape(2,80,3,1)
np.savetxt('vetor20.txt', vetor)


'''