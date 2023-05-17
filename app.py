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
    load_file = np.loadtxt(file,delimiter=',')
    dataShaped = read_text_file(load_file)
    class_prediction = np.argmax(loaded_model.predict(dataShaped),axis=1)
    class_prediction = {"campo":class_prediction}
    encodedNumpyData = json.dumps(class_prediction, cls=NumpyArrayEncoder)
    return(encodedNumpyData)

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
       

#%%
vetor = np.random.rand(480)
file = vetor.reshape(2,80,3,1)
np.savetxt('vetor17.txt', vetor)

'''
