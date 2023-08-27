from distutils.log import debug
from fileinput import filename
#from flask import *  
from flask import Flask,request,jsonify  
import numpy as np
import pandas as pd
import keras.models
from keras.models import model_from_json
import json
from json import JSONEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

@app.route('/api', methods=['POST'])
def api_endpoint():
    # Obtém os dados JSON da solicitação
    data = request.json  
    #Loading CNN model
    json_file = open('model.json','r')
    load=json_file.read()
    json_file.close()
    load_model =model_from_json(load)
    load_model.load_weights('./model.h5')
    load_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    print('Modelo carregdo e compilado com sucesso!!!')
    #pré-processamento
    columns = ['x','y','z']
    df = pd.DataFrame(request.get_json())
    df = pd.DataFrame(df, columns = columns)
    df['x'] = df['x'].astype('float')
    df['y'] = df['y'].astype('float')
    df['z'] = df['z'].astype('float')
      
    #print(df.isnull())
    data = df.to_numpy()
    data[0].ndim
   
    data = data.reshape(-1,80,3)
    class_predict = np.argmax(load_model.predict(data),axis=0)
    
     
    if class_predict.max() == 0:
        return jsonify({'placement':str('Walking')})
    if class_predict.max() == 1:
        return jsonify({'placement':str('Jogging')})
    if class_predict.max() == 2:
       return jsonify({'placement':str('Upstairs')})
    if class_predict.max() == 3:
       return jsonify({'placement':str('Downstairs')})
    if class_predict.max() == 4:
       return jsonify({'placement':str('Sitting')})
        
    else:
       return jsonify({'placement':str('Standing')})
    
    
    # Realiza algum processamento com os dados e prepara uma resposta JSON
    response_data = {
        'message': 'Dados recebidos com sucesso',
        'received_data': data
    }
    
    return jsonify(response_data)
    

app.run(debug=False,host='0.0.0.0',port=3000)

