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

@app.route('/', methods=['GET'])
def teste():
    return('Teste do Protocolo GET')
    
@app.route('/json', methods=['POST'])
def json_example():
    # opening and store file in a variable
    #try:
        #Loading CNN model
        json_file = open('model.json','r')
        load=json_file.read()
        json_file.close()
        load_model =model_from_json(load)
        load_model.load_weights('./model.h5')
        print('Modelo carregdo com sucesso!!!')
        load_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
      
      
       #Call classify function to predict the image class using the loaded CNN model

    
        columns = ['x','y','z']
        df = pd.DataFrame(request.get_json())
        df = pd.DataFrame(df, columns = columns)
        df['x'] = df['x'].astype('float')
        df['y'] = df['y'].astype('float')
        df['z'] = df['z'].astype('float')
        
        print(df.isnull())
        data = df.to_numpy()
        data[0].ndim
      
              
        data = data.reshape(-1,80,3)
        class_predict = np.argmax(load_model.predict(data),axis=None)
       
        if class_predict.max() == 0:
           return jsonify({'placement':('Andando')})
           print(class_predict)
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
        else:
           return jsonify({'placement':str('Movimento não reconhecido')})
    
    


app.run(debug=False,host='0.0.0.0',port=3000)
