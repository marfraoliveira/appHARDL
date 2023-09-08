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

#%%
@app.route('/', methods=['GET'])
def result():
    return('Teste do Protocolo GET')

@app.route('/predict', methods=['POST'])
def predict():
    # opening and store file in a variable
    #try:
        #Loading CNN model
        json_file = open('model.json','r')
        load=json_file.read()
        json_file.close()
        load_model =model_from_json(load)
        load_model.load_weights('./model.h5')
        load_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        print('Modelo carregdo e compilado com sucesso!!!')
      
      
       #Call classify function to predict the image class using the loaded CNN model

    
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
           return jsonify({'placement':'Descendo Escadas'})
        if class_predict.max() == 1:
           return jsonify({'placement':str('lying')})
        if class_predict.max() == 2:
           return jsonify({'Sentado':''})
        if class_predict.max() == 3:
           return jsonify({'placement':'Descendo Escadas'})
        if class_predict.max() == 4:
           return jsonify({'placement':'Em Pé'})
        
        else:
           return jsonify({'placement':str(class_predict)})

app.run(debug=False,host='0.0.0.0',port=3000)



import numpy as np

# Exemplo de previsões para 3 classes (A, B, C)
previsoes = np.array([
    [0.2, 0.5, 0.3],  # Previsões para a primeira entrada
    [0.8, 0.1, 0.1],  # Previsões para a segunda entrada
    [0.3, 0.4, 0.3]   # Previsões para a terceira entrada
])



# Obtendo as classes previstas usando np.argmax
classes_previstas = np.argmax(previsoes, axis=0)

print("Classes Previstas:", classes_previstas)


