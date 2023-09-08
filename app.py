from distutils.log import debug
from fileinput import filename
from flask import *  
from flask import Flask,request,jsonify  
import numpy as np
import pandas as pd
import keras.models
from keras.models import model_from_json
import json
from json import JSONEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)


#%%


# Carregue o modelo uma vez ao iniciar o servidor Flask
model = load_model('./model.h5')

@app.route('/api', methods=['POST'])
def predict():
    try:
        # Obtém o JSON da solicitação
        data = request.get_json()

        # Pré-processamento dos dados
        columns = ['x', 'y', 'z']
        df = pd.DataFrame(data, columns=columns)
        df['x'] = df['x'].astype('float')
        df['y'] = df['y'].astype('float')
        df['z'] = df['z'].astype('float')
        data = df.to_numpy()
        data = data.reshape(-1, 80, 3)

        # Faça uma única previsão com o modelo carregado
        class_predict = np.argmax(model.predict(data), axis=1)

        mapeamento = {0: 'Downstairs', 1: 'Jogging', 2: 'Sitting', 3: 'Standing', 4: 'Upstairs', 5: 'Walking'}
        rotulos = [mapeamento[v] for v in class_predict]

        return jsonify({'args': str(rotulos)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
