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

def read_text_file(file):
     request.get_json()
     file = np.asarray(file)
     file = file.reshape(1,80,3,1)
     return file



class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

app = Flask(__name__)

#%%

teste = None

import scipy.stats as stats
Fs = 20
frame_size = Fs*4 # 80
hop_size = Fs*2 # 40

def get_frames(df, frame_size, hop_size):
    
    N_FEATURES = 3

    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        x = df['x'].values[i: i + frame_size]
        y = df['y'].values[i: i + frame_size]
        z = df['z'].values[i: i + frame_size]
        
        # Retrieve the most often used label in this segment
       # label = stats.mode(df['label'][i: i + frame_size])[0][0]
       # frames.append([x, y, z])
       # labels.append(label)

        # Bring the segments into a better shape
        frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
        labels = np.asarray(labels)

        return frames, labels




#%%  Inicio Balanceamento de dados

#%%  Inicio Standardized data

## Fim Standardized data


#%%

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
      
        #Apply same preprocessing used while training CNN model
        '''    
            #image_small = st.resize(image, (32,32,3))
            #x = np.expand_dims(image_small.transpose(2, 0, 1), axis=0)
        '''    
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
           print(class_predict)
           return jsonify({'placement':str(class_predict)})
		#response = np.array_str(np.argmax(out,axis=1))
		#return response	
        
        #y_pred = np.argmax(model.predict(X_test),axis=1)

        
    #except Exception as e:
        #Store error to pass to the web page

       
    #    message = "Error encountered. Try another image. ErrorClass: {}, Argument: {} and Traceback details are: {}".format(e.__class__,e.args,e.__doc__)
     #   final = pd.DataFrame({'A': ['Error'], 'B': [0]})
        
        
    
        
    
   # y = df[df['y']=='y'].head(500).copy()
   # z = df[df['z']=='z'].head(500).copy()

    #balanced_data = pd.DataFrame()
    #balanced_data = pd.concat([x, y, z])
    #balanced_data.shape
        
    
    
    #data.isnull().sum()
    #data = df.to_numpy()
    
    #data = data.reshape((6,80,3))
    

    

    

   
'''
    
    class_predict = np.argmax(load_model.predict(data),axis=1)
    print(class_prediction)
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
    
  '''  
      
    
    
    
    
   

#%%





#@app.route('/',methods=['GET'])
#def index_view():
#    return jsonify('resultado')

@app.route('/predict',methods=['POST'])
def predict():
    df = pd.DataFrame(request.get_json())
    print(df)
    '''
    f = request.files['file']
    print('Arquivo',f.filename,'salvo com sucesso !!!')
    f.save(f.filename)
    file = open(f.filename)
    lines = file.readlines()
    data_shaped = read_text_file(lines)
    data_numpy = np.asarray(data_shaped)
    data_numpy = np.asarray(data_numpy, dtype = float)
    class_predict = np.argmax(load_model.predict(data_numpy),axis=1)
    print(class_prediction)
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
 '''
@app.route('/teste',methods=['POST'])
def teste():
    f = request.files['file']
    f.save(f.filename)
    request.files['file']
    print('Arquivo',f.filename,'salvo com sucesso !!!')
    return 'testes'    


app.run(debug=False,host='0.0.0.0',port=3000)

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


