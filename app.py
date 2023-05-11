from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from keras.models import load_model
import tensorflow as tf
import numpy as np
from flask import Flask,request,jsonify


app = Flask(__name__)

#tf.keras.models.load_model('model.h5')
# load json and create model
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()

loaded_model = tf.keras.models.load_model('model.h5')


@app.route('/')
def index():
    return "Eficiência de Energia em dispositivos móveis"


def load_file(filename):
    file = read_text_file(filename)
    data = np.loadtxt(file)
    return data


def read_text_file(file_path):
    file = np.loadtxt(file_path)
    file = file.reshape(2,80,3,1)
    return file

#teste = read_text_file('vetor.txt')


@app.route('/predict',methods=['POST'])
def predict():
    file = request.files['file']
    classes_x = read_text_file(file)
    class_prediction = loaded_model.predict(classes_x) 
    class_prediction = loaded_model.predict(teste) 
    result=np.argmax(class_prediction,axis=1)
    if result.max() == 0:
       return jsonify({'placement':str('Andando')})
    if result.max() == 1:
       return jsonify({'placement':str('Correndo')})
    if result.max() == 2:
       return jsonify({'placement':str('Subindo Escadas')})
    if result.max() == 3:
       return jsonify({'placement':str('Descendo Escadas')})
    if result.max() == 4:
       return jsonify({'placement':str('Em Pé')})
    if result.max() == 5:
       return jsonify({'placement':str('Deitado')})
   

if __name__ == '__main__':
    app.run(debug=True)

