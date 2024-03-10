from flask import Flask, jsonify, request
from chat import tester
app = Flask(__name__)
import tensorflow as tf
import numpy as np

celcius = np.array([10, 10, 30, 50, 40, 12, 33], dtype=float)
farenheit = np.array([20, 20, 60, 100, 80, 24, 66], dtype=float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

historial = modelo.fit(celcius, farenheit, epochs=2000, verbose=False)

def tester(a):
    resultado = modelo.predict(np.array([int(a)], dtype=float))
    resultado_list = resultado.tolist()

    return resultado_list



@app.route("/message", methods=['POST'])
def response():
    data = request.json
    finalMsg = tester(data['content'])
    messageResponse = {
        'role': 'system',
        'content': finalMsg
    }
    return jsonify(messageResponse)

if(__name__ == '__main__'):
    app.run(debug=True)