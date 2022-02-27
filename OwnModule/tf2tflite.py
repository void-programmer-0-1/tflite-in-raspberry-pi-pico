import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("./linearRegression.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("./linearRegression.tflite","wb") as f:
    f.write(tflite_model)

