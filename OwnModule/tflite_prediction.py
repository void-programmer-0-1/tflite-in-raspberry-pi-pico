# https://www.tensorflow.org/lite/guide/inference
# https://www.tensorflow.org/guide/keras/save_and_serialize

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow as tf


interpreter = tf.lite.Interpreter(model_path="./linearRegression.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]["shape"]
input_tensor = np.array([[112]],dtype=np.float32)
interpreter.set_tensor(input_details[0]["index"],input_tensor)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]["index"]).item()
print(output_data)
