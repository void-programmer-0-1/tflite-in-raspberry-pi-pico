# https://www.machinelearningplus.com/deep-learning/linear-regression-tensorflow/
# https://rodolfoferro.xyz/linear-regression-w-tf/

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as  tf
import numpy as np
import matplotlib.pyplot as plt

def GenerateData(lim):
    X = list(range(1,lim + 1))
    y = [x * 2 for x in X]
    return X,y

lim = 10000
X,y = GenerateData(lim)
X_train,y_train = np.array(X),np.array(y)

model = tf.keras.Sequential(name="linearRegression")
model.add(tf.keras.layers.Dense(1,input_shape=[1]))
model.add(tf.keras.layers.Dense(1))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,loss="mean_squared_error",metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=500)

model.save("linearRegression.h5")

plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


