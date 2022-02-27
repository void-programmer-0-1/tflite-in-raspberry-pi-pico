
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

model = tf.keras.models.load_model("linearRegression.h5")

prediction = model.predict(X_train)

plt.scatter(X_train, y_train, label='Generated data')
plt.plot(X_train, prediction, label='Predicted with model', color='red')
plt.legend()
plt.show()

