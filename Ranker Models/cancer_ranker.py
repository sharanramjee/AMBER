# coding: utf-8

"""
Cancer Ranker Model for AMBER
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('data_cancer.csv')
X = dataset.iloc[:,2:32] # [all rows, col from index 2 to the last one excluding 'Unnamed: 32']
y = dataset.iloc[:,1] # [all rows, col one only which contains the classes of cancer]

from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import timeit
start = timeit.default_timer()

import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential() # Initialising the ANN
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30))
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 1, epochs = 25, callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')])

y_pred = classifier.predict(X_test)
y_pred = [ 1 if y>=0.5 else 0 for y in y_pred ]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#print(cm)
correct = cm[0][0] + cm[1][1]
total = correct + cm[0][1] + cm[1][0]
acc = (correct+0.0) / total
acc *= 100
print(acc)
#classifier.save('models/ann_cancer.h5')

stop = timeit.default_timer()
print('Time:', stop - start)
