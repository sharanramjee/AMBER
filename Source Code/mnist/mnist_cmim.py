from __future__ import print_function
import math
import keras
import numpy as np
import pandas as pd
from keras import backend as K
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model, Model
from skfeature.function.information_theoretical_based import CMIM
from keras.layers import Dense, Dropout, Activation, Input, Flatten, Conv2D, MaxPooling2D

# final model parameters
batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 784, 1

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

# compute cmim feature scores
idx,_,_ = CMIM.cmim(x_train, y_train, n_selected_features=784)
np.save('features/cmim.npy', idx)
print('Features saved')
#idx = np.load('features/cmim.npy', idx)

# preprocess the data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# create copies of the data
x_train_copy = x_train
y_train_copy = y_train
x_test_copy = x_test
y_test_copy = y_test

# train and compute accuracy of final model trained on selected features
acc_list = []
for img_rows in range(784, 0, -7):
	# load the copies of the original data
	x_train = x_train_copy
	y_train = y_train_copy
	x_test = x_test_copy
	y_test = y_test_copy

	# load the selected features
	x_train = x_train[:, idx[0:img_rows]]
	x_test = x_test[:, idx[0:img_rows]]

	# final model
	model = Sequential()
	model.add(Dense(512, activation='relu', input_shape=(img_rows,)))
	model.add(Dropout(0.2))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(num_classes, activation='softmax'))
	model.summary()
	model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
	history = model.fit(x_train, y_train, batch_size=256, epochs=30, verbose=1, validation_data=(x_test, y_test), callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')])
	score = model.evaluate(x_test, y_test, verbose=0)
	print(img_rows)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	acc_list.append(score[1]*100)

# print final model accuracies for each feature count
for acc_value in acc_list:
	print(acc_value)