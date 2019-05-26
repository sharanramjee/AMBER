from __future__ import print_function
import keras
import os,random
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import tensorflow as tf
from keras import layers
from copy import deepcopy
import keras.models as models
from keras import backend as K
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.regularizers import *
import cPickle, random, sys, keras
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping
os.environ["KERAS_BACKEND"] = "tensorflow"
K.tensorflow_backend._get_available_gpus()
from keras.layers.noise import AlphaDropout
from keras.preprocessing.text import Tokenizer
from keras.optimizers import adam, adagrad, RMSprop
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model, Model
from skfeature.function.similarity_based import fisher_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from skfeature.function.information_theoretical_based import CMIM
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from skfeature.function.statistical_based.chi_square import feature_ranking
from skfeature.function.statistical_based.chi_square import chi_square as RFS
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

# load ranker model for FQI
ranker_model = load_model('models/cnn_mnist.h5')

# compute fqi scores
x_train_final = x_train
x_test_final = x_test
idx_acc_list = []
for img_rows in range(0, 784):
	num_fea = img_rows
	x_train = x_train_copy
	x_train = x_train.transpose()
	new_x_train = np.append(x_train[:img_rows], np.zeros((1, x_train.shape[1])), axis=0)
        x_train = np.append(new_x_train, x_train[img_rows+1:], axis=0)		
	x_train = x_train.transpose()
	x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
	prediction = ranker_model.predict(x_train, batch_size=1000)
	error = (prediction - y_train) ** 2
	error = sum(sum(error))
	print(img_rows, error)
	idx_acc_list.append((img_rows, error))
idx_acc_list.sort(key=lambda x: x[1], reverse=True)
idx = [value[0] for value in idx_acc_list]
np.save('features/fqi.npy', idx)

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
