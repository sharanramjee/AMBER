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
from keras.datasets import reuters
from keras.utils import np_utils
from keras.regularizers import *
import cPickle, random, sys, keras
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping
os.environ["KERAS_BACKEND"] = "tensorflow"
K.tensorflow_backend._get_available_gpus()
from keras.layers.noise import AlphaDropout
from sklearn.metrics import confusion_matrix
from keras.preprocessing.text import Tokenizer
from keras.optimizers import adam, adagrad, RMSprop
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model, Model
from skfeature.function.sparse_learning_based import RFS
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from skfeature.function.similarity_based.fisher_score import feature_ranking
from skfeature.function.information_theoretical_based.CMIM import cmim as cmim
from skfeature.utility.sparse_learning import construct_label_matrix, feature_ranking
from skfeature.function.similarity_based.fisher_score import fisher_score as fisher_score
from keras.layers import Dense, Dropout, Activation, Input, Flatten, Conv2D, MaxPooling2D

# load wisconsin breast cancer dataset
dataset = pd.read_csv('data_cancer.csv')

# split data and preprocess it
X = dataset.iloc[:,2:32] # [all rows, col from index 2 to the last one excluding 'Unnamed: 32']
y = dataset.iloc[:,1] # [all rows, col one only which contains the classes of cancer]
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train = X_train.values
X_test = X_test.values

# compute CMIM scores
idx,_,_ = cmim(X_train, y_train, n_selected_features=30)
np.save('features/cmim.npy', idx)
print('Features saved')
#idx = np.load('features/cmim.npy')

# create copies of the data
X_train_copy = X_train
y_train_copy = y_train
X_test_copy = X_test
y_test_copy = y_test

# train and compute accuracy of final model trained on selected features
final_list = []
for num_fea in range(30, 0, -1):
	# load the copies of the original data
	X_train = X_train_copy
	y_train = y_train_copy
	X_test = X_test_copy
	y_test = y_test_copy
	
	# load the selected features
	X_train = X_train[:, idx[0:num_fea]]
	X_test = X_test[:, idx[0:num_fea]]
	print('after', X_train.shape)

	# final model
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)
	classifier = Sequential() # Initialising the ANN
	classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = num_fea))
	classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
	classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
	classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
	classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
	classifier.fit(X_train, y_train, batch_size = 1, epochs = 50, callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')])

	y_pred = classifier.predict(X_test)
	y_pred = [ 1 if y>=0.5 else 0 for y in y_pred ]
	cm = confusion_matrix(y_test, y_pred)
	correct = cm[0][0] + cm[1][1]
	total = correct + cm[0][1] + cm[1][0]
	acc = (correct+0.0) / total
	acc *= 100
	print(num_fea, acc)
	final_list.append(acc)

# print final model accuracies for each feature count
for accuracy in final_list:
	print(accuracy)

