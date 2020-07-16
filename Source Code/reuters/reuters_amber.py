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
from keras.preprocessing.text import Tokenizer
from keras.optimizers import adam, adagrad, RMSprop
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model, Model
from skfeature.function.sparse_learning_based import RFS
from skfeature.function.similarity_based import fisher_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from skfeature.function.information_theoretical_based import CMIM
from skfeature.function.similarity_based.SPEC import feature_ranking
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from skfeature.function.similarity_based.SPEC import spec as fisher_score
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from skfeature.function.statistical_based.chi_square import feature_ranking
from skfeature.function.statistical_based.chi_square import chi_square as CMIM
from skfeature.utility.sparse_learning import construct_label_matrix, feature_ranking
from keras.layers import Dense, Dropout, Activation, Input, Flatten, Conv2D, MaxPooling2D

# final model parameters
max_words = 1000
batch_size = 32
epochs = 100

# load dataset
print('Loading data...')
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words,
                                                         test_split=0.2)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
num_classes = np.max(y_train) + 1
print(num_classes, 'classes')

# vectorize datset
print('Vectorizing sequence data...')
tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

# create copies of the data
x_train_copy = x_train
y_train_copy = y_train
x_test_copy = x_test
y_test_copy = y_test

x_train_orig = x_train
y_train_orig = y_train
x_test_orig = x_test
y_test_orig = y_test

# load ranker model for FQI
ranker_model = load_model('models/mlp_reuters.h5')

# perform AMBER
R3_value = 0.5
x_train_final = x_train
x_test_final = x_test
feat_idx_list = []
final_acc_list = []
acc_list = []
for img_rows in range(999, 0, -1):
	num_fea = img_rows
	x_train = x_train_orig
	idx_acc_list = []
	for feat_idx in range(1000):
		if feat_idx in feat_idx_list:
			continue
		x_train = x_train_orig
		x_train = x_train.transpose()
		new_x_train = np.append(x_train[:feat_idx], np.full((1, x_train.shape[1]), np.mean(x_train[feat_idx])), axis=0)
		x_train = np.append(new_x_train, x_train[feat_idx+1:], axis=0)
		x_train = x_train.transpose()
		disc_score = ranker_model.evaluate(x_train, y_train, batch_size=10000, verbose=0)
		idx_acc_list.append((feat_idx, disc_score[0]))
		print(feat_idx)
	idx_acc_list.sort(key=lambda x: x[0])
	cost_array = [auto_acc[1] for auto_acc in idx_acc_list]
	cost_array = np.array(cost_array)
	cost_array = cost_array / (max(cost_array) - min(cost_array))
	for auto_index in range(len(idx_acc_list)):
		idx_acc_list[auto_index] = (idx_acc_list[auto_index][0], cost_array[auto_index])
	
	x_train = x_train_final
	x_test = x_test_final
	x_train = x_train.transpose()
	x_test = x_test.transpose()
	remain_idx = list(set(list(range(1000))) - set(feat_idx_list))
	x_train = x_train[remain_idx]
	x_test = x_test[remain_idx]
	x_train = x_train.transpose()
	x_test = x_test.transpose()
	input_dim = Input(shape=(img_rows+1, ))
	encoding_dim = img_rows
	encoded = Dense(encoding_dim, activation='relu')(input_dim)
	decoded = Dense(img_rows+1, activation='sigmoid')(encoded)
	autoencoder = Model(input=input_dim, output=decoded)
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
	autoencoder.fit(x_train, x_train, nb_epoch=20, batch_size=10000, shuffle=True, validation_data=(x_test, x_test), callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')])
	auto_acc_list = []
	x_train_auto = x_train
	for feat_idx in range(img_rows+1):
		x_train = x_train_auto
		x_train = x_train.transpose()
		new_x_train = np.append(x_train[:feat_idx], np.full((1, x_train.shape[1]), np.mean(x_train[feat_idx])), axis=0)
		x_train = np.append(new_x_train, x_train[feat_idx+1:], axis=0)
		x_train = x_train.transpose()
		x_train_encoded = autoencoder.predict(x_train)
		auto_cost = x_train_encoded - x_train_auto
		auto_cost = auto_cost ** 2
		auto_cost = sum(sum(auto_cost))
		auto_acc_list.append((idx_acc_list[feat_idx][0], auto_cost))
		print(feat_idx)
	auto_acc_list.sort(key=lambda x: x[0])
	cost_array = [auto_acc[1] for auto_acc in auto_acc_list]
	cost_array = np.array(cost_array)
	cost_array = cost_array / (max(cost_array) - min(cost_array))
	for auto_index in range(len(auto_acc_list)):
		auto_acc_list[auto_index] = (auto_acc_list[auto_index][0], cost_array[auto_index])
	
	final_list = []
	for final_index in range(len(idx_acc_list)):
		relevance_val = idx_acc_list[final_index][1]
		redundance_val = auto_acc_list[final_index][1]
		final_value = relevance_val + redundance_val / R3_value
		final_list.append((idx_acc_list[final_index][0], final_value))

	final_list.sort(key=lambda x: x[1])
	x_train_orig = x_train_orig.transpose()
	for worst_idx in range(1):
		feat_idx_list.append(final_list[worst_idx][0])
		new_x_train_orig = np.append(x_train_orig[:feat_idx_list[-1]], np.full((1, x_train_orig.shape[1]), np.mean(x_train_orig[feat_idx_list[-1]])), axis=0)
		x_train_orig = np.append(new_x_train_orig, x_train_orig[feat_idx_list[-1]+1:], axis=0)
	x_train_orig = x_train_orig.transpose()

	remain_idx = list(set(list(range(1000))) - set(feat_idx_list))
	if img_rows % 10 == 0:
		np.save('features/amber_' + str(img_rows) + '.npy', remain_idx)

	# train final model
	x_train = x_train_copy[:, remain_idx]
	x_test = x_test_copy[:, remain_idx]

	print('Building model...')
	model = Sequential()
	model.add(Dense(512, input_shape=(img_rows,)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',
								optimizer='adam',
								metrics=['accuracy'])

	history = model.fit(x_train, y_train,
											batch_size=batch_size,
											epochs=epochs,
											verbose=1,
											validation_split=0.1,
											callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')])
	score = model.evaluate(x_test, y_test,
												 batch_size=batch_size, verbose=1)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])
	acc_list.append(score[1]*100)

# print final model accuracies for each feature count
for acc_value in acc_list:
	print(acc_value)
