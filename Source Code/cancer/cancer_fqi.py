# coding: utf-8
import keras
import numpy as np
import pandas as pd
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model, Model

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

# create copies of the data
X_train_copy = X_train
y_train_copy = y_train
X_test_copy = X_test
y_test_copy = y_test

# load ranker model for FQI
ranker_model = load_model('models/ann_cancer.h5')
# compute FQI scores
idx_acc_list = []
for img_rows in range(0, 30):
	num_fea = img_rows
	x_train = X_train_copy
	x_train = x_train.transpose()
	new_x_train = np.append(x_train[:img_rows], np.zeros((1, x_train.shape[1])), axis=0)
        x_train = np.append(new_x_train, x_train[img_rows+1:], axis=0)		
	x_train = x_train.transpose()
	prediction = ranker_model.predict(x_train, batch_size=455)
	error = (prediction - y_train) ** 2
	error = sum(sum(error))
	idx_acc_list.append((img_rows, error))
idx_acc_list.sort(key=lambda x: x[1], reverse=True)
idx = [value[0] for value in idx_acc_list]
np.save('features/fqi.npy', idx)
print('Features saved')

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

