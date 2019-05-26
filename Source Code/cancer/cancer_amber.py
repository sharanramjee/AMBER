# coding: utf-8
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model, Model
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

# create copies of the data
X_train_copy = X_train
y_train_copy = y_train
X_test_copy = X_test
y_test_copy = y_test
x_train_orig = X_train
y_train_orig = y_train
x_test_orig = X_test
y_test_orig = y_test
x_train_final = X_train
y_train_final = y_train
x_test_final = X_test
y_test_final = y_test

# load ranker model for AMBER
ranker_model = load_model('models/ann_cancer.h5')
feat_idx_list = []
final_acc_list = []
final_list = []
# perform AMBER
for img_rows in range(29, 0, -1):
	num_fea = img_rows
	x_train = x_train_orig
	idx_acc_list = []
	for feat_idx in range(30):
		if feat_idx in feat_idx_list:
			continue
		x_train = x_train_orig
		x_train = x_train.transpose()
		new_x_train = np.append(x_train[:feat_idx], np.full((1, x_train.shape[1]), np.mean(x_train[feat_idx])), axis=0)
        	x_train = np.append(new_x_train, x_train[feat_idx+1:], axis=0)		
		x_train = x_train.transpose()
		disc_score = ranker_model.evaluate(x_train, y_train, batch_size=455,verbose=0)
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
	remain_idx = list(set(list(range(30))) - set(feat_idx_list))
	x_train = x_train[remain_idx]
	x_test = x_test[remain_idx]
	x_train = x_train.transpose()
	x_test = x_test.transpose()
	input_dim = Input(shape = (img_rows+1, ))
        encoding_dim = img_rows
	encoded = Dense(encoding_dim, activation = 'relu')(input_dim)
	decoded = Dense(img_rows+1, activation = 'sigmoid')(encoded)
	autoencoder = Model(input = input_dim, output = decoded)
	autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
	autoencoder.fit(x_train, x_train, nb_epoch = 20, batch_size = 455, shuffle = True, validation_data = (x_test, x_test), callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')])
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
		final_value = relevance_val + redundance_val
		final_list.append((auto_acc_list[final_index][0], final_value))

	final_list.sort(key=lambda x: x[1])
	worst_feature = final_list[0][0]
	print(img_rows, 'worst:', worst_feature)
	feat_idx_list.append(worst_feature)
	x_train_orig = x_train_orig.transpose()
	new_x_train_orig = np.append(x_train_orig[:feat_idx_list[-1]], np.full((1, x_train_orig.shape[1]), np.mean(x_train_orig[feat_idx_list[-1]])), axis=0)
        x_train_orig = np.append(new_x_train_orig, x_train_orig[feat_idx_list[-1]+1:], axis=0)	
	x_train_orig = x_train_orig.transpose()

	x_train = x_train_final
	x_test = x_test_final
	x_train = x_train.transpose()
	x_test = x_test.transpose()
	remain_idx = list(set(list(range(30))) - set(feat_idx_list))
	np.save('features/amber_' + str(img_rows) + '.npy', remain_idx)
	
	X_train = x_train[remain_idx]
	X_test = x_test[remain_idx]
	X_train = X_train.transpose()
	X_test = X_test.transpose()

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

