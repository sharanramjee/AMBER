from __future__ import print_function
import keras
import numpy as np
from keras.datasets import reuters
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Dropout, Activation
from skfeature.function.similarity_based import SPEC as fisher_score

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

# compute fisher scores
score = fisher_score.spec(x_train)
idx = fisher_score.feature_ranking(score)
np.save('features/fisher.npy', idx)
print('Features saved')
#idx = np.load('features/fisher.npy')

# create copies of the data
x_train_copy = x_train
y_train_copy = y_train
x_test_copy = x_test
y_test_copy = y_test

# train and compute accuracy of final model trained on selected features
acc_list = []
for img_rows in range(1000, 0, -10):
	# load the copies of the original data
	x_train = x_train_copy
	y_train = y_train_copy
	x_test = x_test_copy
	y_test = y_test_copy

	x_train = x_train[:, idx[0:img_rows]]
	x_test = x_test[:, idx[0:img_rows]]

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
