# -*- coding: utf-8 -*-
"""npy_segmentation_approach.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tXrQfcCWuexau_iKBiqHabmqQQjX2uO7
"""

# from google.colab import drive
# drive.mount('/content/drive')

import numpy as np
import os
from tensorflow import keras
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D, BatchNormalization
import pickle

dataset_dir  = 'npy_dataset'

eeg_data = []
labels = []
for file in os.listdir(dataset_dir):
    if file.endswith(".npy"):
        data = np.load(os.path.join(dataset_dir,file), allow_pickle=True)
        eeg_data.append(data)
        labels.append(file.split("-")[0])

print(len(eeg_data))
print(len(labels))

print(labels[5])

encoder = LabelEncoder()
encoder.fit(labels)
y = encoder.transform(labels)
original_labels = encoder.inverse_transform(y)
#labels2 = encoder.inverse_transform(original_labels)
y = encoder.transform(labels)
Y = np_utils.to_categorical(y, 6)
original_labels, y, Y

print(Y[5])

s_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

for i in range(len(eeg_data)):
    
    # minmax_scaler.fit(eeg_data[i])
    s_scaler.fit(eeg_data[i])
    # eeg_data[i] = minmax_scaler.transform(eeg_data[i])
    eeg_data[i] = s_scaler.transform(eeg_data[i])

print(eeg_data)

print(eeg_data[0].shape)

data = np.array(eeg_data)  #convert array to numpy type array

X_train, X_test, y_train, y_test  = train_test_split(data,Y,test_size=0.2
                                                    # ,random_state=42
                                                     ,stratify=Y
                                                     ,shuffle=True
                                                     )

print(len(X_train))
print(len(X_test))

print(len(y_train))
print(len(y_test))

train_X = np.array(X_train)
test_X = np.array(X_test)

train_y = np.array(y_train)
test_y = np.array(y_test)

train_X

print(train_X.shape)

print(test_y)

# model = Sequential()

# model.add(Conv1D(64, (3), input_shape=train_X.shape[1:]))
# model.add(Activation('relu'))

# model.add(Conv1D(64, (2)))
# model.add(Activation('relu'))
# model.add(MaxPooling1D(pool_size=(2)))

# model.add(Conv1D(64, (2)))
# model.add(Activation('relu'))
# model.add(MaxPooling1D(pool_size=(2)))

# model.add(Flatten())

# model.add(Dense(512))

# model.add(Dense(6))
# model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])


# model.add(Conv1D(64, (8), input_shape=train_X.shape[1:]))
# model.add(Activation('relu'))

# model.add(Conv1D(128, (6)))
# model.add(Activation('relu'))

# model.add(Conv1D(128, (6)))
# model.add(Activation('relu'))

# model.add(Conv1D(128, (4)))
# model.add(Activation('relu'))

# model.add(Conv1D(128, (4)))
# model.add(Activation('relu'))

# model.add(Conv1D(64, (4)))
# model.add(Activation('relu'))
# model.add(MaxPooling1D(pool_size=(4)))

# model.add(Conv1D(64, (2)))
# model.add(Activation('relu'))
# model.add(MaxPooling1D(pool_size=(4)))

# model.add(Flatten())

# # model.add(Dense(512))
# # model.add(Dense(256))
# # model.add(Dense(128))
# # model.add(Dense(64))
# # model.add(Dense(32))
# # model.add(Dense(16))

# model.add(Dense(6))
# model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

# model.add(Conv1D(512, (3), input_shape=train_X.shape[1:]))
# model.add(Activation('relu'))

# model.add(Conv1D(256, (2)))
# model.add(Activation('relu'))

# model.add(Conv1D(128, (2)))
# model.add(Activation('relu'))

# model.add(Conv1D(128, (2)))
# model.add(Activation('relu'))

# model.add(Conv1D(64, (2)))
# model.add(Activation('relu'))
# model.add(MaxPooling1D(pool_size=(2)))

# model.add(Conv1D(64, (2)))
# model.add(Activation('relu'))
# model.add(MaxPooling1D(pool_size=(2)))

# model.add(Flatten())

# model.add(Dense(512))
# model.add(Dense(256))
# model.add(Dense(128))

# model.add(Dense(6))
# model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

reg = keras.regularizers.L1L2(l1=0.0, l2=0.01)
model = Sequential()
model.add(Conv1D(64, (8), input_shape=train_X.shape[1:],activity_regularizer=reg))
model.add(Activation('relu'))

model.add(Conv1D(128, (4)))
model.add(Activation('relu'))

model.add(Conv1D(64, (4)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=(4)))

model.add(Flatten())

model.add(Dense(6))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_X, train_y, epochs=100, batch_size=32)

score = model.evaluate(test_X, test_y, batch_size=32)
print(score)

#Predict
# y_prediction = model.predict(test_X)
# y_prediction = np.argmax (y_prediction, axis = 1)
# y_test=np.argmax(test_y, axis=1)

if score[1] >= 0.53:
  pickle.dump(model, open(str(score[1])+"-acc-cnn-model.pkl", 'wb'))

print(model.summary())