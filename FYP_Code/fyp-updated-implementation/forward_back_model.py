# -*- coding: utf-8 -*-

# from google.colab import drive
# drive.mount('/content/drive')

import numpy as np
from tensorflow import keras
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
# import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D
import pickle
import os
from sklearn.model_selection import train_test_split

parent_folder = "./personal_dataset_250"

# Get the list of sub-folders
sub_folders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]

# Read the .npy files from each sub-folder and store in arrays
data = []
labels = []
for sub_folder in sub_folders:
    sub_folder_path = os.path.join(parent_folder, sub_folder)
    for npy_file in os.listdir(sub_folder_path):
        if npy_file.endswith(".npy"):
            npy_file_path = os.path.join(sub_folder_path, npy_file)
            data.append(np.load(npy_file_path,allow_pickle=True))
            labels.append(sub_folder)


encoder = LabelEncoder()
encoder.fit(labels)
y = encoder.transform(labels)
original_labels = encoder.inverse_transform(y)
#labels2 = encoder.inverse_transform(original_labels)
y = encoder.transform(labels)
Y = np_utils.to_categorical(y, 3)
original_labels, y, Y

#print(Y[5])

# Initialize a StandardScaler object
scaler = StandardScaler()

for i in range(len(data)):
    scaler = StandardScaler()
    scaler.fit(data[i])
    data[i] = scaler.transform(data[i])

#print(data)

#print(data[0].shape)

data = np.array(data)  #convert array to numpy type array

X_train, X_test, y_train, y_test  = train_test_split(data,Y,test_size=0.2
                                                    # ,random_state=42
                                                     ,stratify=Y
                                                     ,shuffle=True)

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

model = Sequential()
model.add(Conv1D(64, (8), input_shape=train_X.shape[1:], padding='same'))
model.add(Activation('relu'))

model.add(Conv1D(128, (4), padding='same'))
model.add(Activation('relu'))

model.add(Conv1D(128, (4), padding='same'))
model.add(Activation('relu'))

model.add(Conv1D(64, (4), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=(2)))

model.add(Conv1D(64, (4), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=(2)))

model.add(Flatten())

model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='nadam',
              metrics=['accuracy'])

model.fit(train_X, train_y, epochs=100, batch_size=32)

score = model.evaluate(test_X, test_y, batch_size=32)
print(score)

print(test_X[1].shape)

if score[1] >= 0.50:
  model.save("event-model-acc-%.2f" % (score[1]*100)+".h5")
  pickle.dump(model, open("event-model-acc-%.2f" % (score[1]*100)+".pkl", 'wb'))

print(model.summary())