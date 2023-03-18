import numpy as np
import os
import pandas as pd
from tensorflow import keras
from keras.layers import LSTM, Dense, Input
from keras.models import Model

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

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
encoder = LabelEncoder()
encoder.fit(labels)
y = encoder.transform(labels)
original_labels = encoder.inverse_transform(y)
#labels2 = encoder.inverse_transform(original_labels)
y = encoder.transform(labels)
Y = np_utils.to_categorical(y, 6)
original_labels, y, Y

print(Y[5])

from sklearn.preprocessing import StandardScaler

# Initialize a StandardScaler object
scaler = StandardScaler()

for i in range(len(eeg_data)):
    scaler = StandardScaler()
    scaler.fit(eeg_data[i])
    eeg_data[i] = scaler.transform(eeg_data[i])

print(eeg_data)

print(eeg_data[0].shape)

from sklearn.model_selection import train_test_split
import numpy

data = numpy.array(eeg_data)  #convert array to numpy type array

X_train, X_test, y_train, y_test  = train_test_split(data,Y,test_size=0.2
                                                    # ,random_state=42
                                                     ,stratify=Y
                                                     ,shuffle=True
                                                     )

print(len(X_train))
print(len(X_test))

print(len(y_train))
print(len(y_test))

train_X = numpy.array(X_train)
test_X = numpy.array(X_test)

train_y = np.array(y_train)
test_y = np.array(y_test)

train_X

print(train_X.shape)

print(test_y)

# Define the LSTM architecture
input_layer = Input(shape=train_X.shape[1:])
lstm_layer1 = LSTM(32, return_sequences=True)(input_layer)
lstm_layer2 = LSTM(32, return_sequences=True)(lstm_layer1)
lstm_layer3 = LSTM(32, return_sequences=True)(lstm_layer2)
lstm_layer4 = LSTM(64, return_sequences=True)(lstm_layer3)
lstm_layer5 = LSTM(128, return_sequences=True)(lstm_layer4)
lstm_layer6 = LSTM(32)(lstm_layer5)

# Add a fully connected layer for classification
output_layer = Dense(6, activation='softmax')(lstm_layer6)

# Create the model
model = Model(input_layer, output_layer)
model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_X, train_y, epochs=100, batch_size=32)

score = model.evaluate(test_X, test_y, batch_size=5)
print(score)