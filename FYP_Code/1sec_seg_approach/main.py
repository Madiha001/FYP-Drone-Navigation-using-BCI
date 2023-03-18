import pandas as pd
import numpy as np
import os

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
from sklearn.preprocessing import StandardScaler

# Initialize a StandardScaler object
scaler = StandardScaler()

for i in range(len(eeg_data)):
    scaler = StandardScaler()
    scaler.fit(eeg_data[i])
    eeg_data[i] = scaler.transform(eeg_data[i])

print(eeg_data)
print(len(labels))