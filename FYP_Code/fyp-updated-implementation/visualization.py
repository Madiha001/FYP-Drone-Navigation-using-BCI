import numpy as np
from dataset_tools import preprocess_raw_eeg
import matplotlib.pyplot as plt


sample = np.load('dataset_250/hands/1675791231.npy')
sample = np.array(sample)   

#Raw Data visualization
_, columns = sample.shape
x = np.linspace(0, 250, columns)
plt.plot(x, sample.T)
plt.show()

#Data after standardization

for i in range(len(sample)):
    mean = sample[i].mean()
    std = sample[i].std()
    if std < 0.001:
        sample[i, :] = (sample[i, :] - mean) / (std + 0.1)
    else:
        sample[i, :] = (sample[i, :] - mean) / std     

plt.plot(x,sample.T)
plt.show()                
data_X, fft_data_X = preprocess_raw_eeg(sample.reshape((1, 8, 250)), lowcut=8, highcut=45, coi3order=0)
nn_input = data_X.reshape((1, 8, 250, 1)) 

