import pandas as pd
import numpy as np
import os

def segmentation(file_path,file_name):
    data = pd.read_csv(file_path, delimiter=',', header=None)
    data = data.drop(data.columns[0], axis=1)
    data = data.drop(data.index[0])
    eeg_data = data.to_numpy()

    # Divide the data into segments
    segment_length = data.shape[0]//15
    for i in range(15):
        start = i * segment_length
        end = (i + 1) * segment_length
        segment = eeg_data[start:end, :]
        np.save('{}_segment_{}.npy'.format(file_name, i), segment)

    print('npy files have been generated')

data_dir = './dataset'

for label_type in ['left','right','back','forward','up','down']:
    dir_name = os.path.join(data_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.csv':      
            f = os.path.join(dir_name, fname)
            segmentation(f,fname)
            
