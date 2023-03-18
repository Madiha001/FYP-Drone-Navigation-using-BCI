import numpy as np
from statistics import mean
import os
from pathlib import Path
from brainflow import DataFilter, FilterTypes
from scipy.fft import fft

ACTIONS = ["chew","clench","eye_blink" ,"feet","head","jerk"]
BOARD_SAMPLING_RATE = 250

def check_std_deviation(sample: np.ndarray, lower_threshold=0.01, upper_threshold=25):
    stds = []
    for i in range(len(sample)):
        std = sample[i].std()
        stds.append(int(std))
        print(f"{i} - {std}")
    for i in range(len(sample)):
        std = sample[i].std()
        if std < lower_threshold:
            print("An electrode may be disconnected")
            return False
        if std > upper_threshold:
            print(f"Noisy_sample, channel{i} - {std}")
    print(f"average std deviation: {mean(stds)}")
    while True:
        input_save = input("Do you want to save this sample? [Y,n]")
        if 'n' in input_save:
            return False
        elif 'y' in input_save.lower() or '' == input_save:
            return True
        else:
            print("Enter 'y' to save the sample or 'n' to discard the sample")

def load_all_raw_data(starting_dir: Path, channels=8, NUM_TIMESTAMP_PER_SAMPLE=250):
    data_X = np.empty((0, channels, NUM_TIMESTAMP_PER_SAMPLE))
    data_y = np.empty(0)
    mapping = {}
    filtered_actions = [action_dir for action_dir in starting_dir.iterdir() if action_dir.name in ACTIONS]
    for index, actions_dir in enumerate(filtered_actions):
        if actions_dir.name in ACTIONS:
            for sample_path in actions_dir.iterdir():
                data_X = np.append(data_X, np.expand_dims(np.load(str(sample_path)), axis=0), axis=0)
                data_y = np.append(data_y, index)
                mapping[index] = actions_dir.name
    return data_X, data_y, mapping


def load_data(starting_dir, shuffle=True, balance=False):
   
    data = [[] for i in range(len(ACTIONS))]
    for i, action in enumerate(ACTIONS):
        data_dir = os.path.join(starting_dir, action)
        for file in sorted(os.listdir(data_dir)):
            data[i].append(np.load(os.path.join(data_dir, file)))

    if balance:
        lengths = [len(data[i]) for i in range(len(ACTIONS))]
        for i in range(len(ACTIONS)):
            data[i] = data[i][:min(lengths)]

        lengths = [len(data[i]) for i in range(len(ACTIONS))]

    combined_data = []

    for i in range(len(ACTIONS)):
        lbl = np.zeros(len(ACTIONS), dtype=int)
        lbl[i] = 1
        for sample in data[i]:
            combined_data.append([sample, lbl])

    if shuffle:
        np.random.shuffle(combined_data)

    X = []
    y = []
    for sample, label in combined_data:
        X.append(sample)
        y.append(label)

    return np.array(X), np.array(y)

def standardize(data, std_type="channel_wise"):
    for k in range(len(data)):
        sample = data[k]
        for i in range(len(sample)):
            mean = sample[i].mean()
            std = sample[i].std()
            if std < 0.001:
                data[k, i, :] = (data[k, i, :] - mean) / (std + 0.1)
            else:
                data[k, i, :] = (data[k, i, :] - mean) / std

    return data

def preprocess_raw_eeg(data, fs=250, lowcut=2.0, highcut=65.0, MAX_FREQ=60, power_hz=50, coi3order=3):
    data = standardize(data)
    fft_data = np.zeros((len(data), len(data[0]), MAX_FREQ))
    for sample in range(len(data)):
        for channel in range(len(data[0])):
            DataFilter.perform_bandstop(data[sample][channel], fs, 2.0, power_hz,
                                        5, FilterTypes.BUTTERWORTH.value, 0)

            if coi3order != 0:
                DataFilter.perform_wavelet_denoising(data[sample][channel], 'coif3', coi3order)

            DataFilter.perform_bandpass(data[sample][channel], fs,
                                        int((lowcut + highcut) / 2), highcut - lowcut, order=5,
                                        filter_type=FilterTypes.BUTTERWORTH.value, ripple=0)

            fft_data[sample][channel] = np.abs(fft(data[sample][channel])[:MAX_FREQ])

    fft_data = standardize(fft_data)
    return data, fft_data