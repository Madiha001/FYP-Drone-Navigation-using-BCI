import time
from serial import Serial
import pickle
import numpy as np
from arduino_connection import sendCommand
from headset_connection import getBoardId, getBoard, getHeadsetData
from sklearn.preprocessing import StandardScaler

# Initialize a StandardScaler object
scaler = StandardScaler()

# define the ports
arduino_port = 'COM5'
headset_port = 'COM3'

# connect to the arduino
ser = Serial(arduino_port, 9600, timeout=2)

# connect to the headset
board_id = getBoardId()
board = getBoard(board_id, headset_port)
#data = getHeadsetData(board) # method to get 1 SEC data from the headset
#print(data)

# load the saved model
model = pickle.load(open('model.sav', 'rb'))

print('Lets Start Controling the Drone!')
predictions = []

while True:

    for i in range(15):
        data = getHeadsetData(board) # method to get 1 SEC data from the headset

        # scale the data
        for i in range(len(data)):
            scaler.fit(data[i])
            data[i] = scaler.transform(data[i])
        
        # convert the data to numpy array
        data = np.array(data)
        # predict the command
        print(data.shape)
        #command = model.predict(data)
        predictions.append(data)
        time.sleep(1)
    
    # get the most common command
    direction = max(set(predictions), key=predictions.count)

    # send the command to the arduino
    sendCommand(ser, direction)

    # clear the predictions
    predictions = []

    # sendCommand(ser, '1')
    # time.sleep(1)
    # sendCommand(ser, '2')
    # time.sleep(1)
    # sendCommand(ser, '3')
    # time.sleep(1)
    # sendCommand(ser, '4')
    # time.sleep(1)
    # sendCommand(ser, '5')
    # time.sleep(1)
    # sendCommand(ser, '6')
    # time.sleep(1)