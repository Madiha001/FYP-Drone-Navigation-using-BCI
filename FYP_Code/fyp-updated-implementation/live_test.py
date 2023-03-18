from brainflow import BoardShim, BrainFlowInputParams, BoardIds
from matplotlib import pyplot as plt
from dataset_tools import check_std_deviation, BOARD_SAMPLING_RATE, preprocess_raw_eeg
import keras
import numpy as np
import argparse
import time

model = keras.models.load_model('./models_6D/fold_n2/2_model.h5')
model1 = keras.models.load_model('./models_6D/fold_n1/1_model.h5')
model2 = keras.models.load_model('./models_6D/fold_n2/2_model.h5')
model3 = keras.models.load_model('./models_6D/fold_n3/3_model.h5')

def predict_sample(model1, model2, model3, sample):
    # Get the predictions from the 3 models
    pred1 = model1.predict(sample)
    pred2 = model2.predict(sample)
    pred3 = model3.predict(sample)

    # Combine the predictions into a list
    predictions = [pred1[0], pred2[0], pred3[0]]

    # Count the number of occurrences of each prediction
    pred_counts = {pred: predictions.count(pred) for pred in predictions}

    # Get the prediction with the highest count (majority vote)
    majority_vote = max(pred_counts, key=pred_counts.get)
    return majority_vote
    # Print the majority vote
    # print("The majority vote of the 3 models is:", majority_vote)



if __name__ == '__main__':

    NUM_CHANNELS = 8
    NUM_TIMESTAMP_PER_SAMPLE = 250

    parser = argparse.ArgumentParser()
    parser.add_argument('--serial-port', type=str, help='serial port',
                        required=False, default='COM4')

    args = parser.parse_args()
    params = BrainFlowInputParams()
    params.serial_port = args.serial_port

    board = BoardShim(0, params)
    board.prepare_session()

    last_act = None

    for i in range(50):
        input("Press enter to acquire a new action")
        print("Think in 4")
        time.sleep(1.5)
        print("Think in 3")
        time.sleep(1.5)
        print("Think in 2")
        time.sleep(1.5) 
        print("Think in 1")
        time.sleep(1.5)
        print("Think in 0")
        time.sleep(1.5)
        print("Think NOW!!")
        time.sleep(1.5)  
         

        board.start_stream() 
        time.sleep(1.5 * (NUM_TIMESTAMP_PER_SAMPLE / BOARD_SAMPLING_RATE))
        data = board.get_current_board_data(NUM_TIMESTAMP_PER_SAMPLE)
        board.stop_stream()

        sample = []
        eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
        for channel in eeg_channels:
            sample.append(data[channel])

        print(np.array(sample).shape)
        
        if np.array(sample).shape == (NUM_CHANNELS, NUM_TIMESTAMP_PER_SAMPLE) and check_std_deviation(np.array(sample)):
            sample = np.array(sample)
            
            data_X, fft_data_X = preprocess_raw_eeg(sample.reshape((1, 8, 250)), lowcut=8, highcut=45, coi3order=0)
            nn_input = data_X.reshape((1, 8, 250, 1)) 
            
            # result = model.predict(nn_input)
            # print(result)
            # result = np.argmax(result)
            
            result = predict_sample(model1, model2, model3, nn_input)
            print(result)
            result = np.argmax(result)
            if result == 0:
                print("chew")
            elif result == 1:
                print("clench")
            elif result == 2:
                print("eye_blink")
            elif result == 3:
                print("feet")    
            elif result == 4:
                print("head")            
            elif result == 5:
                print("jerk")
            # elif result == 6:
            #     print("left_foot")
            # elif result == 7:
            #     print("none")     
            # elif result == 8:
            #     print("right_foot")                
        plt.show()
       
    board.release_session()
