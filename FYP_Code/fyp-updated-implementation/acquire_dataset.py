from brainflow import BoardShim, BrainFlowInputParams, BoardIds
from matplotlib import pyplot as plt
from dataset_tools import check_std_deviation, ACTIONS, BOARD_SAMPLING_RATE
import numpy as np
import argparse
import time
import os


def save_sample(sample, action):
    actiondir = f"{datadir}/{action}"
    if not os.path.exists(actiondir):
        os.mkdir(actiondir)
    print(f"saving {action} personal_dataset...")
    np.save(os.path.join(actiondir, f"{int(time.time())}.npy"), np.array(sample))


if __name__ == '__main__':
    NUM_CHANNELS = 8
    NUM_TIMESTAMP_PER_SAMPLE = 250

    datadir = f"./dataset_actions_{NUM_TIMESTAMP_PER_SAMPLE}"
    if not os.path.exists(datadir):
        os.mkdir(datadir)

    parser = argparse.ArgumentParser()
    parser.add_argument('--serial-port', type=str, help='serial port',
                        required=False, default='COM4')

    args = parser.parse_args()
    params = BrainFlowInputParams()
    params.serial_port = args.serial_port

    board = BoardShim(0, params)
    board.prepare_session()

    last_act = None

    for i in range(60):
        if i % 10 == 0:
            input("Press enter to acquire data for new action")

        rand_act = np.random.randint(len(ACTIONS))
        if rand_act == last_act:
            rand_act = (rand_act + 1) % len(ACTIONS)
        last_act = rand_act
        input(f"next action is {ACTIONS[last_act]}, press enter to continue")
        print("Think ", ACTIONS[last_act], " in 2")
        time.sleep(1.5)
        print("Think ", ACTIONS[last_act], " in 1")
        time.sleep(1.5)
        print("Think ", ACTIONS[last_act], " NOW!!")
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
            save_sample(np.array(sample), ACTIONS[last_act])
        plt.show()

    board.release_session()
