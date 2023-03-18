import argparse
import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams

def getBoardId():
        parser = argparse.ArgumentParser()
        parser.add_argument('--board-id', 
                                type=int, 
                                help='board id, check docs to get a list of supported boards', 
                                required=False, 
                                default=0)
        args = parser.parse_args()
        return args.board_id

def getBoard(board_id, serial_port):
        params = BrainFlowInputParams()
        params.serial_port = serial_port # serial port for Cyton board may be COM5, COM6, COM7 etc
        board = BoardShim(board_id, params)
        return board

def getHeadsetData(board):
        board.prepare_session()
        board.start_stream()
        time.sleep(1)
        data = board.get_board_data(256*1) ## (256 Hz @ 1sec) ## 256*1 = 256 samples
        board.stop_stream()
        board.release_session()
        return data