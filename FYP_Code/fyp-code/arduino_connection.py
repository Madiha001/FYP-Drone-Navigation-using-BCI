import time
from serial import Serial

def sendCommand(ser, command):
    ser.write(str.encode(command))   # send a byte
