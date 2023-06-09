import serial
import cv2
import numpy as np
import datetime
import os
import time
from tqdm import tqdm

import analyze

RUN_3D       =  [0x5A, 0x77, 0xFF, 0x02, 0x00, 0x08, 0x00, 0x0A]
COMMAND_STOP =  [0x5A, 0x77, 0xFF, 0x02, 0x00, 0x02, 0x00, 0x00]

HEADER1, HEADER2, HEADER3, LENGTH_LSB, LENGTH_MSB, PAYLOAD_HEADER, PAYLOAD_DATA, CHECKSUM = 0, 1, 2, 3, 4, 5, 6, 7
NORMAL_MODE = 0x5A
PRODUCT_CODE = 0x77
DEFAULT_ID = 0xFF

normalizeDistanceLimit = 4080
dataLength3D = 14400
allOutputData = np.empty((0, 9600), np.float64)
capturedFrameNumber = 0

def ReceivedCompleteData(receivedData):
    global dataLength3D
    global capturedFrameNumber

    if len(receivedData) == dataLength3D:
        Visualize(receivedData)
    
    capturedFrameNumber += 1
    print("\r%.2fs captured"%(capturedFrameNumber/15), end='')

    if capturedFrameNumber>=50:
        exitProcess(runAnalyze=True)
        exit(0)

def Visualize(receivedData):
    global allOutputData

    distanceData = Get3DDistanceDataFromReceivedData(receivedData)
    allOutputData = np.append(allOutputData, [distanceData], axis=0)

def Get3DDistanceDataFromReceivedData(receivedData):
    global dataLength3D,normalizeDistanceLimit
    index = 0
    distanceData = np.zeros(int(dataLength3D / 3 * 2), dtype=int)
    for i in range(0, dataLength3D-2, 3):
        pixelFirst = receivedData[i] << 4 | receivedData[i+1] >> 4
        pixelSecond = (receivedData[i+1] & 0xf) << 8 | receivedData[i+2]

        if pixelFirst > normalizeDistanceLimit:
            pixelFirst = normalizeDistanceLimit
        if pixelSecond > normalizeDistanceLimit:
            pixelSecond = normalizeDistanceLimit
        
        distanceData[index] = pixelFirst
        index += 1
        distanceData[index] = pixelSecond
        index += 1
    return distanceData

def exitProcess(runAnalyze=False):
    fileName = 'output%d.npy'%round(datetime.datetime.utcnow().timestamp() * 1000)
    np.save("./%s" % fileName, allOutputData)
    # with open(
    #     fileName, 
    #     'wb'
    #     ) as f:
    #     pickle.dump(allOutputData, f)
    
    ser.write(COMMAND_STOP)
    ser.close()

    if runAnalyze:
        analyze.start(fileName)
    print("Done!")

baud = 3000000
ser = serial.Serial(  # port open
    port="/dev/ttyUSB0",  # <- USB connection 
    # '/dev/ttyAMA1',# <- GPIO connection 
    # "COM14", #<- Windows PC
    baudrate=baud,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS
)
if __name__ == "__main__":
    ser.write(RUN_3D)
    # print("send : ", RUN_3D)
    print("START")
    step = HEADER1
    CPC = 0
    
    bufferCounter = 0
    receivedData = np.zeros(dataLength3D, dtype=int)
    startTime = time.time()
    nowFrame = 0
    with tqdm(total=50) as pbar:
        while True:
            nowTime = time.time()
            if (nowFrame < 50) and (int((nowTime-startTime)*15) > nowFrame):
                pbar.update(int((nowTime-startTime)*15) - nowFrame)
                nowFrame = int((nowTime-startTime)*15)

            try:
                for byte in ser.readline():
                    parserPassed = False
                    # Parse Start
                    if step != CHECKSUM:   
                        CPC = CPC ^ byte
                    if step == PAYLOAD_DATA:
                        receivedData[bufferCounter] = byte
                        bufferCounter += 1
                        if bufferCounter >= dataLength :
                            step = CHECKSUM
                    elif step == HEADER1 and byte == NORMAL_MODE:
                        step = HEADER2
                    elif step == HEADER2 and byte == PRODUCT_CODE:
                        step = HEADER3
                    elif step == HEADER3 and byte == DEFAULT_ID:
                        step = LENGTH_LSB
                        CPC = 0
                    elif step == LENGTH_LSB:
                        step = LENGTH_MSB
                        lengthLSB = byte
                    elif step == LENGTH_MSB:
                        step = PAYLOAD_HEADER
                        lengthMSB = byte
                        dataLength = (lengthMSB << 8)  | lengthLSB  - 1
                    elif step == PAYLOAD_HEADER:
                        step = PAYLOAD_DATA
                        if dataLength == 0:
                            step = CHECKSUM
                        bufferCounter = 0
                        receivedData = np.zeros(dataLength3D, dtype=int)  # clear
                    elif step == CHECKSUM:
                        step = HEADER1
                        if CPC == byte:
                            parserPassed = True
                    else:
                        step = HEADER1
                        parserPassed = False 
                    # Parse End
                    
                    if parserPassed:
                        ReceivedCompleteData(receivedData)
            except KeyboardInterrupt:
                exitProcess(runAnalyze=True)