import pickle
import os
import cv2
import time
import math
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
os.chdir( os.path.dirname(os.path.abspath(__file__)) )


##################
# 상수 설정
##################
FILE_NAME = "output1685351265074.npy"
NORMALIZE_DISTANCE_LIMIT = 2000
MINIMUM_OBJECT_AREA = 3

MINIMUM_POSITION_WEIGHT = 0.2
VARIABLE_POSITION_WEIGHT = 1-MINIMUM_POSITION_WEIGHT

SCREEN_MAGNIFICATION = 5

HEIGHT_RESOLUTION = math.radians(13/10)
WIDTH_RESOLUTION = math.radians(2/3)


##################
# 전역 변수 설정
##################
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=100,
    detectShadows=False,
)
objectTrackingLog = np.array([
    np.zeros(3, dtype=int)
])
nowDistanceMap = np.array([])


##################
# 영상 전처리
##################
def makeRgbImage(originalData):
    b = originalData
    np.place(b, b==255, 0)

    r = 255 - originalData
    np.place(r, r==255, 0)
    g = np.zeros((len(originalData), ), dtype=np.uint8)

    return cv2.merge([
        b.reshape(60, 160), 
        g.reshape(60, 160), 
        r.reshape(60, 160)
    ])

def distanceDataToNormalizedNumpyArray(distanceData):
    result = distanceData
    np.place(result, result>4000, 2000)
    result = result / NORMALIZE_DISTANCE_LIMIT * 255
    return result

def limitArrayNumber(array, maxNumber):
    result = np.array([], dtype=np.uint8)

    for i in array:
        if i>maxNumber:
            result = np.append(result, maxNumber)
        else:
            result = np.append(result, i)
    
    return result


##################
# 가져오기
##################
def getDistance(x, y):
    return int(nowDistanceMap[y][x])

def getPositionWeight():
    # MINIMUM_POSITION_WEIGHT + 가중치로 계산
    if len(objectTrackingLog)<2:
        return (1, 1, 1)

    pastDot = objectTrackingLog[-1]
    pastPastDot = objectTrackingLog[-2]
    
    xDelta = max(abs(pastDot[0] - pastPastDot[0]), 1)
    yDelta = max(abs(pastDot[1] - pastPastDot[1]), 1)
    zDelta = max(abs(pastDot[2] - pastPastDot[2]), 1)

    maxDelta = min(xDelta, yDelta, zDelta)

    return (
        MINIMUM_POSITION_WEIGHT + (maxDelta/xDelta)*VARIABLE_POSITION_WEIGHT,
        MINIMUM_POSITION_WEIGHT + (maxDelta/yDelta)*VARIABLE_POSITION_WEIGHT,
        MINIMUM_POSITION_WEIGHT + (maxDelta/zDelta)*VARIABLE_POSITION_WEIGHT
    )

def getProximateDot(dotList):
    global objectTrackingLog
    pastDot = objectTrackingLog[-1]

    proximateDot = [0, 0, 0]
    proximateDistance = 99999999
    for dot in dotList:
        dotDistance = getSquare3Ddistance(pastDot,dot)
        if (dotDistance < proximateDistance):
            proximateDot = dot
            proximateDistance = dotDistance
    
    objectTrackingLog = np.append(objectTrackingLog, [proximateDot], axis=0)
    return proximateDot

def getSquare3Ddistance(dot1, dot2):
    positionWeight = getPositionWeight()
    return (
        ((dot1[0] - dot2[0])**2)*positionWeight[0] + 
        ((dot1[1] - dot2[1])**2)*positionWeight[1] + 
        ((dot1[2] - dot2[2])**2)*positionWeight[2]
    )

def getMoveObjectPosition(frame):
    result = []

    fgmask = fgbg.apply(frame)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask)


    for index, centroid in enumerate(centroids):
        if stats[index][0] == 0 and stats[index][1] == 0:
            continue
        if np.any(np.isnan(centroid)):
            continue


        x, y, width, height, area = stats[index]
        centerX, centerY = int(centroid[0]), int(centroid[1])

        if area > MINIMUM_OBJECT_AREA:
            result.append([centerX, centerY, getDistance(centerX, centerY)])
            # cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255))
    
    return result

def getRealWidth(xDots, zDots):
    result = np.array([])
    processedXlist = (xDots-80)*WIDTH_RESOLUTION
    for x, z in zip(processedXlist, zDots):
        result = np.append(result, math.sin(x)*z)
    
    return result

def getRealHeight(yDots, zDots):
    result = np.array([])
    processedXlist = -1*(yDots-30)*HEIGHT_RESOLUTION
    for x, z in zip(processedXlist, zDots):
        result = np.append(result, math.sin(x)*z)
    
    return result

def getSpeed(processedX, processedY, processedZ):
    arrayLength = len(processedX)
    distance = np.zeros(arrayLength-1, float)

    for i in range(1, arrayLength):
        distance[i-1] = math.sqrt(
            (processedX[i] - processedX[i-1]) **2 + 
            (processedY[i] - processedY[i-1])**2 + 
            (processedZ[i] - processedZ[i-1])**2
        )
    
    return distance * 15   # 1/15초마다거리이므로>15

##################
# 영상 처리 & 시각화
##################
def processImage(distanceData):
    global nowDistanceMap
    linearArray = np.array(
        distanceDataToNormalizedNumpyArray(distanceData),
        dtype=np.uint8
    )

    nowDistanceMap = linearArray.reshape(60, 160)
    image = makeRgbImage(linearArray)
    
    dotList = getMoveObjectPosition(image)
    moveObject = getProximateDot(dotList)

    visualize(image, moveObject, dotList)

def visualize(smallSizeImage, object, dotList):
    # 확대 후 실행
    image = cv2.resize(smallSizeImage, dsize=(160*SCREEN_MAGNIFICATION, 60*SCREEN_MAGNIFICATION), interpolation=cv2.INTER_AREA)
    cv2.circle(image, (object[0]*SCREEN_MAGNIFICATION, object[1]*SCREEN_MAGNIFICATION), 10, (255, 0, 255), 10)
    for dot in dotList:
        cv2.circle(image, (dot[0]*SCREEN_MAGNIFICATION, dot[1]*SCREEN_MAGNIFICATION), 1, (255, 255, 255), 2)
    
    cv2.imshow('OpenCV', image)
    cv2.waitKey(1)
    # while (cv2.waitKey(1)!=27): pass


def start(fileName=FILE_NAME):
    # 파일 받아오기
    global data
    data = np.load("./%s" % fileName)
    
    # 메인 코드
    data = data[3:]
    for i in tqdm(data):
        processImage(i)
        # time.sleep(0.06666666)
    
    processedX = getRealWidth(objectTrackingLog[5:,0], zDots=objectTrackingLog[5:, 2])
    processedY = getRealHeight(objectTrackingLog[5:, 1], zDots=objectTrackingLog[5:, 2])
    processedZ = objectTrackingLog[5:, 2]
    processedSpped = getSpeed(processedX, processedY, processedZ)

    speedFig = plt.figure(figsize=(8, 4))
    ax = speedFig.add_subplot(111)
    ax.plot(
        np.array(range(len(processedSpped)))/15, 
        processedSpped, 
        color='y'
    )

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(
        processedX,
        processedZ,     # 깊이축
        processedY,

        linewidth=1,
        color='y'
    )
    ax.scatter(
        processedX,
        processedZ,     # 깊이축
        processedY,

        c = np.array(range(len(objectTrackingLog[5:]))) / len(objectTrackingLog[5:]),
        cmap="copper"
    )

    plt.show()

    exit(0)

if __name__ == '__main__':
    start()