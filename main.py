import cv2
import random
import array
import numpy as np
thres = 0.2
nms_threshold = 0.32
cap = cv2.VideoCapture('Video.mp4')
cap.set(3,1280)
cap.set(4,720)
# cap.set(10,150)

MAX_LEN = 4
DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
LOCASE_CHARACTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
					'i', 'j', 'k', 'm', 'n', 'o', 'p', 'q',
					'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
					'z']

UPCASE_CHARACTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
					'I', 'J', 'K', 'M', 'N', 'O', 'p', 'Q',
					'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
					'Z']

SYMBOLS = ['@', '#', '$', '%', '=', ':', '?', '.', '/', '|', '~', '>',
		'*', '(', ')', '<']

def getUid():
    rand_digit = random.choice(DIGITS)
    rand_upper = random.choice(UPCASE_CHARACTERS)
    rand_lower = random.choice(LOCASE_CHARACTERS)
    rand_symbol = random.choice(SYMBOLS)
    unqId = rand_lower + rand_symbol +  rand_symbol + rand_digit
    return unqId

no_of_cars = 0

classNames= []
classFile = "coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("n").split("n")

#print(classNames)
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success,img = cap.read()
    cv2.putText(img, 'Nihar.G',(580, 80), cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0), 2)
    cv2.putText(img, f'{str(int(no_of_cars))}',(1100, 80), cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0), 2)
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))
    #print(type(confs[0]))
    #print(confs)

    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
    # print(indices)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        area = w*h
        if area < 18000:
            cv2.rectangle(img, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2)     
            cv2.putText(img,f'I-{getUid()}',(box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX,1 ,(0,255,0),1)
            no_of_cars += 0.5


    cv2.imshow("Output",img)
    cv2.waitKey(1)

