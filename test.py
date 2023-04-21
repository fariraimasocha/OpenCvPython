import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from cvzone.ClassificationModule import Classifier
import math
import time
import pyttsx3
import matplotlib

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

folder = "Data/3"
counter = 0

labels = ["A", "B", "C", "3", "Hello", "I Love You", "Thanks", "L"]

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

while True:
    success, img = cap.read()
    imgOutput = img.copy()

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x + offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction_index = classifier.getPrediction(imgWhite)[1]
            prediction = labels[prediction_index]
            print(prediction)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction_index = classifier.getPrediction(imgWhite)[1]
            prediction = labels[prediction_index]

        cv2.putText(imgOutput, prediction, (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

        engine.say(prediction)
        engine.runAndWait()
        time.sleep(0.5)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)

