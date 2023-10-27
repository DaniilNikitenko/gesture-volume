import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import eel
import numpy as np


eel.init('web')


@eel.expose
def computerVision():
    cap = cv2.VideoCapture(1)
    cap.set(3, 1280)
    cap.set(4, 720)
    detector = HandDetector(detectionCon=0.8, maxHands=2)
    colorR = (150, 0, 90)

    cx, cy, w, h = 100, 100, 200, 200

    class DragRect:
        def __init__(self, posCenter, size=[200, 200]):
            self.posCenter = posCenter
            self.size = size
            self.dragging = False  # Флаг для определения, перетаскивается ли квадрат в данный момент

        def update(self, cursor, hands):
            cx, cy = self.posCenter
            w, h = self.size

            if len(hands) == 1:  # Если обнаружена только одна рука, обновляем положение квадрата
                lmList = hands[0]['lmList']
                if cx - w // 2 < lmList[8][0] < cx + w // 2 and cy - h // 2 < lmList[8][1] < cy + h // 2:
                    self.dragging = True
                    self.posCenter = lmList[8][0], lmList[8][1]
                else:
                    self.dragging = False

            else:  # Если обнаружено две руки, убираем захват с квадрата
                self.dragging = False

        def display(self, img):
            cx, cy = self.posCenter
            w, h = self.size
            cv2.rectangle(img, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
            cvzone.cornerRect(img, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

    rectList = []
    for x in range(5):
        rectList.append(DragRect([x * 250 + 150, 150]))

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img, draw=True)

        for rect in rectList:
            rect.update([], hands)
            rect.display(img)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

eel.start('main.html', size = (500, 700))



