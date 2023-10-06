import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, mode=False, max_hands=2, detection_conf=0.5, tracking_conf=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_conf = detection_conf
        self.tracking_conf = tracking_conf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands,
                                        self.detection_conf, self.tracking_conf)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLandMrks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandMrks,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_number=0, draw=True):
        land_mark_list = []

        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            true_hand = self.results.multi_hand_landmarks[hand_number]
            for id, landm in enumerate(true_hand.landmark):
                cx, cy = int(landm.x * w), int(landm.y * h)
                # print(id, cx, cy)
                land_mark_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 255, 255), cv2.FILLED)
        return land_mark_list


def main():
    pTime = 0
    cTime = 0

    capture = cv2.VideoCapture(0)

    detector = HandDetector()

    while True:
        success, img = capture.read()
        img = detector.find_hands(img)
        land_mark_list = detector.find_position(img)
        if len(land_mark_list) != 0:
            print(land_mark_list[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 4)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()