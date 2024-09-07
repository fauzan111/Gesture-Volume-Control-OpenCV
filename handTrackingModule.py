
'''------------------Creating Hand trackig Package----------------------------'''
import cv2 
import mediapipe as np
import time
import math



'''-------------------------------Creation  of Hand_Detection Class---------------------------------'''
'''
               Methods inside Hand Detection Class

               1.findHands() :-   Detect No of Hands Inside The Frame

               2.FindPosition() :-  Find location Of Hands Points

               3. FingerUp() :-  Count Number Of Finger Up 

               4. Distance() :- Find Distance Between Two Points Of Finger's
         
'''


class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = int(maxHands)  # Ensure maxHands is an integer
        self.detectionCon = float(detectionCon)  # Ensure it's a float
        self.trackCon = float(trackCon)  # Ensure it's a float

        self.mpHands = np.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode, 
            max_num_hands=self.maxHands,  # Integer
            min_detection_confidence=self.detectionCon,  # Float
            min_tracking_confidence=self.trackCon  # Float
        )
        self.mpDraw = np.solutions.drawing_utils  # Added for drawing landmarks
        self.lmlist = []  # Initialize lmlist
        self.tipIds = [4, 8, 12, 16, 20]  # Tip IDs for thumb and fingers
    
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for handLMS in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLMS, self.mpHands.HAND_CONNECTIONS)
      
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmlist = []  # Initialize lmlist inside the method
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myhand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmlist.append([id, cx, cy])
                
                if draw: 
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
                    
            Xmin, Xmax = min(xList), max(xList)
            Ymin, Ymax = min(yList), max(yList)
            bbox = Xmin, Ymin, Xmax, Ymax

            if draw:
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        return self.lmlist, bbox

    def fingerUp(self):
        fingers = []
        if len(self.lmlist) == 0:
            return []

        # Thumb
        if self.lmlist[self.tipIds[0]][1] < self.lmlist[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if self.lmlist[self.tipIds[id]][2] < self.lmlist[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers

    def Distance(self, img, Top_1, Top_2, draw=True):
        if len(self.lmlist) == 0:
            return 0

        x1, y1 = self.lmlist[Top_1][1:]
        x2, y2 = self.lmlist[Top_2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        length = math.hypot(x1 - x2, y1 - y2)

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(img, (cx, cy), 7, (0, 0, 255), cv2.FILLED)
        
        return length


'''--------------------------Main Function-----------------------------------'''

def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        
        if len(lmList) != 0:
            print(lmList[0])  # Print the first landmark position (example)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()