import os
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import sys
import time

if os.name == 'posix':  # Linux or macOS
    devnull = open('/dev/null', 'w')
else:  # Windows
    devnull = open('NUL', 'w')

# Redirect stdout and stderr
sys.stdout = devnull
sys.stderr = devnull

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

model = load_model('mp_hand_gesture')

f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

# Create a loading window
loading_window = np.zeros((200, 400, 3), dtype=np.uint8)
cv2.putText(loading_window, "Loading...", (120, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
cv2.imshow("Loading", loading_window)
cv2.waitKey(1)

cap = cv2.VideoCapture(0)

time.sleep(3)

while not cap.isOpened():
    cv2.waitKey(4000)
    cap = cv2.VideoCapture(0)

cv2.destroyWindow("Loading")

print("Camera is ready!")

while True:
    _, frame = cap.read()

    x, y, c = frame.shape
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(framergb)
    
    className = ''
    
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            prediction = model.predict([landmarks])
            classID = np.argmax(prediction)
            className = classNames[classID]

    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)

    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

cap.release()
cv2.destroyAllWindows()