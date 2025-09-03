import cv2
import numpy as np
from ailib import HandTrackingModule as htm
import time
import autopy

# Setup

wCam, hCam = 640, 480
frameR = 100 # Frame Reduction for the actionable area
smoothening = 7 # Smoothing factor

pTime = 0
plocX, plocY = 0, 0 # Previous location of x, y
clocX, clocY = 0, 0 # Current location of x, y

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1) # We only want to track one hand
wScr, hScr = autopy.screen.size() #  computer's screen size

##########################
# Main Loop
##########################
while True:
    # 1. Finding hand Landmarks
    success, img = cap.read()
    if not success:
        continue
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Tip of the index finger
        x2, y2 = lmList[12][1:] # Tip of the middle finger

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        # Create a "drawing frame" to confine mouse movement
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)

        # 4. Only Index Finger is Up: Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Convert Coordinates from camera space to screen space
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # 6. Smoothen Values to prevent jitter
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 7. Move Mouse
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # 8. Both Index and Middle fingers are up: Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            
            # 10. Click mouse if distance is short
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]),
                           15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
                
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()