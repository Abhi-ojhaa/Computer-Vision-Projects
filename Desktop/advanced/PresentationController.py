import cv2
import time
from Handtracking import HandTrackingModule as htm
import pyautogui # Our "keyboard remote control"

################################
# Setup
################################
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
# NOTE: Make sure your HandTrackingModule.py is in the same folder!
detector = htm.handDetector(detectionCon=0.8, maxHands=1)

# A simple cooldown timer to prevent multiple commands at once
gesture_cooldown = 0
COOLDOWN_FRAMES = 30 # Wait 30 frames before allowing a new gesture

################################
# Main Loop
################################
while True:
    success, img = cap.read()
    if not success:
        continue
    img = cv2.flip(img, 1)

    # 1. Find the hand
    img = detector.findHands(img, draw=True)
    lmList, bbox = detector.findPosition(img, draw=False)

    # Decrement cooldown timer
    if gesture_cooldown > 0:
        gesture_cooldown -= 1

    if len(lmList) != 0:
        # 2. Count the fingers
        fingers = detector.fingersUp()
        totalFingers = fingers.count(1)

        # 3. Recognize Gestures (if cooldown is over)
        if gesture_cooldown == 0:
            # Gesture for Next Slide (2 fingers up)
            if totalFingers == 2 and fingers[1] == 1 and fingers[2] == 1:
                pyautogui.press('right') # Press the right arrow key
                print("Next Slide")
                gesture_cooldown = COOLDOWN_FRAMES # Start cooldown

            # Gesture for Previous Slide (3 fingers up)
            if totalFingers == 3 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                pyautogui.press('left') # Press the left arrow key
                print("Previous Slide")
                gesture_cooldown = COOLDOWN_FRAMES # Start cooldown

        # Gesture for Laser Pointer (Open Hand)
        if totalFingers == 5:
            x1, y1 = lmList[8][1:] # Get index finger tip
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)

    # --- FPS Counter ---
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()