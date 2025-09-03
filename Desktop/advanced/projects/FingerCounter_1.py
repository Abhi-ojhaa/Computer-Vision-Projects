import cv2
import time
import os # The OS library lets us interact with the operating system
from ailib import HandTrackingModule as htm

################################
# Setup
################################
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# --- Loading the Overlay Images ---
folderPath = "FingerImages"
# os.listdir() gives us a list of all the file names in a folder
myList = os.listdir(folderPath)
overlayList = []
# Loop through all the image names and load them using cv2.imread
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
# --------------------------------

pTime = 0
detector = htm.handDetector(detectionCon=0.75)
# These are the IDs for the fingertips
tipIds = [4, 8, 12, 16, 20]

################################
# Main Loop
################################
while True:
    success, img = cap.read()
    if not success:
        continue

    # 1. Find the hand
    img = detector.findHands(img)
    # 2. Get the landmark list (we only need it for the fingersUp logic)
    lmList, bbox = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # 3. Use our module's method to check which fingers are up
        fingers = detector.fingersUp()
        
        # 4. Count the total number of "up" fingers
        totalFingers = fingers.count(1)

        # 5. Display the corresponding image
        # Get the height and width of the overlay image
        h, w, c = overlayList[totalFingers].shape
        # Overlay the image on our main camera feed
        img[0:h, 0:w] = overlayList[totalFingers]

        # 6. Display the count as text
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 25)

    # --- FPS Counter ---
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()