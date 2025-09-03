import cv2
import numpy as np
import time
import PoseModule as pm

# --- Setup ---
cap = cv2.VideoCapture(0) # Load the video file

detector = pm.poseDetector()
count = 0
direction = 0 # 0 for up-stroke, 1 for down-stroke
pTime = 0

# --- Main Loop ---
while True:
    success, img = cap.read()
    if not success:
        # If the video ends, we break the loop
        break

    # Resize the image to a standard size (optional, but good practice)
    img = cv2.resize(img, (1280, 720))

    # 1. Find the Pose
    img = detector.findPose(img, draw=False)
    lmList, bbox = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # 2. Use the new findAngle method for the right arm
        # Landmarks: 12 (shoulder), 14 (elbow), 16 (wrist)
        angle = detector.findAngle(img, 12, 14, 16)

        # 3. Map the angle to a percentage and a bar value
        # The angle for a bicep curl typically ranges from ~30 (curled) to ~160 (extended)
        per = np.interp(angle, (30, 160), (100, 0)) # Map 30deg to 100%, 160deg to 0%
        bar = np.interp(angle, (30, 160), (100, 650)) # Map for the pixel height of the bar

        # 4. Count the Repetitions
        if per == 100: # We are at the top of the curl
            if direction == 0: # Check if we were moving up
                count += 0.5
                direction = 1 # Change direction to down-stroke
        if per == 0: # We are at the bottom of the curl
            if direction == 1: # Check if we were moving down
                count += 0.5
                direction = 0 # Change direction to up-stroke

        # --- Draw the UI elements ---
        # Draw Curl Progress Bar
        cv2.rectangle(img, (1100, 100), (1175, 650), (0, 255, 0), 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4,
                    (0, 255, 0), 4)

        # Draw Repetition Counter
        cv2.rectangle(img, (0, 450), (250, 720), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15,
                    (255, 255, 255), 25)

    # --- FPS Counter ---
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
                (255, 0, 0), 5)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()