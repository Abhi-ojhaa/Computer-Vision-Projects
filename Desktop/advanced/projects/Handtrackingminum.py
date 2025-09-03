import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    if not success:
        continue

    # Flip the image
    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            #find and highlight the index finger
            h, w, c = img.shape
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 8:
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
            
            # Draw the default hand connections
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # --- FPS CALCULATION AND DISPLAY ---
    cTime = time.time()
    if (cTime - pTime) > 0:
        fps = 1 / (cTime - pTime)
    else:
        fps = 0
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
