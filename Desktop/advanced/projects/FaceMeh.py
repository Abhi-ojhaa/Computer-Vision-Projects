import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

# --- Hire the "Face Mesh" Expert and the Artist ---
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2) # Look for up to 2 faces
mpDraw = mp.solutions.drawing_utils
# This is for changing the drawing style
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

# --- Main Loop ---
while True:
    success, img = cap.read()
    if not success:
        continue

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    # Check the report for any faces
    if results.multi_face_landmarks:
        # Loop through each face that was found
        for faceLms in results.multi_face_landmarks:
            # Draw the full mesh on the face
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION,
                                  landmark_drawing_spec=drawSpec,
                                  connection_drawing_spec=drawSpec)

    # --- FPS Counter ---
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    # --- Display ---
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
                        