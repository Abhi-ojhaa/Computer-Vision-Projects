import cv2
import mediapipe as mp
import math

class poseDetector():
    """
    A class to detect and estimate human pose using MediaPipe.
    """
    def __init__(self, mode=False, model_complexity=1, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        """
        Initializes the pose detector.
        :param mode: Whether to treat the input images as a batch or a stream.
        :param model_complexity: Complexity of the pose landmark model: 0, 1, or 2.
        :param smooth: Whether to filter landmarks across different input images to reduce jitter.
        :param detectionCon: Minimum confidence value from the person detection model.
        :param trackCon: Minimum confidence value from the landmark tracking model.
        """
        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        # This uses named arguments to be compatible with newer MediaPipe versions
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     model_complexity=self.model_complexity,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        """
        Finds the pose in an image and draws the landmarks and connections.
        :param img: The image to process.
        :param draw: Whether to draw the pose on the image.
        :return: The image with the pose drawn on it.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        """
        Finds the coordinates of each landmark and an optional bounding box.
        :param img: The image to process.
        :param draw: Whether to draw circles on the landmarks.
        :return: A list of landmark coordinates and the bounding box.
        """
        self.lmList = []
        bbox = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        # This now returns two values to prevent unpacking errors
        return self.lmList, bbox

    def findAngle(self, img, p1, p2, p3, draw=True):
        """
        Finds the angle between three points.
        :param img: The image to draw on.
        :param p1: The first point (e.g., shoulder).
        :param p2: The second point, which is the vertex (e.g., elbow).
        :param p3: The third point (e.g., wrist).
        :param draw: Whether to draw the angle on the image.
        :return: The calculated angle.
        """
        # Get the landmark coordinates
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle using trigonometry
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # Draw the angle on the image for visual feedback
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
        return angle

## Key Features of This Module