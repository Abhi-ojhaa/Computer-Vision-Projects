# AI Vision: A Portfolio of Real-Time Computer Vision Projects

Welcome to my portfolio of computer vision projects built with Python, OpenCV, and MediaPipe. This collection demonstrates a journey from basic real-time tracking to creating practical, gesture-based applications that control a computer's functions.

## Projects Included
This repository contains a series of progressively complex applications, each in its own self-contained script within the `projects/` directory.

| Project Name | Description | Key Skills Demonstrated |
| :--- | :--- | :--- |
| **Volume Hand Control** | Controls the computer's system volume by changing the distance between the thumb and index finger. | Landmark Tracking, Distance Calculation, OS Interfacing (`pycaw`) |
| **AI Virtual Mouse** | Moves the mouse cursor with the index finger and simulates clicks with a pinching gesture. | Coordinate Space Mapping, Movement Smoothing, OS Control (`autopy`) |
| **Finger Counter** | Counts the number of extended fingers held up to the camera in real-time. | Geometric Analysis, Gesture Recognition |
| **AI Personal Trainer** | Tracks the angle of an arm to count bicep curl repetitions, providing visual feedback with a progress bar. | Angle Calculation, State Management (Rep Counting) |
| **Virtual Painter** | Allows a user to draw on the screen with their index finger and select colors with a two-finger gesture. | Mode Switching, UI Interaction, Image Blending |
| **AI Presentation Controller**| Controls a presentation slideshow (e.g., PowerPoint) with distinct hand gestures for next/previous slide. | Inter-Application Control (`pyautogui`), Gesture Recognition |

## Technologies Used
* **Python**
* **OpenCV** for camera management and image processing.
* **MediaPipe** for robust, real-time hand, pose, and face tracking models.
* **NumPy** for numerical operations, especially for mapping values between ranges.
* **PyAutoGUI** & **Pycaw** for controlling the operating system (mouse, keyboard, audio).

## Setup and Installation
To run these projects, you'll need to set up a virtual environment and install the required libraries.

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd <your-repo-name>
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\Activate
    ```
3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run a Project
All runnable scripts are located in the `projects/` directory. To run a project, use the following command from the root folder, replacing `ProjectName` with the name of the script you want to run.

```bash
# Example for the AI Trainer
python -m projects.AITrainer
```
```bash
# Example for the Presentation Controller
python -m projects.PresentationController
```

## Custom Modules
This portfolio utilizes custom, reusable modules located in the `ailib/` package to keep the code clean and organized, demonstrating a key software engineering principle.
* **HandTrackingModule.py**: A robust wrapper for MediaPipe's hand tracking solution.
* **PoseModule.py**: A wrapper for MediaPipe's pose estimation solution.
