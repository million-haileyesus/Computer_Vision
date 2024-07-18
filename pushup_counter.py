import cv2 as cv
import mediapipe as mp
import numpy as np

# Initialize video capture from the default camera (usually the webcam)
camera = cv.VideoCapture(0)

# Initialize MediaPipe Pose model for human pose estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Utility for drawing pose landmarks on the image
mp_draw = mp.solutions.drawing_utils

# Flag to control the main loop
can_quit = False

# Constants representing landmark indices for pose landmarks
RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16
LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15

# Initialize pushup detection variables
is_downward = False  # Flag indicating if the current pose is downward
pushup_count = 0     # Counter for completed pushups

# Main loop for processing video frames
while not can_quit:
    # Read a frame from the camera
    success, image = camera.read()

    # Flip the image horizontally for a mirrored view and resize it
    image = cv.flip(image, 1)
    image = cv.resize(image, (1600, 920))

    # Convert the image to RGB format required by MediaPipe
    rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Process the RGB image to detect pose landmarks
    result = pose.process(rgb_image)

    # Check if pose landmarks were detected
    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark

        # Retrieve landmark coordinates for the right and left arms
        right_shoulder = landmarks[RIGHT_SHOULDER]
        right_elbow = landmarks[RIGHT_ELBOW]
        right_wrist = landmarks[RIGHT_WRIST]
        left_shoulder = landmarks[LEFT_SHOULDER]
        left_elbow = landmarks[LEFT_ELBOW]
        left_wrist = landmarks[LEFT_WRIST]

        # Check if elbows are above shoulders (indicating a downward motion)
        if right_elbow.y < right_shoulder.y and left_elbow.y < left_shoulder.y:
            if not is_downward:
                is_downward = True  # Arm lowering detected

        # Check if elbows are at or below shoulders (indicating an upward motion)
        elif right_elbow.y >= right_shoulder.y and left_elbow.y >= left_shoulder.y:
            if is_downward:
                is_downward = False  # Arm raising detected, increment pushup count
                pushup_count += 1
                print("Pushup Count:", pushup_count)

            # Display the pushup count on the image
            cv.rectangle(image, (0, 450), (250, 720), (0, 255, 0), cv.FILLED)
            cv.putText(image, str(pushup_count), (45, 670), cv.FONT_HERSHEY_PLAIN, 15, (0, 0, 255), 25)

    # Draw pose landmarks and connections on the image
    mp_draw.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the image in a window
    cv.imshow("Pushup Counter", image)

    # Check for 'q' key press to quit the loop
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
pose.close()
camera.release()
cv.destroyAllWindows()

