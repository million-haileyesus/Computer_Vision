import cv2 as cv
import mediapipe as mp
import numpy as np

camera = cv.VideoCapture(0) # Use 0 to change to webcam
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
can_quit = False


RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16
LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15

is_downward = False
pushup_count = 0

while not can_quit:
    success, image = camera.read()

    # Flip the image horizontally and resize image
    image = cv.flip(image, 1)
    image = cv.resize(image, (1600, 920))

    # Process image with MediaPipe Pose model
    rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    result = pose.process(rgb_image)
    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark

        # Get landmark coordinates
        right_shoulder = landmarks[RIGHT_SHOULDER]
        right_elbow = landmarks[RIGHT_ELBOW]
        right_wrist = landmarks[RIGHT_WRIST]
        left_shoulder = landmarks[LEFT_SHOULDER]
        left_elbow = landmarks[LEFT_ELBOW]
        left_wrist = landmarks[LEFT_WRIST]

        if right_elbow.y < right_shoulder.y and left_elbow.y < left_shoulder.y:
            if not is_downward:
                is_downward = True  # Arm lowering detected
        elif right_elbow.y >= right_shoulder.y and left_elbow.y >= left_shoulder.y:
            if is_downward:
                is_downward = False  # Arm raising detected, increment pushup count
                pushup_count += 1
                print("Pushup Count:", pushup_count)

            cv.rectangle(image, (0, 450), (250, 720), (0, 255, 0), cv.FILLED)
            cv.putText(image, str(pushup_count), (45, 670), cv.FONT_HERSHEY_PLAIN, 15, (0, 0, 255), 25)

    # Draw landmarks on image
    mp_draw.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display image
    cv.imshow("Pushup Counter", image)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
pose.close()
camera.release()
cv.destroyAllWindows()


