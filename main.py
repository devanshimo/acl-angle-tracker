import cv2
import mediapipe as mp

from src.pose import PoseEstimator
from src.keypoints import get_knee_points
from src.angles import compute_angle, AngleSmoother

# -----------------------
# Setup
# -----------------------

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # webcam
pose_estimator = PoseEstimator()
smoother = AngleSmoother(window_size=7)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
min_angle=0
max_angle=180
# -----------------------
# Main loop
# -----------------------

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process pose
    result = pose_estimator.process_frame(frame)

    if result.pose_landmarks:
        # Draw full pose skeleton
        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # Image dimensions
        h, w, _ = frame.shape

        # Extract knee joint points
        hip, knee, ankle = get_knee_points(
            result.pose_landmarks.landmark,
            w,
            h,
            side="left"
        )

        # Compute & smooth knee angle
        raw_angle = compute_angle(hip, knee, ankle)
        angle = smoother.smooth(raw_angle)
        min_angle = min(min_angle, angle)
        max_angle = max(max_angle, angle)


        # Draw knee angle
        cv2.putText(
            frame,
            f"Knee Angle: {int(angle)} deg",
            (knee[0] + 10, knee[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f"Min: {int(min_angle)} deg",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Max: {int(max_angle)} deg",
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )


        # Draw joint points (debug clarity)
        cv2.circle(frame, hip, 5, (0, 255, 0), -1)
        cv2.circle(frame, knee, 5, (0, 0, 255), -1)
        cv2.circle(frame, ankle, 5, (255, 0, 0), -1)

    # Show output
    cv2.imshow("ACL Angle Tracker", frame)

    # Exit cleanly
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------
# Cleanup
# -----------------------

cap.release()
cv2.destroyAllWindows()
