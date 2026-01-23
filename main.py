import cv2
import mediapipe as mp
import numpy as np

from src.pose import PoseEstimator
from src.keypoints import get_knee_points
from src.angles import compute_angle, AngleSmoother

# -----------------------
# Setup
# -----------------------

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
pose_estimator = PoseEstimator()
smoother = AngleSmoother(window_size=7)

mp_pose = mp.solutions.pose

# Track knee flexion range (ortho convention)
min_angle = None
max_angle = None

# -----------------------
# Main loop
# -----------------------

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result = pose_estimator.process_frame(frame)

    if result.pose_landmarks:
        h, w, _ = frame.shape

        # Extract LEFT knee points only
        hip, knee, ankle = get_knee_points(
            result.pose_landmarks.landmark,
            w, h,
            side="left"
        )

        # Draw leg only (no full skeleton)
        cv2.circle(frame, hip, 6, (0, 255, 0), -1)
        cv2.circle(frame, knee, 6, (0, 0, 255), -1)
        cv2.circle(frame, ankle, 6, (255, 0, 0), -1)

        cv2.line(frame, hip, knee, (0, 255, 0), 3)
        cv2.line(frame, knee, ankle, (255, 0, 0), 3)

        # Segment length sanity check
        thigh_len = np.linalg.norm(np.array(hip) - np.array(knee))
        shin_len = np.linalg.norm(np.array(ankle) - np.array(knee))

        valid_measurement = True
        if thigh_len < 50 or shin_len < 50:
            valid_measurement = False

        if valid_measurement:
            # Compute & smooth angle
            raw_angle = compute_angle(hip, knee, ankle)
            angle = smoother.smooth(raw_angle)

            # Convert to ortho knee flexion angle
            ortho_angle = 180 - angle

            # Reject impossible values
            if 10 <= ortho_angle <= 160:

                # Initialize or update min/max
                if min_angle is None:
                    min_angle = ortho_angle
                    max_angle = ortho_angle
                else:
                    min_angle = min(min_angle, ortho_angle)
                    max_angle = max(max_angle, ortho_angle)

                # Display current knee flexion
                cv2.putText(
                    frame,
                    f"Knee Flexion: {int(ortho_angle)} deg",
                    (knee[0] + 10, knee[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    # Always show frame (never skip)
    if min_angle is not None:
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

    cv2.imshow("ACL Angle Tracker", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        min_angle = None
        max_angle = None
    if key == ord('q'):
        break

# -----------------------
# Cleanup
# -----------------------

cap.release()
cv2.destroyAllWindows()
