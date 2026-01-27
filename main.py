import cv2
import mediapipe as mp
import numpy as np

from src.pose import PoseEstimator
from src.keypoints import get_knee_points
from src.angles import compute_angle, AngleSmoother
from src.score import ScoreManager   

# -----------------------
# Setup
# -----------------------

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
pose_estimator = PoseEstimator()
smoother = AngleSmoother(window_size=7)
score_manager = ScoreManager()

mp_pose = mp.solutions.pose

# Knee tracking state
min_angle = None
max_angle = None
tracked_side = None
prev_knee = None
MAX_KNEE_JUMP = 40  # pixels

# Progress bar UI
BAR_X, BAR_Y = 30, 140
BAR_WIDTH, BAR_HEIGHT = 300, 20

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

        # -----------------------
        # LEG LOCKING (runs once)
        # -----------------------
        if tracked_side is None:
            left_hip, left_knee, left_ankle = get_knee_points(
                result.pose_landmarks.landmark, w, h, side="left"
            )
            right_hip, right_knee, right_ankle = get_knee_points(
                result.pose_landmarks.landmark, w, h, side="right"
            )

            left_len = np.linalg.norm(np.array(left_ankle) - np.array(left_knee))
            right_len = np.linalg.norm(np.array(right_ankle) - np.array(right_knee))

            tracked_side = "left" if left_len > right_len else "right"
            print(f"Locked onto {tracked_side} leg")

        # -----------------------
        # Extract locked leg
        # -----------------------
        hip, knee, ankle = get_knee_points(
            result.pose_landmarks.landmark, w, h, side=tracked_side
        )

        # -----------------------
        # Temporal knee consistency
        # -----------------------
        if prev_knee is not None:
            knee_jump = np.linalg.norm(np.array(knee) - np.array(prev_knee))
            if knee_jump > MAX_KNEE_JUMP:
                cv2.putText(
                    frame,
                    "Knee unstable — ignoring frame",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
                )
                cv2.imshow("ACL Angle Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

        prev_knee = knee  # accept this knee position

        # -----------------------
        # Draw leg only
        # -----------------------
        cv2.circle(frame, hip, 6, (0, 255, 0), -1)
        cv2.circle(frame, knee, 6, (0, 0, 255), -1)
        cv2.circle(frame, ankle, 6, (255, 0, 0), -1)

        cv2.line(frame, hip, knee, (0, 255, 0), 3)
        cv2.line(frame, knee, ankle, (255, 0, 0), 3)

        # -----------------------
        # Geometry sanity check
        # -----------------------
        thigh_len = np.linalg.norm(np.array(hip) - np.array(knee))
        shin_len = np.linalg.norm(np.array(ankle) - np.array(knee))

        if thigh_len > 50 and shin_len > 50:
            raw_angle = compute_angle(hip, knee, ankle)
            angle = smoother.smooth(raw_angle)

            # Orthopedic knee flexion angle
            ortho_angle = 180 - angle

            if 10 <= ortho_angle <= 160:
                if min_angle is None:
                    min_angle = ortho_angle
                    max_angle = ortho_angle
                else:
                    min_angle = min(min_angle, ortho_angle)
                    max_angle = max(max_angle, ortho_angle)

                rom = max_angle - min_angle
                score_manager.update(rom)

                cv2.putText(
                    frame,
                    f"Knee Flexion: {int(ortho_angle)} deg",
                    (knee[0] + 10, knee[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    # -----------------------
    # UI overlays (always drawn)
    # -----------------------
    if min_angle is not None:
        cv2.putText(frame, f"Min: {int(min_angle)} deg", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Max: {int(max_angle)} deg", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Progress bar
        progress = score_manager.progress_ratio()
        cv2.rectangle(frame, (BAR_X, BAR_Y),
                      (BAR_X + BAR_WIDTH, BAR_Y + BAR_HEIGHT),
                      (50, 50, 50), -1)
        cv2.rectangle(frame, (BAR_X, BAR_Y),
                      (BAR_X + int(BAR_WIDTH * progress), BAR_Y + BAR_HEIGHT),
                      (0, 200, 0), -1)
        cv2.rectangle(frame, (BAR_X, BAR_Y),
                      (BAR_X + BAR_WIDTH, BAR_Y + BAR_HEIGHT),
                      (255, 255, 255), 2)

        cv2.putText(frame, f"ROM: {int(score_manager.current_score)} deg",
                    (BAR_X, BAR_Y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Best: {int(score_manager.high_score)} deg",
                    (BAR_X + BAR_WIDTH + 10, BAR_Y + BAR_HEIGHT),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("ACL Angle Tracker", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        min_angle = None
        max_angle = None
        prev_knee = None
        score_manager.reset_session()
    if key == ord('q'):
        break

# -----------------------
# Cleanup
# -----------------------

cap.release()
cv2.destroyAllWindows()
