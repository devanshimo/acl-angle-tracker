
import cv2
from src.pose import PoseEstimator
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    pose_estimator = PoseEstimator()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = pose_estimator.process_frame(frame)

        if result.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        cv2.imshow("ACL Tracker - Pose", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
