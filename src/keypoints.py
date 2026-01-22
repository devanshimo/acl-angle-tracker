import mediapipe as mp

mp_pose = mp.solutions.pose

def get_knee_points(landmarks, image_width, image_height, side="left"):
    """
    Returns hip, knee, ankle pixel coordinates for the given side.
    side: 'left' or 'right'
    """

    if side == "left":
        hip_id = mp_pose.PoseLandmark.LEFT_HIP
        knee_id = mp_pose.PoseLandmark.LEFT_KNEE
        ankle_id = mp_pose.PoseLandmark.LEFT_ANKLE
    else:
        hip_id = mp_pose.PoseLandmark.RIGHT_HIP
        knee_id = mp_pose.PoseLandmark.RIGHT_KNEE
        ankle_id = mp_pose.PoseLandmark.RIGHT_ANKLE

    def to_pixel(lm):
        return (
            int(lm.x * image_width),
            int(lm.y * image_height)
        )

    hip = to_pixel(landmarks[hip_id])
    knee = to_pixel(landmarks[knee_id])
    ankle = to_pixel(landmarks[ankle_id])

    return hip, knee, ankle
