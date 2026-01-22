
import numpy as np
import math
from src.keypoints import get_knee_points


def compute_angle(a, b, c):
    """
    Computes angle ABC (in degrees)
    a, b, c are (x, y) tuples
    """
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)

    cosine_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc)
    )

    # numerical safety
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = math.degrees(math.acos(cosine_angle))
    return angle

from collections import deque

class AngleSmoother:
    def __init__(self, window_size=5):
        self.window = deque(maxlen=window_size)

    def smooth(self, angle):
        self.window.append(angle)
        return sum(self.window) / len(self.window)

