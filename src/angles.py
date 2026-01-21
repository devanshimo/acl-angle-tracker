
import numpy as np
import math

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
