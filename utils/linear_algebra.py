import numpy as np


def cosine_between(v1: np.array, v2: np.array) -> float:
    """Returns the angle in radians between the two vector"""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)

    return np.dot(v1_u, v2_u)
