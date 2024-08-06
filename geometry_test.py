from math import sqrt
import numpy as np

from uncom.geometry import distance_point_straight, points_straight_distance, rotate_point, rotate_points


if __name__ == "__main__":
    s = (-1 / 3, 2)
    A = (-2, -4)

    # Known correct solution
    print(2 * sqrt(10))

    # Pythagorean solution
    print(distance_point_straight(A, s))

    # Rotation solution
    theta = np.arctan(s[0])
    At = (A[0], A[1] - s[1])
    Ar = rotate_point(At, -theta)
    print(abs(Ar[1]))

    # Rotation solution for multiple points
    points = np.array([A, A, A])
    translation = np.array([0, -s[1]])
    points_translated = points + translation
    points_rotated = rotate_points(points_translated, -theta)
    print(np.abs(points_rotated[:, 1]))

    # Distance from points to straight
    print(points_straight_distance(points, s))
    print(np.argmax(points_straight_distance(points, s)))
