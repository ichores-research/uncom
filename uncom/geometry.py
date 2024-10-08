import numpy as np
import numpy.typing as npt


def straight_from_points(point_a: npt.NDArray, point_b: npt.NDArray) -> float:
    """
    Calculate the straight that passes through two points.

    Args:
        point_a: First point.
        point_b: Second point.

    Returns:
        The slope and intercept of the straight that passes through the two points.
    """
    slope = (point_b[1] - point_a[1]) / (point_b[0] - point_a[0])
    intercept = point_a[1] - slope * point_a[0]
    return np.array([slope, intercept])


def distance_point_straight(point: npt.NDArray, straight: npt.NDArray) -> float:
    """
    Calculate the distance between a point and a straight.
    """
    # Original straight
    slope, intercept = straight

    # Point
    C_x, C_y = point

    # Straight perpendicular to straight
    a_C = -(slope**-1)
    b_C = C_y - a_C * C_x

    # Intersection point
    Ci_x = (b_C - intercept) / (slope - a_C)
    Ci_y = a_C * Ci_x + b_C

    # Distance
    return ((C_x - Ci_x) ** 2 + (C_y - Ci_y) ** 2) ** 0.5


def rotation_angle(point_a: npt.NDArray, point_b: npt.NDArray) -> float:
    """
    Calculate the angle of rotation between two points.
    """
    return np.arctan2(point_b[1] - point_a[1], point_b[0] - point_a[0])


def rotate_point(point: npt.NDArray, angle: float) -> npt.NDArray:
    """
    Rotate a point by a given angle.
    """
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    # Rotation matrix
    R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    # Point
    point = np.array(point)

    # Rotate point
    A_rotated = R @ point

    return (A_rotated[0], A_rotated[1])


def points_straight_distance(points: npt.NDArray, straight: npt.NDArray) -> npt.NDArray:
    """
    Calculate the distance between a set of points and a straight.
    """
    theta = np.arctan(straight[0])
    translation = np.array([0, -straight[1]])

    points_translated = points + translation
    points_rotated = rotate_points(points_translated, -theta)

    return np.abs(points_rotated[:, 1])


def rotate_points(points: npt.NDArray, angle: float) -> npt.NDArray:
    """
    Rotate a set of points by a given angle.
    """
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    # Rotation matrix
    R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    # Rotate points
    points_rotated = R @ points.T

    return points_rotated.T



def third_point(
    point_a: npt.NDArray, point_b: npt.NDArray, distance: float
) -> npt.NDArray:
    """
    Calculate the third point of a triangle given two points and the distance to the third one.
    It extends the line formed by the two points by the given distance.

    Args:
        point_a: First point.
        point_b: Second point.
        distance: Distance from the second point to the third one.

    Returns:
        The third point.
    """
    # Calculate the direction vector
    direction = point_b - point_a
    # Calculate the length of the direction vector
    length = np.linalg.norm(direction)
    # Normalize the direction vector to get the unit vector
    unit_vector = direction / length
    # Scale the unit vector by the desired distance
    scaled_vector = unit_vector * distance
    # Calculate the new point
    point_c = point_b + scaled_vector

    return point_c
