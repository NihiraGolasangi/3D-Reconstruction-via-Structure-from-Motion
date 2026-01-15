import numpy as np


def disambiguate_camera_pose(C_set, R_set, X_set):
    """
    Disambiguates among possible camera poses based on the number of points in front of the camera.

    Args:
        possible_poses (list): List of possible camera poses (R, t).
        points_3d (numpy.ndarray): 3D points in the world coordinate system.

    Returns:
        tuple: The correct camera pose (R, t).
    """
    best_count = 0
    best_C = None
    best_R = None
    best_X = None

    C1 = np.zeros(3)
    R1 = np.eye(3)  

    for i in range(4):
        # Get the i-th configuration
        C2 = C_set[i]
        R2 = R_set[i]
        X = X_set[i]

        # Count points in front of both cameras
        count = 0
        for X_point in X:
            # Check for camera 1
            X_cam1 = R1 @ (X_point - C1)
            in_front_cam1 = X_cam1[2] > 0

            # Check for camera 2
            X_cam2 = R2 @ (X_point - C2)
            in_front_cam2 = X_cam2[2] > 0

            if in_front_cam1 and in_front_cam2:
                count += 1

        if count > best_count:
            best_count = count
            best_C = C2
            best_R = R2
            best_X = X
    return best_C, best_R, best_X

        
  