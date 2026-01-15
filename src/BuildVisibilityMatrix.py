import numpy as np


def BuildVisibilityMatrix(x_all, X_indices, n_cameras, n_points):
    """
    Build visibility matrix from correspondences.
    
    Args:
        x_all: List of Nx2 arrays, 2D observations for each camera
               x_all[i] contains the 2D points seen by camera i
        X_indices: List of N arrays, indices of 3D points corresponding to x_all
                   X_indices[i][j] is the index of the 3D point for x_all[i][j]
        n_cameras: Total number of cameras (I)
        n_points: Total number of 3D points (J)
        
    Returns:
        V: IxJ binary visibility matrix
    """
    V = np.zeros((n_cameras, n_points), dtype=int)
    
    for i in range(n_cameras):
        if i < len(X_indices) and X_indices[i] is not None:
            for point_idx in X_indices[i]:
                if 0 <= point_idx < n_points:
                    V[i, point_idx] = 1
    
    return V


def BuildVisibilityMatrixSimple(correspondences, n_cameras, n_points):
    """
    Simpler version: Build visibility from a list of correspondences.
    
    Args:
        correspondences: List of tuples (camera_idx, point_idx, u, v)
                        Each tuple represents that point_idx is visible 
                        at (u,v) in camera_idx
        n_cameras: Total number of cameras
        n_points: Total number of 3D points
        
    Returns:
        V: IxJ binary visibility matrix
    """
    V = np.zeros((n_cameras, n_points), dtype=int)
    
    for camera_idx, point_idx, u, v in correspondences:
        V[camera_idx, point_idx] = 1
    
    return V