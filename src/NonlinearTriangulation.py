import numpy as np
from scipy.optimize import least_squares



def NonlinearTriangulation(K, C1, R1, C2, R2, x1, x2, X_init):
    """
    Refine 3D points using non-linear optimization to minimize reprojection error.
    
    Args:
        K: 3×3 camera intrinsic matrix
        C1: Camera 1 center (3×1)
        R1: Camera 1 rotation matrix (3×3)
        C2: Camera 2 center (3×1)
        R2: Camera 2 rotation matrix (3×3)
        x1: Nx2 image points from camera 1
        x2: Nx2 image points from camera 2
        X_init: Nx3 initial 3D points (from linear triangulation)
        
    Returns:
        X_refined: Nx3 refined 3D points
    """

    P1 = K @ np.hstack([R1, (-R1 @ C1).reshape(3, 1)])  
    P2 = K @ np.hstack([R2, (-R2 @ C2).reshape(3, 1)])

    N = len(X_init)
    X_refined = []

    for i in range(N):
        u1, v1 = x1[i]
        u2, v2 = x2[i]
        X0 = X_init[i]


        # Optimize using least squares
        # Alternative (slightly more efficient):
        res = least_squares(reprojection_residuals, X0, args=(P1, P2, u1, v1, u2, v2))
        X_refined.append(res.x)

    return np.array(X_refined)




def reprojection_residuals(X, P1, P2, u1, v1, u2, v2):
    """Compute reprojection residuals for a single 3D point."""
    
    # Convert to homogeneous coordinates
    X_homogeneous = np.append(X, 1)
    
    # Project to camera 1
    proj1 = P1 @ X_homogeneous
    u1_pred = proj1[0] / proj1[2]
    v1_pred = proj1[1] / proj1[2]
    
    # Project to camera 2
    proj2 = P2 @ X_homogeneous
    u2_pred = proj2[0] / proj2[2]
    v2_pred = proj2[1] / proj2[2]
    
    # Compute residuals (measured - predicted)
    residuals = np.array([
        u1 - u1_pred,
        v1 - v1_pred,
        u2 - u2_pred,
        v2 - v2_pred
    ])
    
    return residuals