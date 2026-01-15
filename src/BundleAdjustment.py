import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from src.NonlinearPnP import rotation_to_quaternion, quaternion_to_rotation





def bundle_adjustment_residuals(params, n_cameras, n_points, observations, K):
    camera_params = params[:7 * n_cameras].reshape((n_cameras, 7))
    point_params = params[7 * n_cameras:].reshape((n_points, 3))

    residuals = []

    for (i, j), (u_meas, v_meas) in observations.items():
        C_i = camera_params[i, :3]
        q_i = camera_params[i, 3:]
        q_i /= np.linalg.norm(q_i)
        R_i = quaternion_to_rotation(q_i)

        t_i = -R_i @ C_i
        X_j = point_params[j]

        X_cam = R_i @ X_j + t_i
        x_proj = K @ X_cam

        u_pred = x_proj[0] / x_proj[2]
        v_pred = x_proj[1] / x_proj[2]

        residuals.extend([
            u_meas - u_pred,
            v_meas - v_pred
        ])

    return np.array(residuals)


def BundleAdjustment(C_set, R_set, X_set, V, observations, K, fix_first_camera=True):
    """
    Perform bundle adjustment to refine camera poses and 3D points.
    
    Args:
        C_set: List of camera centers, each (3,)
        R_set: List of rotation matrices, each (3x3)
        X_set: Jx3 array of 3D points
        V: IxJ visibility matrix
        observations: Dict mapping (camera_idx, point_idx) -> (u, v)
        K: 3x3 camera intrinsic matrix
        fix_first_camera: If True, fix the first camera pose (recommended)
        
    Returns:
        C_set_refined: List of refined camera centers
        R_set_refined: List of refined rotation matrices
        X_set_refined: Jx3 array of refined 3D points
    """
    n_cameras = len(C_set)
    n_points = X_set.shape[0]
    
    print(f"Bundle Adjustment: {n_cameras} cameras, {n_points} points")
    print(f"Total observations: {len(observations)}")
    
    # Step 1: Convert rotations to quaternions and pack parameters
    params = []
    
    for i in range(n_cameras):
        C_i = C_set[i]
        R_i = R_set[i]
        q_i = rotation_to_quaternion(R_i)
        
        params.extend(C_i)  # Add camera center (3 values)
        params.extend(q_i)  # Add quaternion (4 values)
    
    # Add 3D points
    for j in range(n_points):
        params.extend(X_set[j])  # Add 3D point (3 values)
    
    params = np.array(params)
    
    print(f"Total parameters: {len(params)} (7*{n_cameras} + 3*{n_points})")
    
    # Step 2: Define which parameters to optimize
    if fix_first_camera:
        # Fix first camera by setting bounds
        # First 7 parameters (C1 and q1) are fixed
        lower_bounds = np.full(len(params), -np.inf)
        upper_bounds = np.full(len(params), np.inf)
        
        # Fix first camera parameters
        lower_bounds[:7] = params[:7]
        upper_bounds[:7] = params[:7]
        
        bounds = (lower_bounds, upper_bounds)
    else:
        bounds = (-np.inf, np.inf)
    
    # Step 3: Perform optimization
    print("Starting optimization...")
    
    result = least_squares(
        bundle_adjustment_residuals,
        params,
        args=(n_cameras, n_points, V, observations, K),
        method='trf',  # Trust Region Reflective
        loss='huber',  # Robust loss function
        bounds=bounds,
        verbose=2,
        max_nfev=100  # Maximum function evaluations
    )
    
    print(f"Optimization finished: {result.message}")
    print(f"Cost reduced from {result.cost:.6f} to final cost")
    
    # Step 4: Extract refined parameters
    refined_params = result.x
    
    # Extract camera parameters
    camera_params = refined_params[:7 * n_cameras].reshape((n_cameras, 7))
    C_set_refined = []
    R_set_refined = []
    
    for i in range(n_cameras):
        C_i = camera_params[i, :3]
        q_i = camera_params[i, 3:]
        R_i = quaternion_to_rotation(q_i)
        
        C_set_refined.append(C_i)
        R_set_refined.append(R_i)
    
    # Extract 3D points
    X_set_refined = refined_params[7 * n_cameras:].reshape((n_points, 3))
    
    return C_set_refined, R_set_refined, X_set_refined


