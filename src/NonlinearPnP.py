import numpy as np
from scipy.optimize import least_squares


def rotation_to_quaternion(R):
    """
    Convert rotation matrix to quaternion.
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        q: quaternion [q0, q1, q2, q3]
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q0 = 0.25 / s
        q1 = (R[2, 1] - R[1, 2]) * s
        q2 = (R[0, 2] - R[2, 0]) * s
        q3 = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q0 = (R[2, 1] - R[1, 2]) / s
        q1 = 0.25 * s
        q2 = (R[0, 1] + R[1, 0]) / s
        q3 = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q0 = (R[0, 2] - R[2, 0]) / s
        q1 = (R[0, 1] + R[1, 0]) / s
        q2 = 0.25 * s
        q3 = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q0 = (R[1, 0] - R[0, 1]) / s
        q1 = (R[0, 2] + R[2, 0]) / s
        q2 = (R[1, 2] + R[2, 1]) / s
        q3 = 0.25 * s
    
    return np.array([q0, q1, q2, q3])


def quaternion_to_rotation(q):
    """
    Convert quaternion to rotation matrix.
    
    Args:
        q: quaternion [q0, q1, q2, q3]
        
    Returns:
        R: 3x3 rotation matrix
    """
    # Normalize quaternion
    q = q / np.linalg.norm(q)
    q0, q1, q2, q3 = q
    
    # Convert to rotation matrix
    R = np.array([
        [1 - 2*(q2**2 + q3**2),     2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [    2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2),     2*(q2*q3 - q0*q1)],
        [    2*(q1*q3 - q0*q2),     2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])
    
    return R


def reprojection_residuals_pnp(params, x_world, x_image, K):
    """
    Compute reprojection residuals for PnP optimization.
    
    Args:
        params: [C1, C2, C3, q0, q1, q2, q3] - 7 parameters
        x_world: Nx3 array of 3D world points
        x_image: Nx2 array of 2D image points
        K: 3x3 camera intrinsic matrix
        
    Returns:
        residuals: 2N array of reprojection errors
    """
    # Extract camera center and quaternion
    C = params[:3]
    q = params[3:]
    
    # Convert quaternion to rotation matrix
    R = quaternion_to_rotation(q)
    
    # Build projection matrix P = K R [I | -C]
    t = -R @ C.reshape(3, 1)
    P = K @ np.hstack([R, t])
    
    # Compute residuals for all points
    residuals = []
    N = len(x_world)
    
    for i in range(N):
        # Convert 3D point to homogeneous coordinates
        X_hom = np.append(x_world[i], 1)
        
        # Project to image
        projected = P @ X_hom
        u_pred = projected[0] / projected[2]
        v_pred = projected[1] / projected[2]
        
        # Measured point
        u_measured, v_measured = x_image[i]
        
        # Compute residuals (measured - predicted)
        residuals.append(u_measured - u_pred)
        residuals.append(v_measured - v_pred)
    
    return np.array(residuals)


def nonlinear_pnp(K, x_world, x_image, C_init, R_init):
    """
    Refine camera pose using non-linear optimization to minimize reprojection error.
    
    Args:
        K: 3x3 camera intrinsic matrix
        x_world: Nx3 array of 3D world points
        x_image: Nx2 array of 2D image points
        C_init: Initial camera center (3,) from linear PnP
        R_init: Initial rotation matrix (3x3) from linear PnP
        
    Returns:
        C_refined: Refined camera center (3,)
        R_refined: Refined rotation matrix (3x3)
    """
    # Step 1: Convert initial rotation to quaternion
    q_init = rotation_to_quaternion(R_init)
    
    # Step 2: Pack initial parameters [C, q]
    params_init = np.hstack([C_init, q_init])  # 7 parameters
    
    # Step 3: Optimize using least squares
    result = least_squares(
        reprojection_residuals_pnp,
        params_init,
        args=(x_world, x_image, K),
        method='lm'  # Levenberg-Marquardt algorithm
    )
    
    # Step 4: Extract optimized parameters
    params_opt = result.x
    C_refined = params_opt[:3]
    q_refined = params_opt[3:]
    
    # Step 5: Convert quaternion back to rotation matrix
    R_refined = quaternion_to_rotation(q_refined)
    
    return C_refined, R_refined


