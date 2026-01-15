import numpy as np



def linear_pnp(K, x_world, x_image):
    """
    Linear Perspective-n-Point (PnP) algorithm to estimate camera pose.

    Args:
        K: 3x3 camera intrinsic matrix
        x_world: Nx3 3D world points
        x_image: Nx2 2D image points"""
    
    N = x_world.shape[0]
    # Convert 2D points to homogeneous coordinates
    x_hom = np.hstack((x_image, np.ones((N, 1))))  # (N, 3)

    # Normalize using intrinsic matrix
    K_inv = np.linalg.inv(K)
    x_norm = (K_inv @ x_hom.T).T  # (N, 3)

    # Construct matrix A
    A = []
    for i in range(N):
        X, Y, Z = x_world[i]
        u, v, _ = x_norm[i]

        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v])          
    A = np.array(A)  # (2N, 12) 

    # Solve for P using SVD
    U, S, Vt = np.linalg.svd(A)
    P_vec = Vt[-1]  # (12,)
    P = P_vec.reshape(3, 4)  # (3, 4)   

    # Extract R and t from P
    R_est = P[:, :3]
    t_est = P[:, 3]

    #normalize R    
    scale = np.mean([
    np.linalg.norm(R_est[0]),
    np.linalg.norm(R_est[1]),
    np.linalg.norm(R_est[2])])
    R_est /= scale
    t_est /= scale

    # Ensure R is a valid rotation matrix using SVD
    U_r, S_r, Vt_r = np.linalg.svd(R_est)
    R = U_r @ Vt_r
    if np.linalg.det(R) < 0:
        R = -R
        t_est = -t_est

    # Compute camera center C
    C = -R.T @ t_est    
    return C, R

