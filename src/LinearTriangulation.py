import numpy as np



def LinearTriangulation(K, C1, R1, C2, R2, x1, x2):
    """
    Triangulate 3D points from correspondences.
    
    Args:
        K: 3×3 camera intrinsic matrix
        C1, R1: Camera 1 pose
        C2, R2: Camera 2 pose
        x1: Nx2 image points from camera 1
        x2: Nx2 image points from camera 2
        
    Returns:
        X: Nx3 array of 3D points
    """
    
    # Step 1: Build projection matrices
    P1 = K @ np.hstack([R1, (-R1 @ C1).reshape(3, 1)])
    P2 = K @ np.hstack([R2, (-R2 @ C2).reshape(3, 1)])
    
    N = len(x1)
    X_3D = []
    
    # Step 2: For each correspondence
    for i in range(N):
        u1, v1 = x1[i]
        u2, v2 = x2[i]
        
        # Step 3: Build A matrix (4×4)
        A = np.array([
            u1 * P1[2, :] - P1[0, :],
            v1 * P1[2, :] - P1[1, :],
            u2 * P2[2, :] - P2[0, :],
            v2 * P2[2, :] - P2[1, :]
        ])
        
        # Step 4: Solve using SVD
        U, S, Vt = np.linalg.svd(A)
        X_homogeneous = Vt[-1]
        
        # Step 5: Convert to 3D
        X_point = X_homogeneous[:3] / X_homogeneous[3]
        X_3D.append(X_point)
    
    return np.array(X_3D)