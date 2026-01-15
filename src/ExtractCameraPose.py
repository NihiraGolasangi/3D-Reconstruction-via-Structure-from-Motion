import numpy as np

def extract_camera_pose(E):
    """
    Extracts the camera pose (R, t) from the Essential matrix E.
    """
    
    # Perform SVD of E
    U, S, Vt = np.linalg.svd(E)

     # Define W matrix
    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])

    # Possible rotation matrices
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    # Possible translation vectors (up to scale)
    C1 = U[:, 2]
    C2 = -U[:, 2]

    configurations = [(C1, R1), (C1, R2), (C2, R1), (C2, R2)]
    C_set = []
    R_set = []

    for C, R in configurations:
        if np.linalg.det(R) < 0:
            R = -R
            C = -C
        C_set.append(C)
        R_set.append(R)

    return C_set, R_set