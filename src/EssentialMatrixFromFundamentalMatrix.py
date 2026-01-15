import numpy as np


def esssential_matrix_from_fundamental_matrix(F, K1, K2):
    """
    Compute the essential matrix from the fundamental matrix and camera intrinsic matrices.

    Parameters:
    F : ndarray
        A 3x3 fundamental matrix.
    K1 : ndarray
        A 3x3 intrinsic matrix for the first camera.
    K2 : ndarray
        A 3x3 intrinsic matrix for the second camera.

    Returns:
    E : ndarray
        A 3x3 essential matrix.
    """
    
    E = K2.T @ F @ K1

    U,S,V_t = np.linalg.svd(E)
    S = np.array([1, 1, 0]) # Enforce the singular values to be (1, 1, 0)
    E = U @ np.diag(S) @ V_t
    return E