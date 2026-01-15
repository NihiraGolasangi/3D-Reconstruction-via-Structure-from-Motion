import numpy as np

def estimate_fundamental_matrix(pts1, pts2):
    """
    Estimate the fundamental matrix from corresponding points in two images using the normalized 8-point algorithm.

    Parameters:
    pts1 : ndarray
        An Nx2 array of points from the first image.
    pts2 : ndarray
        An Nx2 array of points from the second image.

    Returns:
    F : ndarray
        A 3x3 fundamental matrix.
    """
    assert pts1.shape == pts2.shape, "Point arrays must have the same shape."
    assert pts1.shape[0] >= 8, "At least 8 point correspondences are required."

    # Normalize points
    def normalize_points(pts):
        mean = np.mean(pts, axis=0)
        std = np.std(pts, axis=0).mean()
        scale = np.sqrt(2) / std
        T = np.array([[scale, 0, -scale * mean[0]],
                      [0, scale, -scale * mean[1]],
                      [0, 0, 1]])
        pts_homogeneous = np.hstack((pts, np.ones((pts.shape[0], 1))))
        normalized_pts = (T @ pts_homogeneous.T).T
        return normalized_pts[:, :2], T

    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)

    # Construct matrix A for the equation Ax=0
    N = pts1_norm.shape[0]
    A = np.zeros((N, 9))
    for i in range(N):
        x1, y1 = pts1_norm[i]
        x2, y2 = pts2_norm[i]
        A[i] = [x2 * x1, x2 * y1, x2,
                 y2 * x1, y2 * y1, y2,
                 x1, y1, 1]

    # Solve for f using SVD
    U, S, Vt = np.linalg.svd(A)
    F_vec = Vt[-1]
    F = F_vec.reshape(3, 3)

    # Enforce rank-2 constraint on F
    U_f, S_f, Vt_f = np.linalg.svd(F)
    S_f[2] = 0  # Set the smallest singular value to zero
    F_rank2 = U_f @ np.diag(S_f) @ Vt_f
    F_final = T2.T @ F_rank2 @ T1
    return F_final