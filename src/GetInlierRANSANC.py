import numpy as np
from src.EstimateFundamentalMatrix import estimate_fundamental_matrix



def GetInlierRANSAC(pts1, pts2, threshold, M):
    """
    Estimates Fundamental Matrix using RANSAC.
    
    Args:
        pts1: Nx2 numpy array of (x, y) coordinates in image 1
        pts2: Nx2 numpy array of (x, y) coordinates in image 2
        threshold: Inlier threshold epsilon for epipolar constraint
        M: Number of RANSAC iterations
        
    Returns:
        F: 3x3 Fundamental Matrix with maximum inliers
    """
    N= len(pts1)
    best_inliers = [] 
    best_F = None


    for _ in range(M):
        # Randomly sample 8 points
        indices = np.random.choice(N, 8, replace=False)
        sample_pts1 = pts1[indices]
        sample_pts2 = pts2[indices]

        # Estimate Fundamental Matrix from the sampled points
        F_candidate = estimate_fundamental_matrix(sample_pts1, sample_pts2)

        # Count inliers
        inliers = []
        for i in range(N):
            pt1_hom = np.array([pts1[i, 0], pts1[i, 1], 1])
            pt2_hom = np.array([pts2[i, 0], pts2[i, 1], 1])
            error = abs(pt2_hom.T @ F_candidate @ pt1_hom)
            #normalize the error
            l2 = F_candidate @ pt1_hom #point to epipolar line distance
            error = error / np.sqrt(l2[0]**2 + l2[1]**2)

            
            if error < threshold:
                inliers.append(i)

        # Update best inliers if current is better
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_F = F_candidate

    best_F = best_F / best_F[2,2]  # Normalize F
    return best_F, best_inliers


