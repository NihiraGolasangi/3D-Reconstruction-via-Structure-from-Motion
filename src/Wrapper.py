import numpy as np
import sys
import datetime
import os
import cv2


from src.NonlinearTriangulation import reprojection_residuals

from src.GetInlierRANSANC import GetInlierRANSAC
from src.EssentialMatrixFromFundamentalMatrix import esssential_matrix_from_fundamental_matrix
from src.ExtractCameraPose import extract_camera_pose
from src.LinearTriangulation import LinearTriangulation
from src.DisambiguateCameraPose import disambiguate_camera_pose
from src.NonlinearTriangulation import NonlinearTriangulation
# from src.BuildVisibilityMatrix import BuildVisibilityMatrix 
# from src.BundleAdjustment import BundleAdjustment
# from NonlinearPnP import nonlinear_pnp


num_images = 6


# Logger setup
log_folder = "logs"
os.makedirs(log_folder, exist_ok=True)
log_filename = os.path.join(
    log_folder,
    f"sfm_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
log_file = open(log_filename, "w")

class Logger:
    def __init__(self, file):
        self.file = file
    def write(self, msg):
        self.file.write(msg)
        self.file.flush()  # write immediately
    def flush(self):
        pass

# sys.stdout = Logger(log_file)
import numpy as np

def compute_reprojection_error(X_3D, P1, P2, x1, x2):
    """
    Compute the reprojection error for all 3D points.
    
    Args:
        X_3D: Nx3 array of 3D points
        P1, P2: 3x4 camera projection matrices
        x1, x2: Nx2 arrays of corresponding 2D points in image 1 and 2
    
    Returns:
        errors: Nx1 array of per-point reprojection errors
        mean_error: mean reprojection error
        median_error: median reprojection error
    """
    errors = []
    
    for i in range(len(X_3D)):
        residuals = reprojection_residuals(X_3D[i], P1, P2, x1[i,0], x1[i,1], x2[i,0], x2[i,1])
        # Euclidean distance in pixels (sqrt of sum of squared residuals)
        error = np.linalg.norm(residuals[:2]) + np.linalg.norm(residuals[2:])
        errors.append(error)
    
    errors = np.array(errors)
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    
    return errors, mean_error, median_error



def StructureFromMotion(image_matches, K, n_images=2):
    """
    Incremental Structure-from-Motion pipeline.

    Args:
        image_matches: dictionary or list containing feature matches between images
        K: 3x3 camera intrinsic matrix
        n_images: number of images to process (default: 2)

    Returns:
        X: Nx3 array of reconstructed 3D points
        Cset: list of camera centers
        Rset: list of rotation matrices
    """

    # ----------------------------------
    # 1. Initialize with first two images
    # ----------------------------------
    x1, x2 = image_matches[(1,2)]['pts1'], image_matches[(1,2)]['pts2']

    F, inliers = GetInlierRANSAC(x1, x2, threshold=1, M=1000)
    print(f"Estimated Fundamental Matrix:\n{F}")
    E = esssential_matrix_from_fundamental_matrix(F, K, K)
    print(f"Estimated Essential Matrix:\n{E}")

    C_candidates, R_candidates = extract_camera_pose(E)
    print(f"Extracted {len(C_candidates)} candidate camera poses.") 
    print("Candidate Camera Poses (C, R):")
    for i, (C, R) in enumerate(zip(C_candidates, R_candidates)):
        print(f"Candidate {i+1}:")
        print(f"C:\n{C}")
        print(f"R:\n{R}")

   
   #lets check F and E using openCV (for debugging purposes)
    x1_cv = x1.astype(np.float32)
    x2_cv = x2.astype(np.float32)
    F_cv, mask = cv2.findFundamentalMat(x1_cv, x2_cv, cv2.FM_RANSAC)
    E_cv, mask = cv2.findEssentialMat(x1_cv, x2_cv, K, cv2.RANSAC)
    print(f"OpenCV Estimated Fundamental Matrix:\n{F_cv}")
    print(f"OpenCV Estimated Essential Matrix:\n{E_cv}")


    # homogeneous coordinates
    pts1_h = np.hstack([x1, np.ones((x1.shape[0],1))])
    pts2_h = np.hstack([x2, np.ones((x2.shape[0],1))])

    errors_custom = np.abs(np.sum(pts2_h * (F @ pts1_h.T).T, axis=1))
    errors_cv     = np.abs(np.sum(pts2_h * (F_cv @ pts1_h.T).T, axis=1))

    print("Median epipolar error (custom F):", np.median(errors_custom))
    print("Median epipolar error (OpenCV F):", np.median(errors_cv))


    # Normalize by camera intrinsics
    pts1_norm = (np.linalg.inv(K) @ pts1_h.T).T
    pts2_norm = (np.linalg.inv(K) @ pts2_h.T).T

    # Epipolar distances for custom E
    l2 = E @ pts1_norm.T
    errors_custom = np.abs(np.sum(pts2_norm * l2.T, axis=1)) / np.sqrt(l2[0,:]**2 + l2[1,:]**2)

    # Epipolar distances for OpenCV E
    l2_cv = E_cv @ pts1_norm.T
    errors_cv = np.abs(np.sum(pts2_norm * l2_cv.T, axis=1)) / np.sqrt(l2_cv[0,:]**2 + l2_cv[1,:]**2)

    print("Median epipolar error (custom E):", np.median(errors_custom))
    print("Median epipolar error (OpenCV E):", np.median(errors_cv))

  

    # ----------------------------------
    # 2. Initial 3D Reconstruction and Camera Pose Disambiguation
    # ----------------------------------
    X_candidates = []
    for C2, R2 in zip(C_candidates, R_candidates):
        # Triangulate 3D points for this candidate
        X = LinearTriangulation(
            K,
            np.zeros(3), np.eye(3),
            C2, R2,
            x1, x2)
        X_candidates.append(X)

    # Disambiguate to select the correct camera pose
    C_correct, R_correct, X_init = disambiguate_camera_pose(C_candidates, R_candidates, X_candidates)

    print("Selected Camera 2 pose:")
    print("C:", C_correct)
    print("R:", R_correct)
    print("Number of valid 3D points:", X_init.shape[0])

    # ----------------------------------
    # 5. Nonlinear triangulation
    # ----------------------------------
    X_refined = NonlinearTriangulation(
    K,
    np.zeros(3),
    np.eye(3),
    C_correct,
    R_correct,
    x1,
    x2,
    X_init
)
    Cset = [np.zeros(3), C]
    Rset = [np.eye(3), R]
    print("Refined 3D points using Nonlinear Triangulation.")   



    # # ----------------------------------
    # # 6. Register remaining images
    # # ----------------------------------
    
    # for i in range(2, num_images):

    #     # Get 2D–3D correspondences
    #     x_i, X_visible = Get2D3DCorrespondences(X, i, inlier_matches)

    #     # PnP with RANSAC
    #     C_new, R_new = PnPRANSAC(X_visible, x_i, K)

    #     # Nonlinear PnP refinement
    #     C_new, R_new = nonlinear_pnp(X_visible, x_i, K, C_new, R_new)

    #     Cset.append(C_new)
    #     Rset.append(R_new)

    #     # ----------------------------------
    #     # 7. Add new 3D points
    #     # ----------------------------------
    #     x_prev, x_curr = GetNewMatches(i, inlier_matches)

    #     X_new = LinearTriangulation(
    #         K,
    #         C0=Cset[0],
    #         R0=Rset[0],
    #         C1=C_new,
    #         R1=R_new,
    #         x1=x_prev,
    #         x2=x_curr
    #     )

    #     X_new = NonlinearTriangulation(
    #         K,
    #         C0=Cset[0],
    #         R0=Rset[0],
    #         C1=C_new,
    #         R1=R_new,
    #         x1=x_prev,
    #         x2=x_curr,
    #         X0=X_new
    #     )

    #     X = np.vstack([X, X_new])

    #     # ----------------------------------
    #     # 8. Bundle Adjustment
    #     # ----------------------------------
    #     V = BuildVisibilityMatrix(traj=inlier_matches)

    #     Cset, Rset, X = BundleAdjustment(
    #         Cset, Rset, X, K, traj=inlier_matches, V=V
    #     )

    # return X, Cset, Rset
