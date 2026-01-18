
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np  

def set_equal_3d_axes(ax, X, Cset, margin=0.1):
    """
    Set equal aspect ratio and tight bounds for 3D plot.
    """
    all_points = np.vstack([X] + Cset)

    min_vals = all_points.min(axis=0)
    max_vals = all_points.max(axis=0)

    center = (min_vals + max_vals) / 2.0
    extent = (max_vals - min_vals).max() * (1 + margin)

    ax.set_xlim(center[0] - extent/2, center[0] + extent/2)
    ax.set_ylim(center[1] - extent/2, center[1] + extent/2)
    ax.set_zlim(center[2] - extent/2, center[2] + extent/2)


def visualize_reconstruction(X, Cset, Rset, colors=None):
    """
    Visualize reconstruction with any number of cameras.
    
    Args:
        X: Nx3 array of 3D points
        Cset: List of camera centers [C1, C2, C3, ...]
        Rset: List of rotation matrices [R1, R2, R3, ...]
        colors: Nx3 RGB colors (optional)
    """

    
    os.makedirs("results", exist_ok=True)
    n_cameras = len(Cset)
    
    # 3D view
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    if colors is not None:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors/255.0, marker='.', s=1, alpha=0.5)
    else:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='blue', marker='.', s=1, alpha=0.5)
    
    # Plot all cameras
    for i, (C, R) in enumerate(zip(Cset, Rset)):
        ax.scatter(C[0], C[1], C[2], c='red', marker='o', s=20, 
                  edgecolors='black', linewidths=2, zorder=10)
        ax.text(C[0], C[1], C[2], f'  {i+1}', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{n_cameras}-Camera Reconstruction ({len(X)} points)')
    set_equal_3d_axes(ax, X, Cset)
    plt.savefig(f"results/reconstruction_3d_{n_cameras}cams.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # Top view (X-Z)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot points
    if colors is not None:
        ax.scatter(X[:, 0], X[:, 2], c=colors/255.0, marker='.', s=5, alpha=0.5)
    else:
        ax.scatter(X[:, 0], X[:, 2], c='blue', marker='.', s=5, alpha=0.5)
    
    # Plot all cameras
    for i, (C, R) in enumerate(zip(Cset, Rset)):
        ax.scatter(C[0], C[2], c='red', marker='o', s=30, 
                  edgecolors='black', linewidths=2, zorder=10)
        ax.text(C[0], C[2], f'  {i+1}', fontsize=14, fontweight='bold')
        
        # Camera viewing direction
        direction = R[2, :]
        scale = 2.0
        ax.arrow(C[0], C[2], -direction[0]*scale, -direction[2]*scale,
                head_width=0.3, head_length=0.2, fc='red', ec='red', 
                alpha=0.7, linewidth=2, zorder=9)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title(f'Top View - {n_cameras} Cameras ({len(X)} points)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    plt.savefig(f"results/reconstruction_topview_{n_cameras}cams.png", dpi=200, bbox_inches='tight')
    plt.close()



