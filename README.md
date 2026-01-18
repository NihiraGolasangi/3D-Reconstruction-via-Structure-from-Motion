# Buildings Built in Minutes: 3D Reconstruction via Structure-from-Motion

A classical SfM pipeline that reconstructs 3D building models from 2D image sequences. Implements fundamental matrix estimation, camera pose recovery, triangulation, and bundle adjustment to generate metric-accurate 3D scenes from monocular images.

**Key Components:**
- Feature matching with RANSAC-based outlier rejection
- Essential matrix decomposition for camera pose estimation
- Linear and non-linear triangulation with cheirality validation
- Perspective-n-Point (PnP) for view registration
- Sparse bundle adjustment for global optimization
