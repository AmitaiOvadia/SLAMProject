# Vision Aided Navigation - SLAM Project

**Amitai Ovadia - 312244254**

Date: August 24, 2024

[GitHub Repository](https://github.com/AmitaiOvadia/SLAMProject/tree/main/VAN_ex/code)


## Introduction

This project was completed as part of the course **67604 - SLAM Video Navigation** at the Hebrew University of Jerusalem, guided by Mr. David Arnon and Dr. Refael Vivanti.
SLAM, short for **Simultaneous Localization And Mapping**, involves constructing or updating a map of an unknown environment while simultaneously tracking the agent’s location within it. SLAM techniques are used in applications such as self-driving cars, UAVs, autonomous underwater vehicles, and even domestic robots.

This project is based primarily on the article **FrameSLAM: From bundle adjustment to real-time visual mapping** by Kurt Konolige and Motilal Agrawal. It tackles the SLAM problem as a pure computer vision task without utilizing other sensors like LIDAR. The project uses the **KITTI Benchmark Suite**, focusing on trajectory estimation of a vehicle using grayscale stereo camera data in an urban setting in Germany.

Below are the main steps implemented in this project:
1. Finding an estimated trajectory using PnP - a deterministic approach
2. Building a features tracking database
3. Performing bundle adjustment optimizations over multiple windows
4. Creating a Pose Graph
5. Refining the Pose Graph using Loop Closures


## KITTI Benchmark

The KITTI dataset is a collaboration between the Karlsruhe Institute of Technology and the Toyota Technological Institute at Chicago. It involves a vehicle equipped with sensors (stereo cameras, GPS, and LIDAR) traveling on streets in Germany. For this project, only the grayscale stereo cameras were used.

The dataset provides:
- Intrinsic camera matrix **K** for the stereo pair
- Extrinsic camera matrix for the right camera with respect to the left camera
- 3360 frames of stereo image data
- Ground truth extrinsic matrices for the entire drive


![KITTI Frame Example](image_1_1.png)

## Camera Matrices

A camera, modeled as a pinhole camera, is described by extrinsic \([R|t]\) and intrinsic **K** matrices. The projection matrix **P = K [R|t]** projects a 3D point **X** in homogeneous coordinates onto the image plane.

**Extrinsic Camera Matrix**
The extrinsic matrix describes the rotation and translation from the global coordinate system to the camera coordinate system. For the left and right cameras in this project:

Left Camera Matrix (**M1**):
```
| 1 0 0 0 |
| 0 1 0 0 |
| 0 0 1 0 |
```
Right Camera Matrix (**M2**):
```
| 1 0 0 -0.54 |
| 0 1 0 0     |
| 0 0 1 0     |
```
The right camera is displaced by 0.54 meters along the x-axis.


**Intrinsic Camera Matrix**

The intrinsic matrix projects a 3D point in the camera coordinate system to a pixel on the image plane:

```
| 707 0 602 |
| 0 707 183 |
| 0   0   1 |
```

## 1. Finding an Estimated Trajectory using PnP

In this section, we estimate the relative movement of the car between consecutive frames using PnP. This involves feature extraction, matching, and triangulation of points between stereo image pairs. The following steps were implemented:

- Feature extraction using the **AKAZE** algorithm for robust and memory-efficient descriptors.
- Matching left and right image descriptors and filtering based on epipolar constraints.

![PnP Illustration](image_3_1.png)

## 1. Finding an Estimated Trajectory using PnP

This section involves estimating the vehicle’s relative movement between consecutive frames using the PnP algorithm. Key steps include feature extraction, matching, triangulation, and relative movement estimation.

**Feature Extraction and Matching**
- Features are extracted using the AKAZE algorithm due to its robust and memory-efficient binary descriptors.
- The extracted features are matched between the left and right images using OpenCV, and epipolar constraints are applied for filtering.
- Matches are filtered based on y-coordinate differences (threshold: 1.5 pixels).

For implementation details, refer to the following code sections:
- [Feature Extraction](https://github.com/AmitaiOvadia/SLAMProject/blob/main/VAN_ex/code/utils/utils.py#L66)
- [Matching](https://github.com/AmitaiOvadia/SLAMProject/blob/main/VAN_ex/code/utils/utils.py#L116)

![PnP Process Illustration](image_3_1.png)

### Triangulation and PnP Estimation

The matched points are triangulated using linear triangulation to obtain 3D points. This involves solving a homogeneous linear system using SVD decomposition. The PnP algorithm then estimates the relative rotation and translation between two consecutive frames using these 3D points.

For more details, refer to the [Triangulation Code](https://github.com/AmitaiOvadia/SLAMProject/blob/main/VAN_ex/code/utils/utils.py#L176).

![Triangulation Illustration](image_3_2.png)

### RANSAC for Outlier Removal

To ensure accurate relative movement estimation, the RANSAC algorithm is used for outlier detection. RANSAC iteratively selects a subset of data points, fits a model, and evaluates inliers based on the reprojection error threshold (1.5 pixels).

For implementation, see [RANSAC Code](https://github.com/AmitaiOvadia/SLAMProject/blob/main/VAN_ex/code/ex3/Ex3.py#L141).

## 2. Building a Features Tracking Database

This step involves creating a database to track landmarks across frames, allowing efficient retrieval of tracked points. The database handles feature extraction, matching, and outlier detection using RANSAC.

For code implementation, refer to [Tracking Database Code](https://github.com/AmitaiOvadia/SLAMProject/blob/main/VAN_ex/code/utils/tracking_database.py).

![Feature Tracking Illustration](image_5_1.png)

## 3. Performing Bundle Adjustment over Multiple Windows

Bundle adjustment is performed to refine camera poses and 3D points by minimizing the reprojection error. Due to the sparsity of the problem, the sequence is divided into smaller 'bundle windows' for efficient optimization.

Key implementation details can be found in the [Bundle Adjustment Code](https://github.com/AmitaiOvadia/SLAMProject/blob/main/VAN_ex/code/utils/BundleAdjusment.py).

![Bundle Adjustment Result](image_5_2.png)

## 4. Pose Graph Creation

A pose graph is constructed to represent the relative camera poses between key frames. This compact representation is optimized to refine the estimated trajectory.

For implementation, refer to [Pose Graph Code](https://github.com/AmitaiOvadia/SLAMProject/blob/main/VAN_ex/code/utils/PoseGraph.py).

## 5. Refining the Pose Graph using Loop Closures

Loop closures are used to correct drift by identifying previously visited locations and adding constraints to the pose graph. This helps in reducing cumulative errors in the estimated trajectory.

Implementation details can be found in [Loop Closure Code](https://github.com/AmitaiOvadia/SLAMProject/blob/main/VAN_ex/code/utils/PoseGraph.py).

## Performance Analysis

Quantitative evaluation of the SLAM system shows improvements in absolute and relative localization errors after bundle adjustment and loop closures. The maximum localization error was reduced from 40 meters to around 5 meters after applying loop closures.


![Absolute Localization Error](image_7_1.png)

![Relative Error Analysis](image_7_2.png)

## Conclusion

The SLAM system demonstrated significant improvements in trajectory estimation using computer vision techniques. The use of bundle adjustment and loop closures effectively reduced the drift and enhanced the accuracy of the final trajectory estimation.
