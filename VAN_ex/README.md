# Vision Aided Navigation - SLAM Project

**Amitai Ovadia - 312244254**

Date: August 24, 2024

[GitHub Repository](https://github.com/AmitaiOvadia/SLAMProject/tree/main/VAN_ex/code)


## Introduction

This project was completed as part of the course **67604 - SLAM Video Navigation** at the Hebrew University of Jerusalem, under the guidance of Mr. David Arnon and Dr. Refael Vivanti. The project tackles the **Simultaneous Localization And Mapping (SLAM)** problem, focusing on a pure computer vision approach without using additional sensors like LIDAR. It is based on the article **FrameSLAM: From bundle adjustment to real-time visual mapping** by Kurt Konolige and Motilal Agrawal.

The dataset used is the **KITTI Benchmark Suite**, capturing a vehicle's trajectory using grayscale stereo cameras in an urban environment. This README provides a detailed description of the methods, implementation, and results of the project.


### Main Steps of the Project

1. Estimating the trajectory using PnP (Perspective-n-Point).
2. Building a features tracking database.
3. Performing bundle adjustment optimizations over multiple windows.
4. Creating a Pose Graph.
5. Refining the Pose Graph using Loop Closures.


## KITTI Benchmark

The KITTI dataset is a collaboration between the Karlsruhe Institute of Technology and the Toyota Technological Institute at Chicago. It involves a vehicle equipped with stereo cameras, GPS, and LIDAR sensors traveling on streets in Germany. In this project, only the grayscale stereo cameras were used.

**Dataset Details:**
- Intrinsic camera matrix **K** for the stereo pair.
- Extrinsic camera matrix for the right camera relative to the left camera.
- 3360 frames of stereo image data.
- Ground truth extrinsic matrices for the entire drive.


![KITTI Frame Example](image_1_1.png)


## Camera Matrices

In the pinhole camera model, a camera is represented by extrinsic \([R|t]\) and intrinsic **K** matrices. The projection matrix **P = K [R|t]** maps 3D points to the 2D image plane.

**Extrinsic Matrices:**
- Left Camera Matrix (**M1**):
```
| 1 0 0 0 |
| 0 1 0 0 |
| 0 0 1 0 |
```
- Right Camera Matrix (**M2**):
```
| 1 0 0 -0.54 |
| 0 1 0 0     |
| 0 0 1 0     |
```
The right camera is displaced by 0.54 meters along the x-axis.

**Intrinsic Matrix:**
```
| 707 0 602 |
| 0 707 183 |
| 0   0   1 |
```


## 1. Estimating Trajectory using PnP

This step involves estimating the relative motion of the vehicle between consecutive frames using the PnP algorithm. Key steps include feature extraction, matching, triangulation, and estimation of relative transformations.

**Feature Extraction:**
- The **AKAZE** algorithm is used for robust feature extraction with memory-efficient binary descriptors.
- Feature matching between left and right images is filtered using epipolar constraints.

**Triangulation:**
Triangulation is performed using a linear method, solving for 3D points using SVD decomposition.
For more details, see the [Triangulation Code](https://github.com/AmitaiOvadia/SLAMProject/blob/main/VAN_ex/code/utils/utils.py#L176).


![PnP Process Illustration](image_3_1.png)


## 3. Bundle Adjustment

Bundle adjustment refines camera poses and 3D points by minimizing the reprojection error across multiple views. The dataset is divided into smaller segments called 'bundle windows' for efficient optimization.


## 4. Pose Graph Creation

A pose graph is constructed using relative camera poses. This graph is optimized to minimize pose estimation errors, providing a compact representation of the trajectory.


## 5. Loop Closures

Loop closures help correct drift by recognizing previously visited locations and adding constraints to the pose graph. This process reduces accumulated errors and improves trajectory accuracy.


## Performance Analysis

The SLAM system was evaluated on absolute and relative localization errors. Significant improvements were observed after bundle adjustment and loop closures, reducing the maximum localization error from 40 meters to around 5 meters.


## Conclusion

This project successfully demonstrated the effectiveness of a computer vision-based SLAM system using stereo camera data. The use of bundle adjustment and loop closures significantly reduced drift, resulting in accurate trajectory estimation.
