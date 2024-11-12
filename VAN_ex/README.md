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
