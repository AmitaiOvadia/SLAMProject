# 🚗 **Vision Aided Navigation 2024 - Final Project** 📷

## 🗺️ Introduction

Welcome to the **Vision Aided Navigation 2024 - Final Project**! This project focuses on integrating computer vision techniques with navigation systems to enhance real-time localization and mapping capabilities. By leveraging visual data from cameras, we aim to develop robust algorithms that support accurate navigation for autonomous vehicles and other mobile platforms.

---

This project was completed as part of the course **"67604 SLAM - Video Navigation"** at The Hebrew University of Jerusalem, taught by Mr. David Arnon and Dr. Refael Vivanti.

**SLAM**, short for **Simultaneous Localization And Mapping**, involves constructing or updating a map of an unknown environment while simultaneously tracking an agent's location within it. SLAM techniques are widely used in self-driving cars, unmanned aerial vehicles, autonomous underwater vehicles, planetary rovers, advanced domestic robots, and even in medical applications inside the human body.

Our implementation is primarily based on the article **"FrameSLAM: From Bundle Adjustment to Real-Time Visual Mapping"** by Kurt Konolige and Motilal Agrawal. This project addresses the SLAM problem purely as a computer vision challenge, without relying on additional sensors like LIDAR.

For data, we utilized the **KITTI Benchmark Suite**, estimating the trajectory of a car equipped with grayscale stereo cameras navigating through urban streets in Germany.

In the graphs below, you can see all the different localization results, along with the landmarks detected throughout the process.


![pasted1](https://github.com/user-attachments/assets/e983acb4-cdbf-474b-b65c-a0ebb68f6a5b)



![pasted37](https://github.com/user-attachments/assets/6eff4181-903c-4098-ac0e-fb3a781179a5)


## 🔧 Project Implementation Steps

These are the main steps used to implement this project:

1. **Finding an estimated trajectory using PnP** – a deterministic approach for initial pose estimation.
2. **Building a features tracking database** – to manage and track feature points across frames.
3. **Performing bundle adjustment optimizations over multiple windows** – to refine the estimated poses and 3D points.
4. **Creating a Pose Graph** – to represent the estimated trajectory and connections between poses.
5. **Refining the Pose Graph using Loop Closures** – to correct drift and improve overall accuracy by detecting previously visited locations.


## 🏎️ KITTI Benchmark

The **KITTI dataset** is a project of the Karlsruhe Institute of Technology and the Toyota Technological Institute at Chicago.

KITTI uses a car equipped with several sensors traveling around various streets in Germany. The sensors include stereo cameras (color and black-and-white), GPS, and LIDAR. KITTI's benchmark provides ground truth data, allowing us to evaluate and compare the results of our algorithms. In this project, we use only the **black-and-white stereo cameras**.

The KITTI dataset also provides:

- The intrinsic camera matrix **K** for both stereo cameras.
- The extrinsic parameters of the right camera with respect to the left camera.
- **3360 frames** of data.
- Ground truth extrinsic matrices recorded throughout the drive.


This is an example for a KITTI frame: the upper image is the left and the lower one is the right image of the stereo pair

![pasted3](https://github.com/user-attachments/assets/481d25e6-84da-47d4-a6b2-326475d6d748)


![pasted4](https://github.com/user-attachments/assets/3cc6ebaf-3a39-4a54-a88b-c0463fbe76b4)


## 🎥 Stereo Camera

A **stereo camera** is a type of camera with two or more lenses, each with a separate image sensor or film frame. This setup allows the camera to simulate human binocular vision.

In the KITTI benchmark, we use a stereo camera setup to capture the 3D structure of the scene.

### 📷 Camera Matrices

Every camera, assuming a pinhole camera model, can be represented by an **extrinsic matrix**  [R|t] and an **intrinsic matrix** K.

Together, they define the **projection matrix**  P = K[R|t], which projects a 3D point X (in homogeneous coordinates) to the sensor's coordinate system.

### 🌍 Extrinsic Camera Matrix

The extrinsic camera matrix holds all the information needed to translate from the global coordinate system to the camera coordinate system.

For a 3D point x in the world, the extrinsic matrix performs Rx + t, which involves the necessary rotation and translation to view it in the camera's coordinate system. The camera center, which is the origin of the camera coordinate system, is transformed to the world origin.

In our case, we have two extrinsic camera matrices:

- The **left camera matrix**, which is at the origin:

![image](https://github.com/user-attachments/assets/db2eba00-d750-4b3a-ae22-142e1ab72e4d)



- The **right camera matrix**, which has a translation of 0.54 meters to the right along the x-axis:

![image](https://github.com/user-attachments/assets/b653bf0f-c8b7-4d7a-9284-84d9b2b06831)


As expected, the difference in the right camera location is only 0.54 meters along the x-axis.

### 🔍 Intrinsic Camera Matrix

The intrinsic camera matrix projects a 3D point in the camera coordinate system to a pixel in the image plane (sensor's coordinate system). It includes the camera's intrinsic parameters, such as focal lengths in the x and y directions fx,fy, the principal point  cx,xy, as well as skew and distortion if they exist.

In our case, the intrinsic matrix K is:

![image](https://github.com/user-attachments/assets/c9a06502-4e33-4a7f-8379-35e6f9279a3f)


## 1. 🚗 Finding an Estimated Trajectory Using PnP

In this section, we obtain an initial estimate of the relative movement of the car between two consecutive frames. By composing all these movements together, we achieve a first estimation of the entire vehicle's path.

### 🔍 How is it done?

#### • Feature Extraction and Matching

For each stereo pair of a given frame, we extract key points and their corresponding descriptors using the **AKAZE** algorithm. This method is implemented with OpenCV and based on the paper **"Accelerated Embedded AKAZE Feature Detection Algorithm on FPGA"**. You can find the code implementation [here](https://github.com/AmitaiOvadia/SLAMProject/blob/fcafb474671f3078c2a305bfc554414367019797/VAN_ex/code/utils/utils.py#L128).

This algorithm was chosen for its robust feature detection capabilities and memory-efficient descriptors, which are binary and have a length of 64.

The next step involves matching the descriptors of the left and right images, resulting in a list of corresponding points.

### ⚙️ Implementation Details

- Before applying the feature detection algorithm, the image is blurred using a Gaussian kernel with \(\sigma = 1\) to reduce noise and help the feature extractor focus on more robust features. [See the code here](https://github.com/AmitaiOvadia/SLAMProject/blob/fcafb474671f3078c2a305bfc554414367019797/VAN_ex/code/utils/utils.py#L66).

- We used the OpenCV implementation of AKAZE, lowering the detection threshold from \(10^{-3}\) to \(10^{-4}\). This adjustment increases the number of features detected, compensating for the image blurring.

### 📏 Evaluating the Matches

Given the epipolar 


























