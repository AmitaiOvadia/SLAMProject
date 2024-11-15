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

- Before applying the feature detection algorithm, the image is blurred using a Gaussian kernel with sigma = 1 to reduce noise and help the feature extractor focus on more robust features. [See the code here](https://github.com/AmitaiOvadia/SLAMProject/blob/fcafb474671f3078c2a305bfc554414367019797/VAN_ex/code/utils/utils.py#L66).

- We used the OpenCV implementation of AKAZE, lowering the detection threshold from 10^-3 to 10^-4. This adjustment increases the number of features detected, compensating for the image blurring.

### 📏 Evaluating the Matches

Given the epipolar constraints of a rectified stereo pair, we know that corresponding points must share the same y-value. We can filter out some matches using this metric by checking the y-value distance between matched points. Matches where the y-value distance exceeds **1.5 pixels** are filtered out. [See the implementation here](https://github.com/AmitaiOvadia/SLAMProject/blob/fcafb474671f3078c2a305bfc554414367019797/VAN_ex/code/utils/utils.py#L116).

![pasted6](https://github.com/user-attachments/assets/dc7bd207-812a-4c97-9c3d-296fad20dfe4)


### 🔼 Triangulation of Matched Points

The next step is to **triangulate** each pair of matched points and find the corresponding 3D point for each pair.

This is accomplished using **linear triangulation**, as implemented in the code [here](https://github.com/AmitaiOvadia/SLAMProject/blob/fcafb474671f3078c2a305bfc554414367019797/VAN_ex/code/utils/utils.py#L176).

Let’s define the points:

- x and x': The matched 2D points in the left and right images, respectively.
- X: The corresponding 3D point.

We assume that all these points are represented in **homogeneous coordinates**.

![image](https://github.com/user-attachments/assets/d5070568-f1e6-4c19-a772-8dc5a637a908)


![pasted7](https://github.com/user-attachments/assets/084c93f3-4627-4d9d-9019-33413d52a2b8)


### 🔄 PnP Trajectory Calculation

In this step, we use the previously established tools to compute the relative movement between two consecutive frames.

The **PnP (Perspective-n-Point)** or **P4P** algorithm allows us to estimate the rotation and translation between two views. To use this algorithm, we need:

- At least **4 3D points** and their corresponding **2D projections** in the two views.
- The **intrinsic camera matrix** \( K \).

#### 🛠️ Approach

Here’s how we implemented the PnP trajectory calculation:

1. **Correspondence Matching**:  
   We identified points corresponding across the two stereo pairs by matching the points in the left cameras of both pairs. For each point in the left camera, we then found the matching point in the right camera.

2. **Triangulation**:  
   We triangulated the points in the first stereo pair’s coordinate system (which we assume to be the origin, as we are only interested in relative movement).

3. **Relative Movement Calculation**:  
   Using the **PnP algorithm**, we determined the relative movement (rotation and translation) between the left cameras of the second pair and the first pair. You can view the code implementation [here](https://github.com/AmitaiOvadia/SLAMProject/blob/fcafb474671f3078c2a305bfc554414367019797/VAN_ex/code/ex3/Ex3.py#L73).


In the figures: an illustration of the PnP trajectory calculation, from finding points correspondences across the 2 stereo pairs, to the calculation of the relative movement using pnp 

![pasted9](https://github.com/user-attachments/assets/5d07bc81-fb7b-4c95-8225-7dcd5a382ffd)


### 🛠️ Outliers Removal Using RANSAC

To ensure the accuracy of the relative movement estimation, we need to address potential outliers in our PnP calculation. We chose to handle this using the **RANSAC (Random Sample Consensus)** algorithm.

**RANSAC** is an iterative method used to estimate the parameters of a mathematical model from a set of observed data containing outliers. It effectively identifies outliers, minimizing their influence on the parameter estimates ([Wikipedia](https://en.wikipedia.org/wiki/RANSAC)).

#### 📥 Input to RANSAC

In our case, the input to RANSAC includes:

- A set of **3D points** and their corresponding **2D projections** in the left and right cameras of both stereo pairs.
- The **intrinsic camera matrix** K.
- The relative matrix [R|t] between the right and left cameras.
- The desired probability that the final set of chosen points are indeed inliers.

You can see the RANSAC implementation [here](https://github.com/AmitaiOvadia/SLAMProject/blob/fcafb474671f3078c2a305bfc554414367019797/VAN_ex/code/ex3/Ex3.py#L141).

#### 🔁 RANSAC Iterative Process

1. **Random Sampling**:  
   A random subset of size 4 is selected from the data as hypothetical inliers (enough for the P4P algorithm).

2. **Model Fitting**:  
   The model is fitted to the subset using PnP, resulting in a relative movement matrix M from left camera 1 to left camera 0 of the form [R|t]

4. **Data Testing**:  
   All data points are tested against the fitted model. Assuming the extrinsic matrix of the left camera of the second pair is M from left camera 1 to left camera 0, we use it to find the extrinsic matrix of the right camera. We then project back all the 3D points to each of the cameras. Points with a reprojection error less than d = 1.5 pixels are considered inliers. 

5. **Updating Iterations**:  
   The number of iterations needed is updated based on the following formula:

   ![image](https://github.com/user-attachments/assets/810dda0a-eb8e-49e6-87c4-71e9fa87506f)


   - The chance that all 4 picks are inliers is  w^4.
   - The chance that not all picks are inliers is 1 - w^4.
   - After k iterations, the chance that not all picks are inliers is:

    ![image](https://github.com/user-attachments/assets/49437302-84eb-4aaa-8930-432aafd81671)


6. **Inliers Ratio Update**:  
   We update the inliers ratio w during the iterations, significantly reducing the overall number of iterations.

7. **Final Model Estimation**:  
   After iterating, we select the largest subset of inliers and use it to run PnP for a refined model estimation.

### 🗺️ Estimating the Entire Trajectory

Now that we have all the relative movements from one frame to the next, we can compute the absolute extrinsic matrices for each frame.

The composition of two extrinsic matrices is done as follows:

![image](https://github.com/user-attachments/assets/a8d7f222-aa46-4b75-80ed-27cdaf63884c)


This way, we can accurately estimate the entire trajectory of the vehicle.


 This is how it looks like in the end of the PnP process:

![pasted2](https://github.com/user-attachments/assets/a2ed382f-a563-4eaf-ab43-7486646f5327)


As is clear from this figure, there is a substantial drift that is an attribute of the accumulating mistakes in the pose estimation. We hope to better this estimation using bundle adjustment.


## 🔍 Bundle Adjustment: An Introduction

To further refine the relative poses between consecutive stereo frames, we employ **bundle adjustment**.

**Bundle adjustment** is a fundamental optimization technique in computer vision, especially in 3D reconstruction and Simultaneous Localization and Mapping (SLAM). It refines both the 3D structure and camera parameters by minimizing the **re-projection error** between the observed image points and the predicted projections across multiple views. This process iteratively adjusts the parameters of both the 3D points and the camera poses to achieve a globally optimal solution.

![pasted10](https://github.com/user-attachments/assets/f1e9c9b3-eca9-40ae-bc15-03cb805c29a5)


### 🗄️ Landmark Tracking Database

To perform bundle adjustment, we first need to create a database that tracks all the detected landmarks. This database allows us to:

- Efficiently find which landmarks correspond to each frame.
- Track in which frames each landmark is observed.

The initial implementation of the database was provided by **David Arnon** and was further modified to include additional functionalities. You can find the code for the database implementation [here](https://github.com/AmitaiOvadia/SLAMProject/blob/main/VAN_ex/code/utils/tracking_database.py).












