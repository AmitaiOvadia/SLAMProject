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

The intrinsic camera matrix projects a 3D point in the camera coordinate system to a pixel in the image plane (sensor's coordinate system). It includes the camera's intrinsic parameters, such as focal lengths in the x and y directions fx,fy, the principal point  cx,cy, as well as skew and distortion if they exist.

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


### 📥 Adding a Frame to the Database

The dataset is populated incrementally, one frame at a time, following this logic: [View the code implementation here](https://github.com/AmitaiOvadia/SLAMProject/blob/fcafb474671f3078c2a305bfc554414367019797/VAN_ex/code/ex4/Ex4.py#L58).

#### 🗂️ Frame Processing Steps

**For each frame:**

- **Extract Features**: Identify keypoints and descriptors from the left and right images.
- **Match Features**: Establish correspondences between matched features in the left and right images.

**For the first frame:**

- **Track Initialization**: Create links between matched features, forming the initial tracks.
- **Add to Database**: Store the initial links and features in the database.

**For each subsequent frame:**

- **Match with Previous Frame**: Match current frame features with those from the previous frame to maintain tracking continuity.
- **Outlier Detection**: Use RANSAC to remove unreliable matches.
- **Inlier Identification**: Update tracks with valid matches and create new links for reliable feature correspondences.

#### 🔗 Track Formation and Updating

- **Track Creation and Continuation**: New tracks are initiated for unmatched features, while existing tracks are extended with matched features.
- **Track Management**: Update existing tracks with the new frame’s features, refining links as necessary.

#### 🗃️ Adding the Frame to the Database

- **Store Links and Features**: Update the database with the new links and features.
- **Update Camera Pose**: Record the camera’s position and orientation for each frame.

**Finalization**: The tracking database is serialized (saved) after processing all frames, preserving the tracking information for future use.

### 📊 Database Objective and Tracking Analysis

The figure below illustrates the purpose of the tracking database, showing an example of tracking a specific landmark across multiple frames.

On the right, you can see a histogram of tracking lengths. It appears linear on a log scale, indicating exponential behavior. This makes sense if we consider the probability of a landmark appearing in the next frame to be p. Thus, the probability of it appearing in the frame after that is p^2, and so on. The probability of a track having a length of n is p^n, which decreases exponentially.

![pasted11](https://github.com/user-attachments/assets/026e0900-cafe-4ba7-a7cd-20c8662dff9a)


![pasted12](https://github.com/user-attachments/assets/69c91d71-1c4a-49a8-ae7e-ee380163912c)


![pasted13](https://github.com/user-attachments/assets/d079a969-353a-4b54-aa6f-c2e8a37659c7)


Here is a graph of the number of matches per frame, and also the the percentage of inliers per frame

![pasted24](https://github.com/user-attachments/assets/fa61de97-3c79-43f9-b3e2-7a82bce5e0e0)


### 📈 Tracking Statistics and Analysis

We observe that these statistics are often correlated. Typically, a high number of matches is associated with a large number of inliers. 

This can be explained by the fact that when there are many matches, the images likely show very little change in the scene between the two frames, reducing the likelihood of points being classified as outliers.

#### 📝 Tracking Statistics

- **Total number of tracks**: 402,226
- **Number of frames**: 3,360
- **Mean track length**: 5.26
- **Maximum track length**: 153
- **Minimum track length**: 2
- **Mean number of frame links**: 629.77

### 🔗 Connectivity Graph

Another important graph is the **connectivity** graph, which shows how many tracks are shared between two consecutive frames. This score is crucial because a sudden drop in connectivity may indicate poor pose estimation for that frame. Such a drop would likely affect future absolute localizations, compromising the accuracy of the trajectory estimation.

Here we can see that there was good connectivity throughout the process, as it never dropped below 145. 
![pasted25](https://github.com/user-attachments/assets/9de5897e-9921-494d-8057-b4f6e8d32728)


### 3. 🔄 Performing Bundle Adjustment Over Multiple Windows

When performing bundle adjustment, we face a challenge: most tracks are short and do not extend throughout the entire sequence. If we attempt to solve bundle adjustment for all frames at once, the problem would become very large and sparse. To address this, we divide the sequence into smaller parts, solving bundle adjustment separately for each part.

Each segment is called a **"bundle window"** or, in our code, a **"bundelon"**. [See the bundelon implementation here](https://github.com/AmitaiOvadia/SLAMProject/blob/fcafb474671f3078c2a305bfc554414367019797/VAN_ex/code/utils/BundleAdjusment.py#L14).

#### 🔑 Division Into Keyframes

We want to divide the sequence into small parts that are:

- Small enough to ensure efficient bundle adjustment solving.
- Not too sparse, so many tracks are shared between the frames.
- Large enough to include plenty of constraints for better estimation of camera poses.

This is done by selecting **keyframes** based on feature track lengths. The process involves:

- Calculating the lengths of feature tracks in the current frame.
- Determining a typical track length using the 40th percentile.
- Choosing the next keyframe by advancing by this percentile length, ensuring consistent feature continuity.

Additionally, each bundle window overlaps by one frame with the next window to maintain continuity (the first frame of each window is the last frame of the previous window). [See the keyframe division code here](https://github.com/AmitaiOvadia/SLAMProject/blob/fcafb474671f3078c2a305bfc554414367019797/VAN_ex/code/utils/BundleAdjusment.py#L464).

#### 🗂️ Single Bundle Creation

All frames between keyframes form the bundle windows, and we solve the bundle adjustment problem for each of them.

We use a **GTSAM factor graph** for this purpose, adding the following factors: [View the bundle creation code here](https://github.com/AmitaiOvadia/SLAMProject/blob/fcafb474671f3078c2a305bfc554414367019797/VAN_ex/code/utils/BundleAdjusment.py#L157).

- **Stereo Camera Poses**:  
  The first frame of each window is assumed to be the "origin," with subsequent poses calculated relative to it.

- **Tracks and Camera Observations**:  
  All tracks are included, along with the cameras in which each track is observed.

- **Measurement Uncertainty**:  
  We add a Gaussian uncertainty factor sigma = 1 for the 2D measurements detected by the AKAZE detector.

- **Prior for the First Frame**:  
  We include a prior on the first frame's pose, assuming it is at the origin. The uncertainties are:
  - **Pose Euler angles**: 1 degree.
  - **X direction**: 0.1 meters.
  - **Y direction**: 0.01 meters.
  - **Z direction**: 1 meter.

Due to the non-convex nature of the bundle adjustment problem, we also provide an initial guess for each factor:
- **Camera Poses**: Initialized using PnP estimations.
- **3D Landmarks**: Initialized using their triangulated location from the last frame where they were observed (as stereo triangulation improves when closer to the 3D point).

#### 🔍 Optimizing the Factor Graph

The final phase is optimizing the factor graph using the **Levenberg-Marquardt algorithm**, which balances between the Gauss-Newton algorithm (GNA) and gradient descent. [View the optimization code here](https://github.com/AmitaiOvadia/SLAMProject/blob/fcafb474671f3078c2a305bfc554414367019797/VAN_ex/code/utils/BundleAdjusment.py#L268).

In the figure below, you can see an example of the improvement achieved by bundle adjustment in terms of the reprojection error, compared to the initial estimation from PnP camera poses.


![pasted15](https://github.com/user-attachments/assets/ce55d435-95e2-4675-80ab-7ad9ab76ba95)


### 📈 Extracting Relative Transformation and Covariance from Bundle Adjustment

The bundle adjustment optimization returns a solution consisting of the camera poses and the 3D landmarks’ positions, maximizing the likelihood based on observed points and the injected prior information.

In addition, the optimization provides the **covariance matrix** of the entire optimized factor graph solution. This covariance matrix allows us to extract the uncertainty of the relative transformation (movement) between the first and last frame of each bundle window. This is done using **marginalization and conditioning** techniques.

#### 🔍 Comparing Bundle Adjustment and PnP Localizations

Below, we present a comparison of the localizations obtained from bundle adjustment versus those from PnP estimations. The results show a slight improvement with bundle adjustment; however, there is still noticeable drift due to the accumulation of errors.

Additionally, the following graph illustrates the division into keyframes. As expected, keyframes are spaced more densely during turns, capturing significant changes in the trajectory, while they are more sparsely distributed when the trajectory is relatively straight.


![pasted14](https://github.com/user-attachments/assets/a3a7ab6c-d820-42fc-93e3-0cb4c62f2c2a)


![pasted19](https://github.com/user-attachments/assets/0683fb36-efff-4ba7-8a5e-f72883ba29c1)



In the following figure we can see the improvement in the median reprojection error, and in the mean factor error, before and after the bundle adjustment.


![pasted26](https://github.com/user-attachments/assets/ff8fab91-3bbe-4c19-b962-7073c6e2b55e)


![pasted27](https://github.com/user-attachments/assets/5b950b7b-a5b2-464e-bc4c-7a39a4f630a0)

We can see that there is an improvement, and also that the median reprojection error initially was not bad, this is because of a strict outlier removal policy of 1.5 pixels reprojection error threshold.


### 4. 🗺️ Pose Graph Creation

The next step after bundle adjustment is the creation of the **pose graph**.

#### 📚 What is a Pose Graph?

A **pose graph** is a data structure that holds only the camera poses obtained for each keyframe, along with the associated movement covariance (uncertainty) relative to the previous keyframe (or any other connected keyframe). 

This provides a much more compact representation of the trajectory compared to the previous bundle adjustment representation, which included all the cameras and landmarks. The primary purpose of the pose graph is to facilitate the detection of **loop closures** in the next step. [View the loop closure code here](https://github.com/AmitaiOvadia/SLAMProject/blob/fcafb474671f3078c2a305bfc554414367019797/VAN_ex/code/utils/PoseGraph.py#L375).

#### 🛠️ Pose Graph Optimization

The pose graph is initialized with only the **relative poses** between cameras and the associated covariance for these poses. It is then optimized using a **maximum likelihood approach**. [View the optimization code here](https://github.com/AmitaiOvadia/SLAMProject/blob/fcafb474671f3078c2a305bfc554414367019797/VAN_ex/code/utils/PoseGraph.py#L408).

![pasted16](https://github.com/user-attachments/assets/39adbd97-d423-4ec5-9f30-d28b06da98cb)


- **Initial Optimization**:  
  When optimizing the pose graph using only the initial keyframe poses, there is no significant change because no new information has been introduced.

- **Incorporating Loop Closures**:  
  However, once additional constraints are added in the form of **loop closures**, the graph is substantially updated to accommodate the new information, leading to improved trajectory estimation.

This compact and optimized representation is essential for refining the global trajectory and reducing drift in the final localization results.


### 5. 🔗 Refining the Pose Graph Using Loop Closures

[View the implementation here](https://github.com/AmitaiOvadia/SLAMProject/blob/fcafb474671f3078c2a305bfc554414367019797/VAN_ex/code/utils/PoseGraph.py#L21).

In this step, we address the **drifting problem** caused by accumulating small errors in the relative transformations, especially in the azimuth angle. By composing these transformations, even minor inaccuracies can lead to significant drift over time.

We can exploit the fact that the vehicle revisits some of its previous locations. If we recognize these revisits, we can impose additional constraints known as **loop closures**. These constraints act as "staples" or anchors, correcting the drift locally and redistributing errors across the entire pose graph.

#### 🛠️ Loop Closure Process

1. **Creating the Loop Closure Graph**  
   We initialize a loop closure graph object, where each vertex represents a camera pose for a specific keyframe. The weights on the graph’s edges are based on the uncertainty between two poses p1 and p2:

![image](https://github.com/user-attachments/assets/3cf00907-1954-4a20-9314-f1f949322bb6)

   This weight is proportional to the volume of the covariance matrix. Each edge also stores the covariance matrix itself.  
   [See the code here](https://github.com/AmitaiOvadia/SLAMProject/blob/fcafb474671f3078c2a305bfc554414367019797/VAN_ex/code/utils/PoseGraph.py#L353).

2. **Finding Loop Closure Candidates**  
   We use pose distance measurements to identify keyframes that are likely at the same location and orientation:

   - For each keyframe, we check previous keyframes starting from 60 frames back (to avoid checking close frames).
   - For each previous frame, we find the shortest path in the graph based on the uncertainty weights.  
     [See the code here](https://github.com/AmitaiOvadia/SLAMProject/blob/fcafb474671f3078c2a305bfc554414367019797/VAN_ex/code/utils/PoseGraph.py#L169).
   - We construct an approximated covariance matrix by summing the covariances of the intermediate edges.  
     [See the code here](https://github.com/AmitaiOvadia/SLAMProject/blob/fcafb474671f3078c2a305bfc554414367019797/VAN_ex/code/utils/PoseGraph.py#L336).
   - We compute the Mahalanobis distance:

 ![image](https://github.com/user-attachments/assets/90d4a6bb-aeba-4c7b-b23f-f35c8d71d6f4)


     [See the code here](https://github.com/AmitaiOvadia/SLAMProject/blob/fcafb474671f3078c2a305bfc554414367019797/VAN_ex/code/utils/PoseGraph.py#L346).
   - If the Mahalanobis distance is below a threshold (750 in our case), then p_cur, p_prev are considered loop closure candidates.

3. **Performing Loop Closure Optimization**  
   After compiling the loop closure candidates for p_{cur}:

   - For each candidate  p_{cand}, we find the relative pose using the PnP algorithm, discarding candidates with fewer than 50 inliers.  
     [See the code here](https://github.com/AmitaiOvadia/SLAMProject/blob/fcafb474671f3078c2a305bfc554414367019797/VAN_ex/code/utils/PoseGraph.py#L272).
   - Using the PnP result as the initial estimate, we perform a small bundle adjustment involving only the two cameras and retrieve the associated covariance.  
     [See the code here](https://github.com/AmitaiOvadia/SLAMProject/blob/fcafb474671f3078c2a305bfc554414367019797/VAN_ex/code/utils/PoseGraph.py#L247).
   - We update the pose graph by adding the new edge {p_cur, p_cand}, optimize the pose graph, and update the loop closure graph object.

#### 🗺️ Results

Below, you can see how the pose graph evolves as more loop closures are added, along with a comparison to the ground truth locations. The addition of loop closures significantly reduces drift and improves the overall accuracy of the trajectory estimation.


![pasted20](https://github.com/user-attachments/assets/f4bfe4ff-2e15-4890-94b9-4136e1fd9837)
![pasted21](https://github.com/user-attachments/assets/69affa89-8674-44ea-93ca-6e065756dbff)
![pasted22](https://github.com/user-attachments/assets/2c594e7b-5aaf-4e99-9c25-35e5495eb715)
![pasted23](https://github.com/user-attachments/assets/f27291b7-67a4-475d-af74-b571e91c75f1)



## 📊 Performance Analysis

In this section, I provide additional quantitative measures of the SLAM system.

Here are graphs of the **absolute localization errors** in x, y, z, and L2 norm, after:

- The initial PnP estimation.
- The bundle adjustment (before loop closure).
- The loop closure.

The graphs illustrate the improvement in localization accuracy as we progress through each stage of the SLAM pipeline.


![pasted28](https://github.com/user-attachments/assets/29125330-bae0-45f7-9386-4233ab8fbdc6)  ![pasted29](https://github.com/user-attachments/assets/a5d55000-cd7a-4dd2-8852-30f7c0360d1e)



![pasted30](https://github.com/user-attachments/assets/1278603a-d763-41e5-8894-aa942c4cbee9)   ![pasted1](https://github.com/user-attachments/assets/88e2d7a7-4d9f-46d1-9622-f44992e35f59)





It's clear from the graphs the there is a real boost in the absolute estimation accuracy after the loop closure part, the maximum error comes down from around 40 meters to around 5 meters. The reason is of course the elimination of the drift, or the accumulating error. 


## 🔍 Relative Error Analysis

In the following graphs, we focus on the **relative errors**, both in localization and in pose angles.

A basic relative estimation error graph compares the relative pose estimation between two consecutive frames ![image](https://github.com/user-attachments/assets/f864c270-d074-4376-8062-1f44159271f0)
 and the ground truth relative pose between them ![image](https://github.com/user-attachments/assets/32929d6f-0b6f-4808-a127-00293c921b18)


![pasted31](https://github.com/user-attachments/assets/60f61ff2-6020-4f0c-b52c-c9ddbfef5313)


![pasted32](https://github.com/user-attachments/assets/871b5a91-7b00-405e-975b-a04cae16eb30)


![pasted33](https://github.com/user-attachments/assets/37d7ab07-5484-4c9a-81cd-54e9f5456f81)

















