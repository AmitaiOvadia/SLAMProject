import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import plotly.io as pio
import plotly.graph_objects as go
from utils import utils
NUMBER_OF_FEATURES_TO_SHOW = 2000


class Visualizer:
    @staticmethod
    def plot_point_clouds_and_cameras(points_3D_0, points_3D_1, extrinsic_mat_0, extrinsic_mat_1,
                                      file_name="point_clouds_and_cameras.html"):
        """
        Plot two 3D point clouds and camera pairs on top of each other using Plotly and save as an HTML file.

        Parameters:
        points_3D_0 (np.ndarray): Nx3 array of 3D points for the first point cloud.
        points_3D_1 (np.ndarray): Nx3 array of 3D points for the second point cloud.
        extrinsic_mat_0 (np.ndarray): 3x4 extrinsic matrix for the first camera.
        extrinsic_mat_1 (np.ndarray): 3x4 extrinsic matrix for the second camera.
        file_name (str): Name of the file to save the plot.
        """
        points_3D_0 = points_3D_0[points_3D_0[:, 2] < 50]
        points_3D_1 = points_3D_1[points_3D_1[:, 2] < 50]
        fig = go.Figure()

        # First point cloud
        fig.add_trace(go.Scatter3d(
            x=points_3D_0[:, 0],
            y=points_3D_0[:, 1],
            z=points_3D_0[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color='blue',
                opacity=0.8
            ),
            name='Point Cloud 0'
        ))

        # Second point cloud
        fig.add_trace(go.Scatter3d(
            x=points_3D_1[:, 0],
            y=points_3D_1[:, 1],
            z=points_3D_1[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color='red',
                opacity=0.8
            ),
            name='Point Cloud 1'
        ))

        # Extract R and t from extrinsic matrices
        R_0, t_0 = utils.extract_R_t(extrinsic_mat_0)
        R_1, t_1 = utils.extract_R_t(extrinsic_mat_1)

        # Compute camera centers
        center_0 = utils.get_camera_center_from_Rt(extrinsic_mat_0)
        center_1 = utils.get_camera_center_from_Rt(extrinsic_mat_1)

        # Debug prints for camera centers
        # print("Camera Center 0:", center_0)
        # print("Camera Center 1:", center_1)

        # Create camera cones
        cone_0 = Visualizer.create_camera_cone(center_0, R_0)
        cone_1 = Visualizer.create_camera_cone(center_1, R_1)

        # Add camera cones to the plot
        Visualizer.add_cone_to_plot(fig, cone_0, "Camera 0")
        Visualizer.add_cone_to_plot(fig, cone_1, "Camera 1")

        # Add labels for each camera
        fig.add_trace(go.Scatter3d(
            x=[center_0[0]], y=[center_0[1]], z=[center_0[2]],
            text=["L"], mode='text', name='Camera 0 Label'
        ))
        fig.add_trace(go.Scatter3d(
            x=[center_1[0]], y=[center_1[1]], z=[center_1[2]],
            text=["R"], mode='text', name='Camera 1 Label'
        ))

        # Set plot layout with a tight range around the cones
        all_centers = np.array([center_0, center_1])
        min_vals = np.min(all_centers, axis=0) - 1  # Add some margin
        max_vals = np.max(all_centers, axis=0) + 1  # Add some margin

        fig.update_layout(
            title="3D Point Clouds and Cameras",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                # xaxis=dict(range=[min_vals[0], max_vals[0]]),
                # yaxis=dict(range=[min_vals[1], max_vals[1]]),
                # zaxis=dict(range=[min_vals[2], max_vals[2]]),
                aspectmode='data',
                camera=dict(
                    up=dict(x=0, y=-1, z=0),  # Set the Z axis as up
                    eye=dict(x=0, y=0, z=-1)  # Adjust the camera eye position
                )
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(r=20, l=10, b=10, t=10)
        )

        # Save the plot as an HTML file
        fig.write_html(file_name)
        print(f"Plot saved to {file_name}")

    @staticmethod
    def display_matches_4_cams(left_0_points_2D, right_0_points_2D, left_1_points_2D,
                               right_1_points_2D, image_pair_0=0, image_pair_1=1, num_points_display=None):
        indeices = np.random.randint(0, len(left_0_points_2D), num_points_display)
        if num_points_display is not None:
            left_0_points_2D = left_0_points_2D[indeices]
            right_0_points_2D = right_0_points_2D[indeices]
            left_1_points_2D = left_1_points_2D[indeices]
            right_1_points_2D = right_1_points_2D[indeices]

            # Read images
        left_0, right_0 = utils.read_images(image_pair_0)
        left_1, right_1 = utils.read_images(image_pair_1)

        # Convert images to RGB (OpenCV loads in BGR)
        left_0 = cv2.cvtColor(left_0, cv2.COLOR_BGR2RGB)
        right_0 = cv2.cvtColor(right_0, cv2.COLOR_BGR2RGB)
        left_1 = cv2.cvtColor(left_1, cv2.COLOR_BGR2RGB)
        right_1 = cv2.cvtColor(right_1, cv2.COLOR_BGR2RGB)

        # Define colors
        colors = plt.cm.tab20(np.linspace(0, 1, len(left_0_points_2D)))

        # Draw points and lines on the images
        for idx in range(len(left_0_points_2D)):
            color = tuple((colors[idx][:3] * 255).astype(int).tolist())

            # Points and lines for image pair 0
            cv2.circle(left_0, (int(left_0_points_2D[idx][0]), int(left_0_points_2D[idx][1])), 5, color, -1)
            cv2.circle(right_0, (int(right_0_points_2D[idx][0]), int(right_0_points_2D[idx][1])), 5, color, -1)
            cv2.line(left_0, (int(left_0_points_2D[idx][0]), int(left_0_points_2D[idx][1])),
                     (int(right_0_points_2D[idx][0] + left_0.shape[1]), int(right_0_points_2D[idx][1])), color, 2)
            cv2.line(right_0, (int(right_0_points_2D[idx][0]), int(right_0_points_2D[idx][1])),
                     (int(left_0_points_2D[idx][0] - left_0.shape[1]), int(left_0_points_2D[idx][1])), color, 2)

            # Points and lines for image pair 1
            cv2.circle(left_1, (int(left_1_points_2D[idx][0]), int(left_1_points_2D[idx][1])), 5, color, -1)
            cv2.circle(right_1, (int(right_1_points_2D[idx][0]), int(right_1_points_2D[idx][1])), 5, color, -1)
            cv2.line(left_1, (int(left_1_points_2D[idx][0]), int(left_1_points_2D[idx][1])),
                     (int(right_1_points_2D[idx][0] + left_1.shape[1]), int(right_1_points_2D[idx][1])), color, 2)
            cv2.line(right_1, (int(right_1_points_2D[idx][0]), int(right_1_points_2D[idx][1])),
                     (int(left_1_points_2D[idx][0] - left_1.shape[1]), int(left_1_points_2D[idx][1])), color, 2)

            # Draw lines from upper left to lower left image
            cv2.line(left_0, (int(left_0_points_2D[idx][0]), int(left_0_points_2D[idx][1])),
                     (int(left_1_points_2D[idx][0]), int(left_1_points_2D[idx][1] + left_0.shape[0])), color, 2)
            cv2.line(left_1, (int(left_1_points_2D[idx][0]), int(left_1_points_2D[idx][1])),
                     (int(left_0_points_2D[idx][0]), int(left_0_points_2D[idx][1] - left_1.shape[0])), color, 2)

        # Combine images into a single display
        top_row = np.hstack((left_0, right_0))
        bottom_row = np.hstack((left_1, right_1))
        combined_image = np.vstack((top_row, bottom_row))

        # Display the result using matplotlib
        plt.figure(figsize=(15, 10))
        plt.imshow(combined_image)
        plt.axis('off')
        plt.title('Corresponding Points Across the 2 Stereo Pairs', fontsize=16)
        plt.tight_layout()
        plt.savefig('4cams_opencv.png')
        # plt.show()

    # Function to create a cone representing the camera
    @staticmethod
    def create_camera_cone(center, R, height=0.5, radius=0.2):
        # Cone points in camera coordinates
        cone_points = np.array([
            [0, 0, 0],
            [radius, radius, height],
            [-radius, radius, height],
            [-radius, -radius, height],
            [radius, -radius, height],
        ])

        # Rotate and translate the cone points
        cone_points = np.dot(cone_points, R.T) + center

        # Define the faces of the cone
        faces = [
            [cone_points[0], cone_points[1], cone_points[2]],
            [cone_points[0], cone_points[2], cone_points[3]],
            [cone_points[0], cone_points[3], cone_points[4]],
            [cone_points[0], cone_points[4], cone_points[1]],
        ]

        return faces

    # Function to add a cone to the plot
    @staticmethod
    def add_cone_to_plot(fig, cone, name):
        for face in cone:
            x, y, z = zip(*face)
            fig.add_trace(go.Mesh3d(x=x, y=y, z=z, opacity=0.5, name=name))

    @staticmethod
    def plot_cameras(left_0_center, right_0_center, left_1_center, right_1_center,
                     R_left_0, R_right_0, R_left_1, R_right_1, file_name="camera_pairs_plot.html"):
        # Create camera cones
        left_0_cone = Visualizer.create_camera_cone(left_0_center, R_left_0)
        right_0_cone = Visualizer.create_camera_cone(right_0_center, R_right_0)
        left_1_cone = Visualizer.create_camera_cone(left_1_center, R_left_1)
        right_1_cone = Visualizer.create_camera_cone(right_1_center, R_right_1)

        # Plotting
        fig = go.Figure()

        Visualizer.add_cone_to_plot(fig, left_0_cone, "Left 0")
        Visualizer.add_cone_to_plot(fig, right_0_cone, "Right 0")
        Visualizer.add_cone_to_plot(fig, left_1_cone, "Left 1")
        Visualizer.add_cone_to_plot(fig, right_1_cone, "Right 1")

        # Add labels for each camera
        fig.add_trace(go.Scatter3d(
            x=[left_0_center[0]], y=[left_0_center[1]], z=[left_0_center[2]],
            text=["Left 0"], mode='text', name='Left 0 Label'
        ))
        fig.add_trace(go.Scatter3d(
            x=[right_0_center[0]], y=[right_0_center[1]], z=[right_0_center[2]],
            text=["Right 0"], mode='text', name='Right 0 Label'
        ))
        fig.add_trace(go.Scatter3d(
            x=[left_1_center[0]], y=[left_1_center[1]], z=[left_1_center[2]],
            text=["Left 1"], mode='text', name='Left 1 Label'
        ))
        fig.add_trace(go.Scatter3d(
            x=[right_1_center[0]], y=[right_1_center[1]], z=[right_1_center[2]],
            text=["Right 1"], mode='text', name='Right 1 Label'
        ))

        # Set plot layout with a tight range around the cones
        all_centers = np.array([left_0_center, right_0_center, left_1_center, right_1_center])
        min_vals = np.min(all_centers, axis=0) - 1  # Add some margin
        max_vals = np.max(all_centers, axis=0) + 1  # Add some margin

        fig.update_layout(scene=dict(
            xaxis=dict(range=[min_vals[0], max_vals[0]], title='X'),
            yaxis=dict(range=[min_vals[1], max_vals[1]], title='Y'),
            zaxis=dict(range=[min_vals[2], max_vals[2]], title='Z'),
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=-1, z=0),  # Set the Z axis as up
                eye=dict(x=2, y=2, z=2)  # Adjust the camera eye position
            )
        ),
            # width=700,
            margin=dict(r=20, l=10, b=10, t=10)
        )

        # Save the figure to an HTML file
        pio.write_html(fig, file=file_name, auto_open=False)
        print(f"Plot saved to {file_name}")

    @staticmethod
    def add_point_and_line(fig, start, end, color):
        fig.add_trace(go.Scatter3d(
            x=[start[0]], y=[start[1]], z=[start[2]],
            mode='markers', marker=dict(size=5, color=color),
            showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode='lines', line=dict(color=color, width=5),
            showlegend=False
        ))

    @staticmethod
    def plot_trajectories_plt(estimated_camera_centers, ground_truth_centers):
        # Ensure the arrays are of the same length
        assert len(estimated_camera_centers) == len(ground_truth_centers), "Arrays must have the same length"

        # Select the desired dimensions
        estimated_camera_centers = estimated_camera_centers[:, [0, 2]]
        ground_truth_centers = ground_truth_centers[:, [0, 2]]
        num_points = len(estimated_camera_centers)

        # Generate colors that change gradually with the index
        camera_colors = plt.cm.Blues(np.linspace(0.3, 1, num_points))
        ground_truth_colors = plt.cm.Reds(np.linspace(0.6, 1, num_points))

        plt.figure(figsize=(10, 6))
        s = 1
        # Plot camera centers
        plt.scatter(estimated_camera_centers[:, 0], estimated_camera_centers[:, 1], color=camera_colors,
                    label='Camera Centers', s=s)

        # Plot ground truth centers
        plt.scatter(ground_truth_centers[:, 0], ground_truth_centers[:, 1], color=ground_truth_colors,
                    label='Ground Truth Centers', s=s)

        save_name = 'Camera Centers vs Ground 2D.png'
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.title('Camera Centers vs Ground Truth Centers')
        plt.legend()
        plt.savefig(save_name)
        print(f"saved to {save_name}")
        # plt.show()

    @staticmethod
    def plot_all_cameras(camera_centers, file_name="all_cameras_plot.html", no_y_axis=False):
        fig = go.Figure()
        if no_y_axis:
            camera_centers[:, 1] = 0
            file_name = "no_y_axis_" + file_name
        # Generate colors that change gradually with the index
        num_cameras = len(camera_centers)
        colors = [f'rgba({int(255 * i / num_cameras)}, 0, {int(255 * (num_cameras - i) / num_cameras)}, 1)' for i in
                  range(num_cameras)]
        for i, center in enumerate(camera_centers):
            color = colors[i]
            if i < len(camera_centers) - 1:
                next_center = camera_centers[i + 1]
                Visualizer.add_point_and_line(fig, center, next_center, color)
            else:
                Visualizer.add_point_and_line(fig, center, center, color)

        # Set plot layout with a tight range around the points
        all_centers = np.array(camera_centers)
        min_vals = np.min(all_centers, axis=0) - 1  # Add some margin
        max_vals = np.max(all_centers, axis=0) + 1  # Add some margin

        fig.update_layout(scene=dict(
            xaxis=dict(range=[min_vals[0], max_vals[0]], title='X'),
            yaxis=dict(range=[min_vals[1], max_vals[1]], title='Y'),
            zaxis=dict(range=[min_vals[2], max_vals[2]], title='Z'),
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=-1, z=0),  # Set the Z axis as up
                eye=dict(x=2, y=2, z=2)  # Adjust the camera eye position
            )
        ),
            margin=dict(r=20, l=10, b=10, t=10)
        )

        # Save the figure to an HTML file without opening it
        pio.write_html(fig, file=file_name, auto_open=False)
        print(f"Plot saved to {file_name}")

    @staticmethod
    def plot_all_ground_truth_vs_estimated_cameras(estimated_centers, ground_truth_centers,
                                                   file_name="all_cameras_plot_ground_truth_vs_estimated.html",
                                                   no_y_axis=False):
        fig = go.Figure()
        if no_y_axis:
            estimated_centers[:, 1] = 0
            ground_truth_centers[:, 1] = 0
            file_name = "no_y_axis_" + file_name
        # Generate blueish colors that change slightly with the index for estimated centers
        num_estimated = len(estimated_centers)
        estimated_colors = [f'rgba(0, 0, {int(255 - 100 * i / num_estimated)}, 1)' for i in range(num_estimated)]

        for i, center in enumerate(estimated_centers):
            color = estimated_colors[i]
            if i < len(estimated_centers) - 1:
                next_center = estimated_centers[i + 1]
                Visualizer.add_point_and_line(fig, center, next_center, color)
            else:
                Visualizer.add_point_and_line(fig, center, center, color)

        # Generate yellowish-reddish colors that change gradually with the index for ground truth centers
        num_ground_truth = len(ground_truth_centers)
        ground_truth_colors = [f'rgba(255, {int(255 * (num_ground_truth - i) / num_ground_truth)}, 0, 1)' for i in
                               range(num_ground_truth)]

        for i, center in enumerate(ground_truth_centers):
            color = ground_truth_colors[i]
            if i < len(ground_truth_centers) - 1:
                next_center = ground_truth_centers[i + 1]
                Visualizer.add_point_and_line(fig, center, next_center, color)
            else:
                Visualizer.add_point_and_line(fig, center, center, color)

        # Combine all centers for layout range calculation
        all_centers = np.array(estimated_centers + ground_truth_centers)
        min_vals = np.min(all_centers, axis=0) - 1  # Add some margin
        max_vals = np.max(all_centers, axis=0) + 1  # Add some margin

        fig.update_layout(scene=dict(
            xaxis=dict(range=[min_vals[0], max_vals[0]], title='X'),
            yaxis=dict(range=[min_vals[1], max_vals[1]], title='Y'),
            zaxis=dict(range=[min_vals[2], max_vals[2]], title='Z'),
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=-1, z=0),  # Set the Z axis as up
                eye=dict(x=2, y=-2, z=2)  # Adjust the camera eye position
            )
        ),
            margin=dict(r=20, l=10, b=10, t=10)
        )

        # Save the figure to an HTML file without opening it
        pio.write_html(fig, file=file_name, auto_open=False)
        print(f"Plot saved to {file_name}")

    @staticmethod
    def display_2D(array, legend, save, save_name, show, title, xlabel, ylabel):
        plt.figure()
        plt.plot(array)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(legend)
        plt.title(title)
        if save:
            plt.savefig(save_name)
        if show:
            plt.show()
    @staticmethod
    def display_key_points(Left_img, right_img, keypoints1, keypoints2, good_matches=None,
                bad_matches=None, gap=10, label_width=50,
                number_of_features_to_show=NUMBER_OF_FEATURES_TO_SHOW,
                show_matches=False, line_thickness=1, show_lines=False,
                title="", save_name=""):

        left_label = np.ones((Left_img.shape[0], label_width), dtype=np.uint8) * 255
        right_label = np.ones((right_img.shape[0], label_width), dtype=np.uint8) * 255
        Left_0_with_label = np.hstack((left_label, Left_img))
        Right_0_with_label = np.hstack((right_label, right_img))
        max_width = max(Left_0_with_label.shape[1], Right_0_with_label.shape[1])
        gap_array = np.ones((gap, max_width), dtype=np.uint8) * 255

        if Left_0_with_label.shape[1] < max_width:
            padding = np.ones((Left_0_with_label.shape[0], max_width - Left_0_with_label.shape[1]),
                              dtype=np.uint8) * 255
            Left_0_with_label = np.hstack((Left_0_with_label, padding))

        if Right_0_with_label.shape[1] < max_width:
            padding = np.ones((Right_0_with_label.shape[0], max_width - Right_0_with_label.shape[1]),
                              dtype=np.uint8) * 255
            Right_0_with_label = np.hstack((Right_0_with_label, padding))

        combined_image = np.vstack((Left_0_with_label, gap_array, Right_0_with_label))
        combined_image_color = cv2.cvtColor(combined_image, cv2.COLOR_GRAY2BGR)

        if show_matches:
            for match in good_matches[:number_of_features_to_show]:
                img1_idx = match.queryIdx
                img2_idx = match.trainIdx
                (x1, y1) = keypoints1[img1_idx].pt
                (x2, y2) = keypoints2[img2_idx].pt
                color = (255, 165, 0)
                cv2.circle(combined_image_color, (int(x1) + label_width, int(y1)), 3, color, -1)
                cv2.circle(combined_image_color, (int(x2) + label_width, int(y2) + Left_0_with_label.shape[0] + gap), 3,
                           color, -1)
                if show_lines:
                    cv2.line(combined_image_color, (int(x1) + label_width, int(y1)),
                             (int(x2) + label_width, int(y2) + Left_0_with_label.shape[0] + gap), color, line_thickness)

            if bad_matches:
                for match in bad_matches[:number_of_features_to_show]:
                    img1_idx = match.queryIdx
                    img2_idx = match.trainIdx
                    (x1, y1) = keypoints1[img1_idx].pt
                    (x2, y2) = keypoints2[img2_idx].pt
                    color = (0, 165, 255)
                    cv2.circle(combined_image_color, (int(x1) + label_width, int(y1)), 3, color, -1)
                    cv2.circle(combined_image_color,
                               (int(x2) + label_width, int(y2) + Left_0_with_label.shape[0] + gap), 3,
                               color, -1)
                    if show_lines:
                        cv2.line(combined_image_color, (int(x1) + label_width, int(y1)),
                                 (int(x2) + label_width, int(y2) + Left_0_with_label.shape[0] + gap), color,
                                 line_thickness)
        else:
            for kp in keypoints1[:number_of_features_to_show]:
                x, y = kp.pt
                cv2.circle(combined_image_color, (int(x) + label_width, int(y)), 3, (0, 255, 0), -1)

            for kp in keypoints2[:number_of_features_to_show]:
                x, y = kp.pt
                cv2.circle(combined_image_color, (int(x) + label_width, int(y) + Left_0_with_label.shape[0] + gap), 3,
                           (0, 255, 0), -1)

        plt.figure(figsize=(12, 8))
        plt.imshow(combined_image_color)
        plt.title(title)
        plt.axis('off')
        plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
        plt.close()

    @staticmethod
    def plot_matches_and_supporters(left_0_reprojected, left_1_reprojected,
                                    left_0_points_2D, left_1_points_2D,
                                    left_0, left_1):
        """
        Plot on images left_0 and left_1 the matches, with supporters in different color.
        Also, plot the reprojected points in different colors.

        Parameters:
        left_0_reprojected (np.ndarray): Reprojected points on the left_0 image.
        left_1_reprojected (np.ndarray): Reprojected points on the left_1 image.
        left_0_points_2D (np.ndarray): Original 2D points on the left_0 image.
        left_1_points_2D (np.ndarray): Original 2D points on the left_1 image.
        left_0 (np.ndarray): Image left_0.
        left_1 (np.ndarray): Image left_1.
        inliers_inds (list or np.ndarray): Indices of the inlier points.
        """

        # Convert images to grayscale
        N = left_0_points_2D.shape[0]
        if len(left_0.shape) == 3:
            left_0_gray = cv2.cvtColor(left_0, cv2.COLOR_BGR2GRAY)
        else:
            left_0_gray = left_0

        if len(left_1.shape) == 3:
            left_1_gray = cv2.cvtColor(left_1, cv2.COLOR_BGR2GRAY)
        else:
            left_1_gray = left_1

        # Convert grayscale images back to RGB for plotting colored points
        left_0_draw = cv2.cvtColor(left_0_gray, cv2.COLOR_GRAY2RGB)
        left_1_draw = cv2.cvtColor(left_1_gray, cv2.COLOR_GRAY2RGB)

        # Define colors
        points_2D_color = (255, 0, 0)  # Red color for original 2D points
        reprojected_color = (0, 255, 0)  # Green color for reprojected points

        # Draw original 2D points with 'o' markers
        for idx in range(len(left_0_points_2D)):
            pt1 = (int(left_0_points_2D[idx][0]), int(left_0_points_2D[idx][1]))
            pt2 = (int(left_1_points_2D[idx][0]), int(left_1_points_2D[idx][1]))
            cv2.circle(left_0_draw, pt1, 4, points_2D_color, 2)
            cv2.circle(left_1_draw, pt2, 4, points_2D_color, 2)

        # Draw reprojected points with '+' markers
        for idx in range(N):
            pt1 = (int(left_0_reprojected[idx][0]), int(left_0_reprojected[idx][1]))
            pt2 = (int(left_1_reprojected[idx][0]), int(left_1_reprojected[idx][1]))
            cv2.drawMarker(left_0_draw, pt1, reprojected_color, markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)
            cv2.drawMarker(left_1_draw, pt2, reprojected_color, markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)

        # Display the images with matches, inliers, and reprojected points
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 1, 1)
        plt.title("Left 0 \noriginals in red 'o', reprojected in green '+'")
        plt.imshow(left_0_draw)
        plt.axis('off')
        plt.subplot(2, 1, 2)
        plt.title("Left 1 \noriginals in red 'o', reprojected in green '+'")
        plt.imshow(left_1_draw)
        plt.axis('off')
        plt.tight_layout()
        # Save the plot as an image
        plt.savefig('left0_left1_matches_and_supporters.png')

    @staticmethod
    def plot_left0_left1_inliers_and_outliers(left_0_points_2D, left_1_points_2D,
                                              max_inliers_indices, left_0, left_1):
        """
        Plots the inliers and outliers on the given images left_0 and left_1.
        Inliers are shown in one color, and outliers in another color.
        Images are displayed with left_0 above left_1.

        Parameters:
        left_0_points_2D (np.ndarray): 2D points in the left_0 image.
        left_1_points_2D (np.ndarray): 2D points in the left_1 image.
        max_inliers_indices (list or np.ndarray): Indices of inliers.
        left_0 (np.ndarray): The left_0 image.
        left_1 (np.ndarray): The left_1 image.
        """
        # Convert inlier indices to set for quick lookup
        inlier_set = set(max_inliers_indices)

        # Determine inliers and outliers
        inliers_0 = [pt for i, pt in enumerate(left_0_points_2D) if i in inlier_set]
        outliers_0 = [pt for i, pt in enumerate(left_0_points_2D) if i not in inlier_set]
        inliers_1 = [pt for i, pt in enumerate(left_1_points_2D) if i in inlier_set]
        outliers_1 = [pt for i, pt in enumerate(left_1_points_2D) if i not in inlier_set]

        # Plot the images
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 10))
        s = 3
        # Plot left_0 image and points
        ax0.imshow(left_0, cmap='gray')
        ax0.scatter(*zip(*inliers_0), color='orange', label='Inliers', s=s)
        ax0.scatter(*zip(*outliers_0), color='cyan', label='Outliers', s=s)
        ax0.legend()
        ax0.set_title('left_0 inliers in orange and outliers in cyan')

        # Plot left_1 image and points
        ax1.imshow(left_1, cmap='gray')
        ax1.scatter(*zip(*inliers_1), color='orange', label='Inliers', s=s)
        ax1.scatter(*zip(*outliers_1), color='cyan', label='Outliers', s=s)
        ax1.legend()
        ax1.set_title('left_1 inliers in orange and outliers in cyan')

        plt.tight_layout()
        plt.savefig('section 3.5 lef0 and left1 inliers and outliers.png')