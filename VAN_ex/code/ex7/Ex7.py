
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
import gtsam
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.BundleAdjusment import BundleAdjusment, Bundelon
from utils.PoseGraph import PoseGraph
from ex3.Ex3 import ImageProcessor
from utils.tracking_database import TrackingDB
from ex5.Ex5 import display_3d_trajectory_gtsam_function
import pandas as pd
import plotly.graph_objects as go

# TRACKING_DB_PATH = '../ex4/tracking_db_1.5_acc'
TRACKING_DB_PATH = '../ex4/tracking_db_1.5_acc'



def plot_camera_centers_with_labels(bundle_object):
    key_frames_to_camera_locations = bundle_object.key_frames_to_camera_centers
    key_frames_centers = np.array([key_frames_to_camera_locations[frame] for frame in bundle_object.key_frames])
    # Convert to DataFrame for easier handling
    key_frames_centers[:, 1] = 0
    df = pd.DataFrame(key_frames_centers, columns=['x', 'y', 'z'])
    df['label'] = np.array(bundle_object.key_frames).astype(int)
    # Create a scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=df['x'],
        y=df['y'],
        z=df['z'],
        mode='markers+text',
        marker=dict(size=5, color='blue'),
        text=df['label'],
        textposition="top center"
    )])
    # Set plot title and labels
    fig.update_layout(
        title='3D Trajectory with Labels',
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        )
    )
    # Save the plot to an HTML file
    fig.write_html("trajectory_plot.html")


def task_7_1(tracking_db, processor):
    bundle_object = BundleAdjusment(tracking_db)
    bundle_object.create_and_solve_all_bundle_windows()
    plot_camera_centers_with_labels(bundle_object)
    pose_graph = PoseGraph(bundle_object, processor, do_loop_closure=True)
    all_optimized_values = pose_graph.loop_closure_pair_to_optimized_values
    all_loop_closure_frames = [all_optimized_values[i][1] for i in range(len(all_optimized_values))]

    # How many successful loop closures were detected?
    print(f"the total number of loop closure detected was: {pose_graph.total_num_loop_closures} "
          f"across {pose_graph.loop_closure_frames_counter} loop closure frames (frames from which at least one loop closure was detected)")

    # Plot the match results of a single successful consensus match of your choice. (For the left
    # images, inliers and outliers in different color)
    # *** done inside the PoseGraph class in the function Visualizer.display_key_points_with_inliers_outliers inside
    # get_initial_pose_estimate_and_links ***

    # Choose 4 versions of the pose graph along the process and plot them (including location
    # covariance).
    plot_locations_with_covariance_gtsam(all_loop_closure_frames, all_optimized_values, pose_graph.ground_truth_locations)

    # Plot the pose graph locations along with the ground truth both with and without loop
    # closures.
    # *** done during the pose graph optimization in the function:
    # self.save_cur_camera_locations_status_vs_ground_truth() ***

    # Plot a graph of the absolute location error for the whole pose graph both with and without
    # loop closures.
    absolute_location_error_before_vs_after_loop_closures(all_optimized_values, pose_graph)

    # Plot a graph of the location uncertainty size for the whole pose graph both with and
    # without loop closures. (What measure of uncertainty size did you choose?)
    plot_location_uncertainty(pose_graph)


def plot_location_uncertainty(pose_graph):
    plt.figure()
    loop_closure_frames = pose_graph.loop_closure_keyframes_indices
    initial_uncertainty_sizes = pose_graph.initial_location_uncertainty_per_frame
    final_uncertainty_sizes = pose_graph.final_location_uncertainty_per_frame

    plt.plot(initial_uncertainty_sizes, label="Initial uncertainty", color="red")
    plt.plot(final_uncertainty_sizes, label="Final uncertainty", color="blue")

    # Scatter plot for loop closure frames, setting y-values slightly above zero for visibility
    plt.scatter(loop_closure_frames, final_uncertainty_sizes[loop_closure_frames], label="Loop closure frames", color="green", s=10, zorder=5)

    plt.yscale('log', base=2)
    plt.xlabel('Key frame')
    plt.ylabel('Uncertainty Size (log scale, base 2)')
    plt.title('Uncertainty Sizes on Logarithmic Scale (base 2)')
    plt.legend()
    plt.savefig("Uncertainty_sizes_before_and_after_loop_closure.png", dpi=600)
    plt.close()


def absolute_location_error_before_vs_after_loop_closures(all_optimized_values, pose_graph):
    all_loop_closure_frames, loop_closure_keyframes_indices = pose_graph.get_loop_closure_frames()
    ground_truth_key_frames = pose_graph.ground_truth_locations[pose_graph.key_frames]
    all_camera_centers_initial = all_optimized_values[0][-1]
    all_camera_centers_final = all_optimized_values[len(all_optimized_values) - 1][-1]
    l2_distance_before_loop_closure = np.linalg.norm(all_camera_centers_initial - ground_truth_key_frames, axis=-1)
    l2_distance_after_loop_closure = np.linalg.norm(all_camera_centers_final - ground_truth_key_frames, axis=-1)
    plt.figure()
    plt.plot(l2_distance_before_loop_closure, label="L2 distance before loop closure", color="red")
    plt.plot(l2_distance_after_loop_closure, label="L2 distance after loop closure", color="blue")
    plt.scatter(loop_closure_keyframes_indices, np.zeros_like(loop_closure_keyframes_indices),
                label="Loop closure keyframes", color="green")
    plt.legend()
    plt.title("L2 Distance Before and after loop closure")
    plt.savefig("L2_distance_before_vs_after_loop_closure.png", dpi=600)


def plot_locations_with_covariance_gtsam(all_loop_closure_frames,
                                         all_optimized_values,
                                         ground_truth_locations):
    # Calculate limits
    margin = 50
    xmin = ground_truth_locations[:, 0].min() - margin
    xmax = ground_truth_locations[:, 0].max() + margin
    zmin = ground_truth_locations[:, 2].min() - margin
    zmax = ground_truth_locations[:, 2].max() + margin
    start = 0
    finish = len(all_loop_closure_frames) - 1
    first_loop_closure = 1  # first loop closure
    half_way = int((start + finish) // 2)  # middle
    for i, loop_closure_ind in enumerate([start, first_loop_closure, half_way, finish]):
        print(i)
        optimized_values = all_optimized_values[loop_closure_ind][0]
        marginals = all_optimized_values[loop_closure_ind][2]
        plt.close()
        fig = plt.figure()
        gtsam.utils.plot.plot_trajectory(0, optimized_values, marginals=marginals, scale=1)
        plt.tight_layout()

        ax = plt.gca()
        # Set axis limits
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([zmin, zmax])
        ax.view_init(elev=0, azim=270)
        # ax.set_aspect('equal')
        plt.savefig(f"loop closure ind {loop_closure_ind}.png", dpi=600)
        plt.close(fig)
        plt.close()
        del fig


if "__main__" == __name__:
    processor = ImageProcessor()
    tracking_db = TrackingDB(processor.K, processor.M1, processor.M2)
    tracking_db.load(TRACKING_DB_PATH)
    task_7_1(tracking_db, processor)
