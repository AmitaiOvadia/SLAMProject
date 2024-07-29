
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
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
    a=0



if "__main__" == __name__:
    processor = ImageProcessor()
    tracking_db = TrackingDB(processor.K, processor.M1, processor.M2)
    tracking_db.load(TRACKING_DB_PATH)
    task_7_1(tracking_db, processor)
