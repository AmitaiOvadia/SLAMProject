
import gtsam.utils.plot
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.BundleAdjusment import BundleAdjusment, Bundelon, PoseGraph
from ex3.Ex3 import ImageProcessor
from utils.tracking_database import TrackingDB
from ex5.Ex5 import display_3d_trajectory_gtsam_function
# TRACKING_DB_PATH = '../ex4/tracking_db_1.5_acc'
TRACKING_DB_PATH = '../ex4/tracking_db_2_acc_y_2_5e-4_blur_0_it_5_akaze_good'
DATA_PATH = r"/cs/labs/tsevi/amitaiovadia/SLAMProject/VAN_ex/dataset/sequences/00"
matplotlib.use('TKAgg')



def task_6_1(tracking_db):
    bundle_object = BundleAdjusment(tracking_db, calculate_only_first_bundle=True)
    bundle_object.create_and_solve_all_bundle_windows()

    # Extract the marginal covariances of the solution.
    # Plot the resulting frame locations as a 3D graph including the covariance of the locations.
    # (all the frames in the bundle, not just the 1st and last)
    first_bundelon = bundle_object.key_frames_to_bundelons[0]
    graph = first_bundelon.get_factor_graph()
    optimized_values = first_bundelon.get_optimized_values_gtsam_format()
    marginals = gtsam.Marginals(graph, optimized_values)

    gtsam.utils.plot.plot_trajectory(fignum=0, marginals=marginals, values=optimized_values, title="Trajectory of first "
                                                                                                   "bundle frames")
    plt.show()

    # Print the resulting relative pose between the first two keyframes and the covariance
    # associated with it.
    key_0, key_k = bundle_object.key_frames
    keys = gtsam.KeyVector()
    keys.append(gtsam.symbol('c', key_0))
    keys.append(gtsam.symbol('c', key_k))
    joint_covariance = np.linalg.inv(marginals.jointMarginalInformation(keys).at(keys[-1], keys[-1]))

    _, camera_poses_gtsam = first_bundelon.get_all_optimized_camera_poses()
    first_camera_gtsam = camera_poses_gtsam[key_0]
    second_camera_gtsam = camera_poses_gtsam[key_k]
    relative_pose_gtsam = first_camera_gtsam.between(second_camera_gtsam)

    # pose_0 = optimized_values.atPose3(gtsam.symbol('c', key_0))
    # pose_k = optimized_values.atPose3(gtsam.symbol('c', key_k))
    # relative_pose = pose_0.between(pose_k)

    Rt_relative_pose = Bundelon.get_Rt_from_gtsam_Pose3(relative_pose_gtsam)
    print(f"the relative pose between the keyframes is: {relative_pose_gtsam}\n"
          f"and the associated covariance is {joint_covariance}")


def task_6_2(tracking_db):
    bundle_object = BundleAdjusment(tracking_db)
    bundle_object.create_and_solve_all_bundle_windows()
    pose_graph = PoseGraph(bundle_object)
    initial_error = pose_graph.get_total_graph_error()
    pose_graph.optimize()
    final_error = pose_graph.get_total_graph_error()

    initial_poses = pose_graph._initial_est
    optimized_values = pose_graph._optimized_values
    
    # plt.figure()
    gtsam.utils.plot.plot_trajectory(1, initial_poses, scale=1, title="Initial poses")
    ax = plt.gca()
    ax.view_init(elev=0, azim=270)
    plt.show()

    # Plot optimized trajectory without covariance
    gtsam.utils.plot.plot_trajectory(2, optimized_values, scale=1, title="Optimized poses")
    ax = plt.gca()
    ax.view_init(elev=0, azim=270)
    plt.show()

    # Optimized trajectory with covariance
    marginals = pose_graph.get_marginals()
    gtsam.utils.plot.plot_trajectory(3, optimized_values, marginals=marginals, scale=1)
    ax = plt.gca()
    ax.view_init(elev=0, azim=270)
    plt.show()

    print(f"The initial graph error is {initial_error}\nAnd the final error is {final_error}")


if "__main__" == __name__:
    processor = ImageProcessor()
    tracking_db = TrackingDB(processor.K, processor.M1, processor.M2)
    tracking_db.load(TRACKING_DB_PATH)
    task_6_1(tracking_db)
    task_6_2(tracking_db)
