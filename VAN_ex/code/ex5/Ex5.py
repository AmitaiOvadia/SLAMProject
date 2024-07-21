import gtsam.utils.plot
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.ndimage import median_filter
from matplotlib.patches import Circle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.tracking_database import TrackingDB
from utils.BundleAdjusment import BundleAdjusment, Bundelon
from ex3.Ex3 import ImageProcessor
from utils import utils
import cv2
TRACKING_DB_PATH = '../ex4/tracking_db_1.5_acc'
DATA_PATH = r"/cs/labs/tsevi/amitaiovadia/SLAMProject/VAN_ex/dataset/sequences/00"
import matplotlib
matplotlib.use('TKAgg')


def get_factors_errors(factors, values):
    errors = np.array([factor.error(values) for factor in factors])
    return errors


def read_images(idx):
    """
    Reads a pair of images from the dataset.

    Parameters:
    idx (int): Index of the image pair.

    Returns:
    tuple: A tuple containing the left and right images.
    """
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(DATA_PATH + '/image_0/' + img_name, 0)
    img2 = cv2.imread(DATA_PATH + '/image_1/' + img_name, 0)
    return img1, img2


def calculate_reprojection_error(measured, projected):
    x_measured, y_measured = measured.uL(), measured.v()
    x_projected, y_projected = projected.uL(), projected.v()
    error = np.sqrt((x_measured - x_projected)**2 + (y_measured - y_projected)**2)
    return error

def triangulate_track_gtsam(frames, links, tracking_db):
    left_cameras_relative_Rts = [tracking_db.frameId_to_relative_extrinsic_Rt[frame] for frame in frames]
    left_cameras_from_first_frame = BundleAdjusment.composite_relative_Rts(left_cameras_relative_Rts)
    # get gtsm Rts (from camera to world)
    left_cameras_from_first_frame_gtsam = [BundleAdjusment.flip_Rt(left_Rt)  # flip from world -> cam to cam -> world
                                           for left_Rt in left_cameras_from_first_frame]
    # set K
    fx = tracking_db.K[0, 0]
    fy = tracking_db.K[1, 1]
    cx = tracking_db.K[0, 2]
    cy = tracking_db.K[1, 2]
    s = tracking_db.K[0, 1]
    baseline = -tracking_db.M2[0, -1]
    calibration = gtsam.Cal3_S2Stereo(fx, fy, s, cx, cy, baseline)
    left_camera_last_frame_gtsam = left_cameras_from_first_frame_gtsam[-1]
    t = left_camera_last_frame_gtsam[:, 3]
    gtsam_left_camera_pose = gtsam.Pose3(left_camera_last_frame_gtsam)
    gtsam_frame_to_triangulate_from = gtsam.StereoCamera(gtsam_left_camera_pose, calibration)
    last_frame_link = links[-1]
    gtsam_stereo_point2_for_triangulation = gtsam.StereoPoint2(last_frame_link.x_left,
                                                               last_frame_link.x_right,
                                                               last_frame_link.y)
    gtsam_p3d = gtsam_frame_to_triangulate_from.backproject(gtsam_stereo_point2_for_triangulation)
    values = gtsam.Values()
    p3d_sym = gtsam.symbol("q", 0)
    values.insert(p3d_sym, gtsam_p3d)
    left_pnt2 = []
    right_pnt2 = []
    left_projections = []
    right_projections = []
    factors = []
    for relative_frame_ind in range(len(left_cameras_from_first_frame_gtsam)):
        left_pose_sym = gtsam.symbol("c", relative_frame_ind)
        cur_cam_pose = left_cameras_from_first_frame_gtsam[relative_frame_ind]
        gtsam_left_cam_pose = gtsam.Pose3(cur_cam_pose)
        values.insert(left_pose_sym, gtsam_left_cam_pose)
        cur_link = links[relative_frame_ind]
        gtsam_measurment_pt2 = gtsam.StereoPoint2(cur_link.x_left, cur_link.x_right, cur_link.y)
        gtsam_frame = gtsam.StereoCamera(gtsam_left_cam_pose, calibration)
        gtsam_preojected_stereo_point2 = gtsam_frame.project(gtsam_p3d)
        xl, xr, y = (gtsam_preojected_stereo_point2.uL(),
                     gtsam_preojected_stereo_point2.uR(),
                     gtsam_preojected_stereo_point2.v())

        projection_uncertaincy = gtsam.noiseModel.Isotropic.Sigma(dim=3, sigma=1.0)

        factor = gtsam.GenericStereoFactor3D(gtsam_measurment_pt2,
                                             projection_uncertaincy,
                                             gtsam.symbol("c", relative_frame_ind),
                                             p3d_sym,
                                             calibration)

        left_pnt2.append([cur_link.x_left, cur_link.y])
        right_pnt2.append([cur_link.x_right, cur_link.y])

        left_projections.append([xl, y])
        right_projections.append([xr, y])

        factors.append(factor)
    # find the reprojection error for each frame
    left_pnt2, right_pnt2, left_projections, right_projections = (np.array(left_pnt2), np.array(right_pnt2),
                                                                  np.array(left_projections),
                                                                  np.array(right_projections))
    return factors, left_pnt2, left_projections, right_pnt2, right_projections, values


def display_factor_errors_vs_reprojection_erros(factor_errors, reprojection_errors, track_id):
    plt.figure(figsize=(8, 6))
    # Plotting the reprojection errors
    plt.plot(reprojection_errors, factor_errors,
             label='factor errors as a fucntion of reprojection erros', color='b')
    plt.xlabel("reprojection erros")
    plt.ylabel("factor erros")
    # Adding titles and labels
    plt.title(
        f'factor errors as a fucntion of reprojection erros\nusing the estimated extrinsic matrices\ntrack ID: {track_id}')
    plt.legend()
    # Show the plot
    plt.savefig(f"factor errors as a fucntion of reprojection erros.png")


def display_factor_erros(factor_errors, track_id):
    plt.figure(figsize=(8, 6))
    # Plotting the reprojection errors
    plt.plot(factor_errors, label='factor errors', color='b')
    # Adding titles and labels
    plt.title(
        f'projection error vs track length\nusing the estimated extrinsic matrices\ntrack ID: {track_id}')
    plt.xlabel('distance from reference')
    plt.ylabel('projection error (pixels)')
    plt.legend()
    # Show the plot
    plt.savefig(f"reprojection errors along track.png")


def display_3d_trajectory_gtsam_function(graph, initial_estimates, optimized_values):
    # display the trajectories using gtsam fucntion
    plt.figure()
    marginals_results = gtsam.Marginals(graph, optimized_values)
    gtsam.utils.plot.plot_trajectory(fignum=0, marginals=marginals_results, values=optimized_values)
    # Get the current axis
    ax = plt.gca()
    # Flip the y-axis
    # ax.set_ylim(ax.get_ylim()[::-1])
    # Set the labels to make the x-z plane the 'world' plane
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    # Adjust the aspect ratio if needed
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1
    ax.view_init(elev=30, azim=90)  # Adjust the elevation and azimuth to get the desired view
    # Save the figure
    plt.savefig("gtsam_3D_trajectory_for_first_bundle_window.png", dpi=600)


def display_initial_vs_optimized_measured_and_reprojected_worse_factor(left_image, q_measured, q_projection_final,
                                                                       q_projection_initial, reprojection_error_final,
                                                                       reprojection_error_initial, right_image):
    # Create a figure with a 1x2 grid of subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    fig.suptitle(f"Stereo Points Tracking\nReprojection Error Initial: {reprojection_error_initial:.2f}, "
                 f"Reprojection Error Final: {reprojection_error_final:.2f}", fontsize=16)

    # Function to plot the stereo images with measured points
    def plot_stereo_images(ax_left, ax_right, left_image, right_image, measured, initial, final, title,
                           markersize=15, padding=40, markeredgewidth=3):
        x_left_measured, x_right_measured, y_measured = measured.uL(), measured.uR(), measured.v()
        x_left_initial, x_right_initial, y_initial = initial.uL(), initial.uR(), initial.v()
        x_left_final, x_right_final, y_final = final.uL(), final.uR(), final.v()

        # Calculate bounding box for the region of interest (ROI) with padding
        x_left_min = int(min(x_left_measured, x_left_initial, x_left_final) - padding)
        x_left_max = int(max(x_left_measured, x_left_initial, x_left_final) + padding)
        y_left_min = int(min(y_measured, y_initial, y_final) - padding)
        y_left_max = int(max(y_measured, y_initial, y_final) + padding)

        x_right_min = int(min(x_right_measured, x_right_initial, x_right_final) - padding)
        x_right_max = int(max(x_right_measured, x_right_initial, x_right_final) + padding)
        y_right_min = int(min(y_measured, y_initial, y_final) - padding)
        y_right_max = int(max(y_measured, y_initial, y_final) + padding)

        # Ensure bounding box is within image bounds
        x_left_min = max(0, x_left_min)
        x_left_max = min(left_image.shape[1], x_left_max)
        y_left_min = max(0, y_left_min)
        y_left_max = min(left_image.shape[0], y_left_max)

        x_right_min = max(0, x_right_min)
        x_right_max = min(right_image.shape[1], x_right_max)
        y_right_min = max(0, y_right_min)
        y_right_max = min(right_image.shape[0], y_right_max)

        # Crop the images based on the ROI
        roi_left = left_image[y_left_min:y_left_max, x_left_min:x_left_max]
        roi_right = right_image[y_right_min:y_right_max, x_right_min:x_right_max]

        # Transform coordinates to the ROI coordinate system
        def transform_to_roi(x, y, x_min, y_min):
            return x - x_min, y - y_min

        feature_loc_left_measured = transform_to_roi(x_left_measured, y_measured, x_left_min, y_left_min)
        feature_loc_left_initial = transform_to_roi(x_left_initial, y_initial, x_left_min, y_left_min)
        feature_loc_left_final = transform_to_roi(x_left_final, y_final, x_left_min, y_left_min)

        feature_loc_right_measured = transform_to_roi(x_right_measured, y_measured, x_right_min, y_right_min)
        feature_loc_right_initial = transform_to_roi(x_right_initial, y_initial, x_right_min, y_right_min)
        feature_loc_right_final = transform_to_roi(x_right_final, y_final, x_right_min, y_right_min)

        ax_left.imshow(roi_left, cmap='gray')
        ax_left.plot(*feature_loc_left_measured, 'rx', label='Measured', markersize=markersize,
                     markeredgewidth=markeredgewidth)
        ax_left.plot(*feature_loc_left_initial, 'gx', label='Initial', markersize=markersize,
                     markeredgewidth=markeredgewidth)
        ax_left.plot(*feature_loc_left_final, 'bx', label='Final', markersize=markersize,
                     markeredgewidth=markeredgewidth)
        ax_left.set_title(title + " - Left")
        ax_left.legend()
        ax_left.axis('off')

        ax_right.imshow(roi_right, cmap='gray')
        ax_right.plot(*feature_loc_right_measured, 'rx', label='Measured', markersize=markersize,
                      markeredgewidth=markeredgewidth)
        ax_right.plot(*feature_loc_right_initial, 'gx', label='Initial', markersize=markersize,
                      markeredgewidth=markeredgewidth)
        ax_right.plot(*feature_loc_right_final, 'bx', label='Final', markersize=markersize,
                      markeredgewidth=markeredgewidth)
        ax_right.set_title(title + " - Right")
        ax_right.legend()
        ax_right.axis('off')

    # Plot the projections on the stereo images
    plot_stereo_images(axes[0], axes[1], left_image, right_image, q_measured, q_projection_initial, q_projection_final,
                       "Projection")
    # Adjust the layout and save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust the layout to make room for the title
    plt.savefig('stereo_points_tracking.png')


def filter_landmarks(all_landmarks, x_range, z_range, tolerance=50):
    x_min, x_max = x_range
    z_min, z_max = z_range

    # Adjust the ranges with the given tolerance
    x_min -= tolerance
    x_max += tolerance
    z_min -= tolerance
    z_max += tolerance

    # Apply the filter
    filtered_landmarks = all_landmarks[
        (all_landmarks[:, 0] >= x_min) & (all_landmarks[:, 0] <= x_max) &
        (all_landmarks[:, 1] >= z_min) & (all_landmarks[:, 1] <= z_max)
        ]

    return filtered_landmarks


def display_bundle(bundle_object, processor, tracking_db,
                   title="",
                   first_bundle_window=False,
                   add_ground_truth=True,
                   only_key_frames=True,
                   add_initial_guesses=True,
                   add_landmarks=True):
    key_frames = np.array(bundle_object.key_frames[:-1])
    camera_centers_initial_guesses = np.array([tracking_db.frameId_to_camera_center[0][frame] for frame in
                                               range(len(tracking_db.frameId_to_camera_center[0]))])
    all_landmarks, key_frames_camera_centers = bundle_object.get_camera_centers_and_landmarks()
    all_estimated_camera_centers = np.array(bundle_object.get_all_camera_centers())
    bundle_size = len(all_estimated_camera_centers)
    true_Rt = processor.load_all_ground_truth_camera_matrices(processor.GROUND_TRUTH_PATH)
    ground_truth_centers = np.array([utils.get_camera_center_from_Rt(Rt) for Rt in true_Rt])

    if only_key_frames:
        all_estimated_camera_centers = all_estimated_camera_centers[key_frames]
        ground_truth_centers = ground_truth_centers[key_frames]

    x_min, y_min, z_min = np.min(all_estimated_camera_centers, axis=0)
    x_max, y_max, z_max = np.max(all_estimated_camera_centers, axis=0)

    # Select the desired dimensions
    all_landmarks = all_landmarks[:, [0, 2]]
    all_landmarks = filter_landmarks(all_landmarks, (x_min, x_max), (z_min, z_max), tolerance=20)
    camera_centers_initial_guesses = camera_centers_initial_guesses[:bundle_size, [0, 2]]
    all_estimated_camera_centers = all_estimated_camera_centers[:bundle_size, [0, 2]]
    key_frames_camera_centers = key_frames_camera_centers[:bundle_size, [0, 2]]  # project to x-z
    ground_truth_centers = ground_truth_centers[:bundle_size, [0, 2]]  # project to x-z
    # Generate colors that change gradually with the index
    camera_colors = plt.cm.Blues(np.linspace(0.3, 1, len(key_frames_camera_centers)))
    ground_truth_colors = plt.cm.Reds(np.linspace(0.6, 1, len(ground_truth_centers)))
    plt.figure(figsize=(10, 6))
    s = 1

    if add_landmarks:
        # add landmarks
        plt.scatter(all_landmarks[:, 0], all_landmarks[:, 1], color='orange', label='landmarks', s=0.01)

    if not only_key_frames:
        # Plot camera centers
        plt.scatter(all_estimated_camera_centers[:, 0], all_estimated_camera_centers[:, 1],
                    label='centers after bundle', s=2)

    if add_initial_guesses:
        # Plot camera centers
        plt.scatter(camera_centers_initial_guesses[:, 0], camera_centers_initial_guesses[:, 1],
                    label='initial guesses', s=2, color='yellow')

    if add_ground_truth:
        # Plot ground truth centers
        plt.scatter(ground_truth_centers[:, 0], ground_truth_centers[:, 1], color=ground_truth_colors,
                    label='Ground Truth Centers', s=5)
    # Plot key frames
    plt.scatter(key_frames_camera_centers[:, 0], key_frames_camera_centers[:, 1], color='navy',
                label='Key Frames Camera Centers', s=10)

    title_add = " first bundle window" if first_bundle_window else ""
    save_name = f"{title}{title_add}.png"
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title(f"{title}{title_add}")
    plt.legend()
    plt.savefig(save_name, dpi=600)
    print(f"saved to {save_name}")


def task_5_1(tracking_db):
    track_id, frames, links = tracking_db.find_random_track(min_length=30, max_length=None, shuffle=True)
    factors, left_pnt2, left_projections, right_pnt2, right_projections, values = triangulate_track_gtsam(frames, links,
                                                                                                          tracking_db)
    left_errors = np.linalg.norm(left_pnt2 - left_projections, axis=1)
    right_errors = np.linalg.norm(right_pnt2 - right_projections, axis=1)

    reprojection_errors = (left_errors + right_errors) / 2
    factor_errors = get_factors_errors(factors, values)
    # coefficients = np.polyfit(reprojection_errors, factor_errors, 2)
    display_factor_erros(factor_errors[::-1], track_id)
    display_factor_errors_vs_reprojection_erros(factor_errors, reprojection_errors, track_id)


def task_5_3(tracking_db):
    size = 12
    first_bundelon = Bundelon(tracking_db, frames=np.arange(0, size))
    first_bundelon.create_bundelon_factor_graph()
    initial_estimates = first_bundelon.get_initial_estimates()
    initial_bad_factor = first_bundelon.highest_error_factor
    initial_factor_error = initial_bad_factor.error(first_bundelon.get_initial_estimates())

    total_initial_error = first_bundelon.get_total_graph_error()
    average_initial_graph_error = first_bundelon.get_average_graph_error()

    print(f"the number of factors is {first_bundelon.get_factor_graph().size()}")

    first_bundelon.optimize_bundelon()

    total_final_error = first_bundelon.get_total_graph_error()
    average_final_graph_error = first_bundelon.get_average_graph_error()
    optimized_values = first_bundelon.get_optimized_values_gtsam_format()

    # Print the total factor graph error before and after the optimization.
    print(f"total factor graph error * before * optimization: {total_initial_error}")
    print(f"total factor graph error * after * optimization: {total_final_error}")
    print()
    print(f"average factor graph error * before * optimization: {average_initial_graph_error}")
    print(f"average factor graph error * after * optimization: {average_final_graph_error}")

    # get the factor before optimization
    bad_factor_cam_key, bad_factor_point_key = initial_bad_factor.keys()
    left_camera = first_bundelon._frameId_to_left_camera_gtsam[0]
    gtsam_left_camera_pose = gtsam.Pose3(left_camera)
    bad_factor_camera_before = gtsam.StereoCamera(gtsam_left_camera_pose, first_bundelon._gtsam_calibration)
    q_measured = initial_bad_factor.measured()

    gtsam_camera_after_optimization = optimized_values.atPose3(bad_factor_cam_key)
    camera_after_optimization = gtsam.StereoCamera(gtsam_camera_after_optimization, first_bundelon._gtsam_calibration)

    final_factor_error = initial_bad_factor.error(optimized_values)
    print()
    print(f"initial factor error: {initial_factor_error}")
    print(f"factor error after optimization: {final_factor_error}")

    # before
    q_initial_estimate_3D = initial_estimates.atPoint3(bad_factor_point_key)
    q_projection_initial = bad_factor_camera_before.project(q_initial_estimate_3D)

    # after
    q_final_estimate_3D = optimized_values.atPoint3(bad_factor_point_key)
    q_projection_final = camera_after_optimization.project(q_final_estimate_3D)

    left_image, right_image = read_images(0)

    reprojection_error_initial = calculate_reprojection_error(q_measured, q_projection_initial)
    reprojection_error_final = calculate_reprojection_error(q_measured, q_projection_final)

    display_initial_vs_optimized_measured_and_reprojected_worse_factor(left_image, q_measured, q_projection_final,
                                                                       q_projection_initial, reprojection_error_final,
                                                                       reprojection_error_initial, right_image)

    graph = first_bundelon.get_factor_graph()
    display_3d_trajectory_gtsam_function(graph, initial_estimates, optimized_values)

    # do bundle with first window only
    bundle_object = BundleAdjusment(tracking_db, calculate_only_first_bundle=True)
    bundle_object.create_and_solve_all_bundle_windows()

    display_bundle(bundle_object, processor, tracking_db, first_bundle_window=True)


def task_5_4(tracking_db, processor):
    bundle_object = BundleAdjusment(tracking_db)
    bundle_object.create_and_solve_all_bundle_windows()

    # or the last bundle window print the position of the first frame of that bundle in the result
    # of the optimization. (i.e. the location after the optimization)
    show_last_budle_factor_prior(bundle_object)

    # Present a view from above (2d) of the scene, with all keyframes (left camera only, no
    # need to present the right camera of the frame) and 3D points.
    display_bundle(bundle_object, processor, tracking_db,
                   first_bundle_window=False,
                   add_ground_truth=False,
                   only_key_frames=True,
                   add_initial_guesses=False,
                   add_landmarks=True,
                   title="view from above (2d) "
                         "with all keyframes and 3D points")

    # Overlay the estimated keyframes with the Ground Truth poses of the keyframes
    display_bundle(bundle_object, processor, tracking_db,
                   first_bundle_window=False,
                   add_ground_truth=True,
                   only_key_frames=True,
                   add_initial_guesses=False,
                   add_landmarks=True,
                   title="view from above (2d) "
                         "with all keyframes, landmarks and ground truth")

    # display all the frames including initial guesses
    display_bundle(bundle_object, processor, tracking_db,
                   first_bundle_window=False,
                   add_ground_truth=True,
                   only_key_frames=False,
                   add_initial_guesses=True,
                   add_landmarks=True,
                   title="camera centers 2D full display")

    # Present the keyframe localization error in meters (location difference only - Euclidean
    # distance) over time
    display_localization_erros_vs_ground_truth(bundle_object, processor)


def display_localization_erros_vs_ground_truth(bundle_object, processor, use_key_frames=False):
    key_frames = bundle_object.key_frames
    all_estimated_camera_centers = np.array(bundle_object.get_all_camera_centers())
    true_Rt = processor.load_all_ground_truth_camera_matrices(processor.GROUND_TRUTH_PATH)
    ground_truth_centers = np.array([utils.get_camera_center_from_Rt(Rt) for Rt in true_Rt])
    if use_key_frames:
        all_estimated_camera_centers = all_estimated_camera_centers[key_frames]
        ground_truth_centers = ground_truth_centers[key_frames]
    l2_localization_error = np.linalg.norm(all_estimated_camera_centers - ground_truth_centers, axis=-1)
    plt.figure()
    plt.plot(l2_localization_error,
             label='l2 localization errors as a fucntion of frames', color='b')
    plt.xlabel("frames")
    plt.ylabel("localization errors from ground truth")
    # Adding titles and labels
    plt.title("keyframe localization error in meters over time")
    plt.legend()
    # Show the plot
    plt.savefig(f"keyframe localization error in meters over time.png")
    # plt.show()
    a=0


def show_last_budle_factor_prior(bundle_object):
    last_bundle_key_frame = bundle_object.key_frames[-2]
    last_bundle_after_optimization = bundle_object.key_frames_to_bundelons[last_bundle_key_frame]
    camera = gtsam.symbol("c", last_bundle_key_frame)
    optimized_values = last_bundle_after_optimization.get_optimized_values_gtsam_format()
    camera_pose = optimized_values.atPose3(camera)
    graph = last_bundle_after_optimization.get_factor_graph()  # Your factor graph
    print(f"camera pose is: {camera_pose}")
    for i in range(graph.size()):
        factor = graph.at(i)
        # Check if the factor involves the 'camera'
        if camera in factor.keys():
            # Print or store the factor associated with the camera
            print(f"factor error is: {factor.error(optimized_values):.4g}")
            # If you only need the first associated factor, you can break here
            break


if "__main__" == __name__:
    processor = ImageProcessor()
    tracking_db = TrackingDB(processor.K, processor.M1, processor.M2)
    tracking_db.load(TRACKING_DB_PATH)
    # task_5_1(tracking_db)
    # task_5_3(tracking_db)
    task_5_4(tracking_db, processor)
