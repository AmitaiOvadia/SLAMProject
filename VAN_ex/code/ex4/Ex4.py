import os
import sys
import numpy as np
import cv2
import pickle
from typing import List, Tuple, Dict, Sequence, Optional
from timeit import default_timer as timer
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import networkx as nx

import random
from utils import utils
from utils.visualize import Visualizer
from utils.tracking_database import TrackingDB
# Add the path to the VAN_ex directory to the system path
from ex3.Ex3 import ImageProcessor
import matplotlib.pyplot as plt



AKAZE = 'AKAZE'
SAVE_NAME = "tracking_db_2_acc_y_2_4e-4_blur_1_it_5_akaze"
RANSAC_ACCURACY = 2
Y_DIST_TOLERATION = 2
MIN_RANSAC_ITERATIONS = 5


# Initialize ImageProcessor
def initialize_image_processor(feature_extractor_name=AKAZE,
                               y_dist_threshold=Y_DIST_TOLERATION,
                               accuracy=RANSAC_ACCURACY,
                               min_ransac_iterations=MIN_RANSAC_ITERATIONS):
    return ImageProcessor(feature_extractor_name=feature_extractor_name,
                          y_dist_threshold=y_dist_threshold,
                          accuracy=accuracy,
                          min_ransac_iterations=min_ransac_iterations)


# Initialize TrackingDB
def initialize_tracking_db(K, M1, M2):
    return TrackingDB(K, M1, M2)


def create_matches_list(descriptors_frame_current, descriptors_frame_prev):
    # Create a list of cv2.DMatch objects
    matches = []
    # Ensure the length of both descriptors arrays are the same
    assert len(descriptors_frame_prev) == len(
        descriptors_frame_current), "Descriptor arrays must be of the same length."
    # Iterate over the descriptors and create matches
    for i in range(len(descriptors_frame_prev)):
        match = cv2.DMatch(_queryIdx=i, _trainIdx=i, _imgIdx=0, _distance=0.0)
        matches.append(match)
    return matches


def fill_tracking_db():
    directory_path = os.path.join(utils.DATA_PATH, "image_0")
    num_frames = sum(os.path.isfile(os.path.join(directory_path, f)) for f in os.listdir(directory_path))
    # num_frames = 30 # Set the number of frames you want to process

    # Initialize ImageProcessor and TrackingDB
    processor = initialize_image_processor()
    tracking_db = initialize_tracking_db(processor.K, processor.M1, processor.M2)

    left_descriptors_frames = []
    right_descriptors_frames = []
    matches_dics_frames = []
    left_key_points_frames = []
    right_key_points_frames = []
    for frame_idx in tqdm(range(num_frames)):
        # print(frame_idx)
        left, right = utils.read_images(frame_idx)
        (cur_left_descriptors, cur_right_descriptors, left_key_points,
         matches_dict, right_key_points, matches_init) = processor.get_matches(left, right)

        # update lists
        (left_descriptors_frames.append(cur_left_descriptors),
         right_descriptors_frames.append(cur_right_descriptors))
        matches_dics_frames.append(matches_dict),
        left_key_points_frames.append(left_key_points)
        right_key_points_frames.append(right_key_points)

        # frame 0:
        if frame_idx == 0:
            filtered_left_features, filtered_right_features, links = (
                         tracking_db.create_links(
                         left_features=cur_left_descriptors,
                         right_features=cur_right_descriptors,
                         kp_left=left_key_points,
                         kp_right=right_key_points,
                         matches=matches_init,
                         inliers=None))
            cur_frameId = tracking_db.add_frame(links=links,
                                                left_features=filtered_left_features,
                                                right_features=filtered_right_features,
                                                matches_to_previous_left=None,
                                                total_num_keypoints=None,
                                                num_inliers=None,
                                                inliers=None,
                                                left_extrinsic_cam=None)
            continue

        # find outliers using RANSAC now
        prev_left_descriptors = left_descriptors_frames[frame_idx-1]
        left_0_left_1_matches = utils.find_closest_features(prev_left_descriptors, cur_left_descriptors)

        matches_dict_left0_left1 = processor.create_matches_dict(left_0_left_1_matches)
        left_0_inds, right_0_inds, left_1_inds, right_1_inds = processor.get_points_in_4_images(
            matches_dict_left0_left1,
            matches_dics_frames[frame_idx - 1],
            matches_dics_frames[frame_idx])

        final_left1_extrinsic_mat, num_iterations, pnp_inliers = get_outliers_using_ransac(frame_idx, left_0_inds,
                                                                                           left_1_inds,
                                                                                           left_key_points_frames,
                                                                                           processor, right_0_inds,
                                                                                           right_1_inds,
                                                                                           right_key_points_frames,
                                                                                           accuracy=RANSAC_ACCURACY)
        # left_0_inliers = [left_0_inds[i] for i in pnp_inliers]
        # right_0_inliers = [right_0_inds[i] for i in pnp_inliers]
        left_1_inliers = [left_1_inds[i] for i in pnp_inliers]
        right_1_inliers = [right_1_inds[i] for i in pnp_inliers]

        inliers = np.zeros(len(cur_left_descriptors), dtype=bool)
        inliers[left_1_inliers] = True
        kp_left = [left_key_points[i] for i in left_1_inliers]
        kp_right = [right_key_points[i] for i in right_1_inliers]
        features_left = cur_left_descriptors[left_1_inliers]
        features_right = cur_right_descriptors[right_1_inliers]
        matches_right_left = utils.find_closest_features(features_left, features_right)
        _, _, links = tracking_db.create_links(left_features=features_left,
                                               right_features=features_right,
                                                   kp_left=kp_left,
                                                   kp_right=kp_right,
                                                   matches=matches_right_left,
                                                   inliers=None)

        prev_features = tracking_db.last_features()
        matches_to_previous_left = utils.find_closest_features(prev_features, features_left)
        inliers = remove_track_outliers(final_left1_extrinsic_mat, links, matches_to_previous_left, processor,
                                        tracking_db, accuracy=RANSAC_ACCURACY)
        ratio = np.sum(inliers)/len(left_0_inds)
        # print(f"frame: inliers: {np.sum(inliers)}/{len(matches_to_previous_left)} ratio: {ratio}, num ransac iterations: {num_iterations}")
        cur_frameId = tracking_db.add_frame(links=links,
                                            left_features=features_left,
                                            right_features=features_right,
                                            matches_to_previous_left=matches_to_previous_left,
                                            total_num_keypoints=len(left_1_inds),
                                            num_inliers=np.sum(inliers),
                                            inliers=inliers,
                                            left_extrinsic_cam=final_left1_extrinsic_mat)
    tracking_db.serialize(SAVE_NAME)


def remove_track_outliers(final_left1_extrinsic_mat, links,
                          matches_to_previous_left, processor,
                          tracking_db, accuracy=2):
    links_0 = tracking_db.prev_frame_links
    links_1 = links
    left0_2D_points, right0_2D_points, left1_2D_points, right1_2D_points = [], [], [], []
    for mtch in matches_to_previous_left:
        l0 = mtch.queryIdx
        l1 = mtch.trainIdx
        x_left_0, x_right_0, y_0 = links_0[l0].x_left, links_0[l0].x_right, links_0[l0].y
        y_left_0 = y_right_0 = y_0
        left0_2D_points.append([x_left_0, y_left_0])
        right0_2D_points.append([x_right_0, y_right_0])

        x_left_1, x_right_1, y_1 = links_1[l1].x_left, links_1[l1].x_right, links_1[l1].y
        y_left_1 = y_right_1 = y_1
        left1_2D_points.append([x_left_1, y_left_1])
        right1_2D_points.append([x_right_1, y_right_1])
    left0_2D_points, right0_2D_points, left1_2D_points, right1_2D_points = (np.array(left0_2D_points),
                                                                            np.array(right0_2D_points),
                                                                            np.array(left1_2D_points),
                                                                            np.array(right1_2D_points))
    T_left_1 = processor.get_T(final_left1_extrinsic_mat)
    T_right_1_to_left_1 = processor.get_T(processor.M2)
    T_right_1_to_left_0 = T_right_1_to_left_1 @ T_left_1
    right1_to_left0_extrinsic_mat = processor.get_Rt_from_T(T_right_1_to_left_0)
    left1_P_est = processor.K @ final_left1_extrinsic_mat
    right1_P_est = processor.K @ right1_to_left0_extrinsic_mat
    left0_P = processor.K @ processor.M1
    right0_P = processor.K @ processor.M2
    world_3D_points = utils.triangulate_points(Pa=left0_P, Pb=right0_P,
                                               points_a=left0_2D_points,
                                               points_b=right0_2D_points)
    inliers_inds, _ = processor.check_reprojections(left0_2D_points, right0_2D_points, left1_2D_points,
                                                    right1_2D_points,
                                                    left0_P, right0_P, left1_P_est, right1_P_est, world_3D_points,
                                                    accuracy=accuracy)
    inliers = np.zeros(len(left0_2D_points), dtype=bool)
    inliers[inliers_inds] = True
    # print(f"number of outliers: {len(left0_2D_points) - len(inliers_inds)}")
    return inliers


def get_outliers_using_ransac(frame_idx, left_0_inds, left_1_inds, left_key_points_frames, processor, right_0_inds,
                              right_1_inds, right_key_points_frames, accuracy=None):
    left_0_shared_kpnts = [left_key_points_frames[frame_idx - 1][i] for i in left_0_inds]
    right_0_shared_kpnts = [right_key_points_frames[frame_idx - 1][i] for i in right_0_inds]
    left_1_shared_kpnts = [left_key_points_frames[frame_idx][i] for i in left_1_inds]
    right_1_shared_kpnts = [right_key_points_frames[frame_idx][i] for i in right_1_inds]
    left_0_points_2D, right_0_points_2D = utils.get_points_from_key_points(left_0_shared_kpnts,
                                                                           right_0_shared_kpnts)
    left_1_points_2D, right_1_points_2D = utils.get_points_from_key_points(left_1_shared_kpnts,
                                                                           right_1_shared_kpnts)
    points_3D_0 = utils.triangulate_points(processor.P1, processor.P2, left_0_points_2D, right_0_points_2D)
    final_left1_extrinsic_mat, pnp_inliers, num_iterations = processor.run_RANSAC(world_3D_points=points_3D_0,
                                                                                  left0_2D_points=left_0_points_2D,
                                                                                  right0_2D_points=right_0_points_2D,
                                                                                  left1_2D_points=left_1_points_2D,
                                                                                  right1_2D_points=right_1_points_2D,
                                                                                  K=processor.K,
                                                                                  M_L0=processor.M1,
                                                                                  M_R0=processor.M2,
                                                                                  accuracy=accuracy)
    return final_left1_extrinsic_mat, num_iterations, pnp_inliers



def find_long_track(tracking_db, min_length=6):
    track_ids = tracking_db.all_tracks()
    random.shuffle(track_ids)  # Shuffle the list to randomize the selection
    for trackId in track_ids:
        if len(tracking_db.frames(trackId)) >= min_length:
            return trackId
    return None  # Return None if no such track is found


def display_long_track_features(tracking_db, track_length=12):
    directory_path_0 = os.path.join(utils.DATA_PATH, "image_0")
    all_frames_left = sorted(os.listdir(directory_path_0))

    trackId = find_long_track(tracking_db, min_length=track_length)
    if trackId is None:
        print("No track of length ≥ 6 found.")
        return

    # Display the feature locations on all the relevant (left) images
    frames = tracking_db.frames(trackId)[:track_length]

    fig, axes = plt.subplots(len(frames), 2, figsize=(10, len(frames)))
    fig.suptitle(f"Feature Tracking for Track #{trackId}", fontsize=16)

    for i, frameId in enumerate(frames):
        image_path = os.path.join(directory_path_0, all_frames_left[frameId])
        if not image_path:
            print(f"Image file for frame {frameId} not found.")
            continue

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to load image for frame {frameId}.")
            continue

        link = tracking_db.link(frameId, trackId)
        feature_loc = (int(link.x_left), int(link.y))
        top_left = (max(0, feature_loc[0] - 10), max(0, feature_loc[1] - 10))
        bottom_right = (min(image.shape[1], feature_loc[0] + 10), min(image.shape[0], feature_loc[1] + 10))

        patch = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]].copy()

        # Plot the full image with the hollow circle marker
        from matplotlib.patches import Circle
        axes[i, 0].imshow(image, cmap='gray')
        circle = Circle(feature_loc, radius=20, edgecolor='r', facecolor='none')
        axes[i, 0].add_patch(circle)
        axes[i, 0].axis('off')

        # Plot the zoomed-in patch with the red X marker
        axes[i, 1].imshow(patch, cmap='gray', aspect='equal')
        axes[i, 1].plot(10, 10, 'rx')
        axes[i, 1].set_xlim(0, patch.shape[1])
        axes[i, 1].set_ylim(patch.shape[0], 0)
        axes[i, 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust the layout to make room for the title
    plt.savefig('ex4_tracking.png')
    # plt.show()

def create_trivial_dict(n):
    matches_dict = {}
    for i in range(n):
        matches_dict[i] = i
    return matches_dict


def connectivity_graph(tracking_db):
    all_frames = tracking_db.all_frames()
    outgoing_tracks_counts = []

    for frameId in all_frames[:-1]:  # Exclude the last frame since it has no next frame
        next_frameId = frameId + 1
        current_tracks = set(tracking_db.tracks(frameId))
        next_tracks = set(tracking_db.tracks(next_frameId))
        outgoing_tracks = current_tracks.intersection(next_tracks)
        num_outgoing_tracks = len(outgoing_tracks)
        outgoing_tracks_counts.append(num_outgoing_tracks)

    mean = np.mean(outgoing_tracks_counts)
    plt.figure(figsize=(14, 6))
    plt.plot(all_frames[:-1], outgoing_tracks_counts, label='Outgoing Tracks')
    plt.axhline(y=mean, color='g', linestyle='-', label='Mean Outgoing Tracks')
    plt.xlabel('Frame')
    plt.ylim([0, max(outgoing_tracks_counts)*1.1])
    plt.ylabel('Outgoing Tracks')
    plt.title(f"Connectivity\nmean: {mean:.5g}, minimum: {min(outgoing_tracks_counts)}")
    plt.legend()
    plt.savefig('connectivity.png')
    # plt.show()


def display_percentage_of_inliers_per_frame(tracking_db):
    percentage_of_inliers_per_frame = tracking_db.inlers_per_frame
    mean = np.mean(percentage_of_inliers_per_frame)
    plt.figure()
    plt.plot(percentage_of_inliers_per_frame, label='percentage of inliers per frame')
    plt.axhline(y=mean, color='g', linestyle='-',
                label=f'Mean percentage {mean:.4g}')
    plt.xlabel('Frame')
    plt.ylabel('percentage of inliers')
    plt.title('percentage of inliers per frame')
    plt.ylim(0, 110)
    plt.legend()
    plt.savefig('percentage of inliers per frame.png')
    pass


def track_length_histogram(tracking_db):
    # Present a track length histogram graph
    trackId_to_frames = tracking_db.trackId_to_frames
    num_tracks = len(trackId_to_frames)
    all_track_lengths = []
    for trackId in range(num_tracks):
        track_length = len(trackId_to_frames[trackId])
        all_track_lengths.append(track_length)
    all_track_lengths = np.array(all_track_lengths)
    plt.figure()
    plt.hist(all_track_lengths, bins=100, log=True, color='blue')
    plt.xlabel('Track length')
    # plt.xlim(0, 80)
    plt.ylabel('Number of Tracks')
    plt.title('Track Lengths Histogram')
    plt.savefig('Track Lengths Histogram.png')
    # plt.show()


def shift_camera_extrinsic_matrix(Rt, delta_t):
    t = Rt[:, 3]
    R = Rt[:, :3]
    # Ensure t is a column vector
    # t = t.reshape((3, 1))

    # Update the translation vector
    t += delta_t
    t = t.reshape((3, 1))
    # Construct the new extrinsic matrix
    extrinsic_matrix = np.hstack((R, t))

    return extrinsic_matrix


def check_reprojections_along_track(min_length, tracking_db):
    trackId = find_long_track(tracking_db, min_length=min_length)
    present_track(0, trackId, tracking_db)
    present_track(-1, trackId, tracking_db)


def present_track(frame_ind_for_3D, trackId, tracking_db):
    # Display the feature locations on all the relevant (left) images
    track = tracking_db.track(trackId)
    frames = tracking_db.frames(trackId)
    processor = initialize_image_processor()
    translation_vector = processor.M2[:, 3]
    K = processor.K
    # Define the translation vector (0.53 meters shift in the x-axis)
    left_cameras = np.array(processor.load_all_ground_truth_camera_matrices(processor.GROUND_TRUTH_PATH))[frames]
    # Initialize the right_cameras array
    right_cameras = create_right_cameras_Rt(left_cameras, translation_vector)
    # all 2D tracks
    left_track_2D = []
    right_track_2D = []
    for frame in frames:
        track_frame = track[frame]
        x_left = track_frame.x_left
        x_right = track_frame.x_right
        y = track_frame.y
        left_track_2D.append([x_left, y])
        right_track_2D.append([x_right, y])
    left_track_2D, right_track_2D = np.array(left_track_2D), np.array(right_track_2D)
    point_3D = utils.triangulate_pair(Pa=K @ left_cameras[frame_ind_for_3D],
                                      Pb=K @ right_cameras[frame_ind_for_3D],
                                      point_a=left_track_2D[frame_ind_for_3D],
                                      point_b=right_track_2D[frame_ind_for_3D])
    # reproject to all the cameras
    reprojection_errors = []
    for frame in range(len(frames)):
        P_left = K @ left_cameras[frame]
        P_right = K @ right_cameras[frame]

        point_2d_left_reprojected = processor.reproject(P_left, np.array([point_3D]))[0]
        point_2d_left_original = left_track_2D[frame]
        rep_error_left = np.linalg.norm(point_2d_left_reprojected - point_2d_left_original)

        point_2d_right_reprojected = processor.reproject(P_right, np.array([point_3D]))[0]
        point_2d_right_original = right_track_2D[frame]
        rep_error_right = np.linalg.norm(point_2d_right_reprojected - point_2d_right_original)

        reprojection_errors.append([rep_error_left, rep_error_right])
    reprojection_errors = np.array(reprojection_errors)
    trianglated_frame = "last" if frame_ind_for_3D == -1 else "first"
    if trianglated_frame == "last":
        reprojection_errors = reprojection_errors[::-1]
    plt.figure(figsize=(8, 6))
    # Plotting the reprojection errors
    plt.plot(reprojection_errors[:, 0], label='Left', color='b')
    plt.plot(reprojection_errors[:, 1], label='Right', color='orange')
    # Adding titles and labels
    plt.title(f'PnP - projection error vs track length\ntriangulate using {trianglated_frame} frame\ntrack ID: {trackId}')
    plt.xlabel('distance from reference')
    plt.ylabel('projection error (pixels)')
    plt.legend()
    # Show the plot
    plt.savefig(f"reprojection errors along track, {trianglated_frame} frame.png")


def create_right_cameras_Rt(left_cameras, translation_vector):
    right_cameras = np.zeros_like(left_cameras)
    # left_cameras = np.array([processor.M1])
    # Calculate the right camera matrices
    for i in range(len(left_cameras)):
        R = left_cameras[i, :, :3]  # Extract the rotation matrix
        t = left_cameras[i, :, 3]  # Extract the translation vector

        # Apply the translation in the camera coordinate system
        new_t = t + translation_vector

        # Construct the right camera matrix
        right_cameras[i, :, :3] = R
        right_cameras[i, :, 3] = new_t
    return right_cameras


def Q42(tracking_db):
    tracking_db.present_tracking_statistics()


def Q43(tracking_db):
    display_long_track_features(tracking_db)


def Q44(tracking_db):
    connectivity_graph(tracking_db)


def Q45(tracking_db):
    display_percentage_of_inliers_per_frame(tracking_db)


def Q46(tracking_db):
    track_length_histogram(tracking_db)


def Q47(tracking_db):
    min_length = 50
    check_reprojections_along_track(tracking_db=tracking_db,
                                    min_length=min_length)


def show_camera_centers(tracking_db, processor):

    camera_centers_dict = tracking_db.frameId_to_camera_center[0]
    num_frames = len(camera_centers_dict)

    true_Rt = processor.load_all_ground_truth_camera_matrices(processor.GROUND_TRUTH_PATH)[:num_frames]
    ground_truth_centers = [utils.get_camera_center_from_Rt(Rt) for Rt in true_Rt]

    all_centers = []
    for i in range(num_frames):
        center_i = camera_centers_dict[i]
        all_centers.append(center_i)
    all_centers, ground_truth_centers =np.array(all_centers), np.array(ground_truth_centers)
    Visualizer.plot_trajectories_plt(all_centers, ground_truth_centers)


if __name__ == '__main__':
    fill_tracking_db()
    processor = initialize_image_processor()
    tracking_db = TrackingDB(processor.K, processor.M1, processor.M2)
    tracking_db.load(SAVE_NAME)
    show_camera_centers(tracking_db, processor)

    Q42(tracking_db)
    Q43(tracking_db)
    Q44(tracking_db)
    Q45(tracking_db)
    Q46(tracking_db)
    Q47(tracking_db)




