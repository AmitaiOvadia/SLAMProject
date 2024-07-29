import os
import sys
import numpy as np
import cv2
import time
import multiprocessing as mp
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go

# Add the path to the VAN_ex directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import utils
from utils.visualize import Visualizer
from ex2.Ex2 import *
MAX_RANSAC_ITERATIONS = 1000

AKAZE = 'AKAZE'
SIFT = 'SIFT'
ORB = 'ORB'
BRISK = 'BRISK'


class ImageProcessor:
    def __init__(self, feature_extractor_name=AKAZE, y_dist_threshold=1, accuracy=1, min_ransac_iterations=50):
        self.K, self.M1, self.M2 = utils.read_cameras()
        self.P1, self.P2 = self.K @ self.M1, self.K @ self.M2
        self.min_ransac_iterations = min_ransac_iterations
        self.feature_extractor_name = feature_extractor_name
        self.y_dist_threshold = y_dist_threshold
        self.accuracy = accuracy
        self.GROUND_TRUTH_PATH = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'VAN_ex', 'dataset', 'poses', '00.txt'))
        self.NUMBER_OF_POINTS_FOR_PNP = 4
        self.PNP_FAILED = 2

    @staticmethod
    def create_matches_dict(matches):
        matches_dict = dict()
        for match in matches:
            left_ind = match.queryIdx
            right_ind = match.trainIdx
            matches_dict[left_ind] = right_ind
        return matches_dict

    def get_points_in_4_images(self, matches_dict_left0_left1, matches_dict_0, matches_dict_1):
        left_0_inds, right_0_inds, left_1_inds, right_1_inds = [], [], [], []
        for left_0_keypoint in matches_dict_left0_left1:
            left_1_keypoint = matches_dict_left0_left1[left_0_keypoint]  # find the corresponding match in left 1 image

            # find the right_0 corresponding to it
            if left_0_keypoint in matches_dict_0:
                right_0_keypoint = matches_dict_0[left_0_keypoint]
            else:
                continue
            if left_1_keypoint in matches_dict_1:
                right_1_keypoint = matches_dict_1[left_1_keypoint]
            else:
                continue
            left_0_inds.append(left_0_keypoint)
            right_0_inds.append(right_0_keypoint)
            left_1_inds.append(left_1_keypoint)
            right_1_inds.append(right_1_keypoint)
        return left_0_inds, right_0_inds, left_1_inds, right_1_inds

    @staticmethod
    def rodriguez_to_mat(rvec, tvec):
        rot, _ = cv2.Rodrigues(rvec)
        return np.hstack((rot, tvec))

    def get_R_t_using_pnp(self, points_3D_world, points_2d_image, K, flag):
        success, rotation_vector, translation_vector = cv2.solvePnP(points_3D_world, points_2d_image, K, None, flags=flag)
        if not success:
            return self.PNP_FAILED
        R_t = self.rodriguez_to_mat(rotation_vector, translation_vector)
        return R_t

    @staticmethod
    def get_T(R_t):
        T = np.vstack((R_t, [0, 0, 0, 1]))
        return T

    @staticmethod
    def get_Rt_from_T(T):
        Rt = T[:-1]
        return Rt

    def do_extrinsic_mat_transform(self, first_ex_mat, second_ex_mat):
        """
        find the new extrinsic matrix.
        first_cam_mat : A -> B
        second_cam_mat : B -> C
        T : A -> C
        """
        T = self.get_T(first_ex_mat)
        return second_ex_mat @ T

    @staticmethod
    def to_homogeneous(points):
        return np.column_stack((points, np.ones((len(points), 1))))

    @staticmethod
    def from_homogeneous(points):
        points = points[:, :-1] / points[:, -1][:, np.newaxis]
        return points

    @staticmethod
    def reproject(P, points_3D):
        points_3D_hom = ImageProcessor.to_homogeneous(points_3D)
        points_2D_hom = (P @ points_3D_hom.T).T
        reprojected_points = ImageProcessor.from_homogeneous(points_2D_hom)
        return reprojected_points

    def check_reprojections(self, left0_2D, right0_2D, left1_2D, right1_2D,
                            left0_P, right0_P, left1_P, right1_P, points_3D_world, accuracy=None):
        all_2D_points = [left0_2D, right0_2D, left1_2D, right1_2D]
        all_ps = [left0_P, right0_P, left1_P, right1_P]
        inliers = []
        reprojections = []
        if accuracy is None:
            accuracy = self.accuracy
        for points_2D, P in zip(all_2D_points, all_ps):
            reprojected_points = self.reproject(P, points_3D_world)
            reprojections.append(reprojected_points)
            reprojection_errors = np.linalg.norm(reprojected_points - points_2D, axis=-1)
            are_inliers = reprojection_errors < accuracy
            inliers.append(are_inliers)
        inliers = np.array(inliers).T
        inliers = np.all(inliers, axis=1)
        inliers_inds = np.nonzero(inliers)[0]
        return inliers_inds, reprojections

    @staticmethod
    def estimate_number_of_ransac_iterations(probability_for_success, outliers_ratio, number_of_model_params=4):
        num_iterations = np.ceil(np.log(1 - probability_for_success) / np.log(1 - np.power(1 - outliers_ratio, number_of_model_params)))
        num_iterations = min(num_iterations, MAX_RANSAC_ITERATIONS)
        return num_iterations

    def run_RANSAC(self, world_3D_points, left0_2D_points, right0_2D_points, left1_2D_points,
                   right1_2D_points, K, M_L0, M_R0, accuracy=None):
        probability_for_success = 0.999  # probability for successes
        outliers_ratio = 0.99  # outliers ratio, later updated
        num_iterations_needed = self.estimate_number_of_ransac_iterations(probability_for_success=probability_for_success,
                                                                          outliers_ratio=outliers_ratio,
                                                                          number_of_model_params=self.NUMBER_OF_POINTS_FOR_PNP)
        num_points = len(world_3D_points)
        min_outliers_num = -1
        max_inliers_num = -1
        iteration_counter = 0
        max_inliers_indices = None
        while outliers_ratio != 0 and iteration_counter < num_iterations_needed:
            indices = np.random.randint(0, num_points, self.NUMBER_OF_POINTS_FOR_PNP)
            left1_extrinsic_mat = self.get_R_t_using_pnp(world_3D_points[indices], left1_2D_points[indices], K,
                                                         flag=cv2.SOLVEPNP_AP3P)
            if left1_extrinsic_mat is self.PNP_FAILED:
                continue

            right1_to_left1_extrinsic_matrix = M_R0
            T_left_1 = self.get_T(left1_extrinsic_mat)
            T_right_1_to_left_1 = self.get_T(right1_to_left1_extrinsic_matrix)
            T_right_1_to_left_0 = T_right_1_to_left_1 @ T_left_1
            right1_to_left0_extrinsic_mat = self.get_Rt_from_T(T_right_1_to_left_0)

            left1_P_est = K @ left1_extrinsic_mat
            right1_P_est = K @ right1_to_left0_extrinsic_mat
            left0_P = K @ M_L0
            right0_P = K @ M_R0
            inliers_inds, _ = self.check_reprojections(left0_2D_points, right0_2D_points, left1_2D_points, right1_2D_points,
                                                       left0_P, right0_P, left1_P_est, right1_P_est, world_3D_points,
                                                       accuracy=accuracy)
            num_inliers = len(inliers_inds)
            if num_inliers == 0:
                continue
            num_outliers = num_points - num_inliers
            if num_inliers > max_inliers_num:
                max_inliers_num = num_inliers
                min_outliers_num = num_outliers
                max_inliers_indices = inliers_inds

            outliers_ratio = min_outliers_num / num_points
            num_iterations_needed = self.estimate_number_of_ransac_iterations(probability_for_success=probability_for_success,
                                                                              outliers_ratio=outliers_ratio,
                                                                              number_of_model_params=self.NUMBER_OF_POINTS_FOR_PNP)
            num_iterations_needed = max(num_iterations_needed, self.min_ransac_iterations)
            iteration_counter += 1

        final_left1_extrinsic_mat = self.get_R_t_using_pnp(world_3D_points[max_inliers_indices],
                                                           left1_2D_points[max_inliers_indices],
                                                           K, flag=cv2.SOLVEPNP_ITERATIVE)
        return final_left1_extrinsic_mat, max_inliers_indices, iteration_counter

    @staticmethod
    def apply_T_to_points_3D(T, points_3D):
        points_3D_hom = ImageProcessor.to_homogeneous(points_3D)
        points_3D_hom_after_T = (T @ points_3D_hom.T).T
        points_3D_new = ImageProcessor.from_homogeneous(points_3D_hom_after_T)
        return points_3D_new

    def display_2_stereo_pairs_using_p3p(self, K, M1, M2, left_0_points_2D, left_1_points_2D, points_3D_0):
        indices = np.random.randint(0, len(left_0_points_2D), 4)
        left_1_extrinsic_mat = self.get_R_t_using_pnp(points_3D_0[indices], left_1_points_2D[indices], K,
                                                      flag=cv2.SOLVEPNP_AP3P)
        right1_to_left1_extrinsic_matrix = M2
        right_1_extrinsic_mat = self.do_extrinsic_mat_transform(left_1_extrinsic_mat, right1_to_left1_extrinsic_matrix)
        left_0_extrinsic_mat = M1
        right_0_extrinsic_mat = M2
        self.display_cameras_given_extrinsic_matrices(left_0_extrinsic_mat, left_1_extrinsic_mat, right_0_extrinsic_mat,
                                                      right_1_extrinsic_mat)

    @staticmethod
    def display_cameras_given_extrinsic_matrices(left_0_extrinsic_mat, left_1_extrinsic_mat, right_0_extrinsic_mat, right_1_extrinsic_mat):
        left_0_center = utils.get_camera_center_from_Rt(left_0_extrinsic_mat)
        right_0_center = utils.get_camera_center_from_Rt(right_0_extrinsic_mat)
        left_1_center = utils.get_camera_center_from_Rt(left_1_extrinsic_mat)
        right_1_center = utils.get_camera_center_from_Rt(right_1_extrinsic_mat)
        R_left_0, _ = utils.extract_R_t(left_0_extrinsic_mat)
        R_right_0, _ = utils.extract_R_t(right_0_extrinsic_mat)
        R_left_1, _ = utils.extract_R_t(left_1_extrinsic_mat)
        R_right_1, _ = utils.extract_R_t(right_1_extrinsic_mat)
        Visualizer.plot_cameras(left_0_center, right_0_center, left_1_center, right_1_center,
                                R_left_0, R_right_0, R_left_1, R_right_1, file_name="camera_pairs_plot.html")

    def find_shared_points_across_2_stereo_pairs(self, left_0, left_1, right_0, right_1):
        left_descriptors_0, right_descriptors_0, left_key_points_0, matches_dict_0, right_key_points_0, matches_0 = self.get_matches(left_0, right_0)
        left_descriptors_1, right_descriptors_1, left_key_points_1, matches_dict_1, right_key_points_1, matches_1 = self.get_matches(left_1, right_1)
        left_0_left_1_matches = utils.find_closest_features(left_descriptors_0, left_descriptors_1)
        matches_dict_left0_left1 = self.create_matches_dict(left_0_left_1_matches)
        left_0_inds, right_0_inds, left_1_inds, right_1_inds = self.get_points_in_4_images(matches_dict_left0_left1,
                                                                                           matches_dict_0, matches_dict_1)
        left_0_shared_kpnts = [left_key_points_0[i] for i in left_0_inds]
        right_0_shared_kpnts = [right_key_points_0[i] for i in right_0_inds]
        left_1_shared_kpnts = [left_key_points_1[i] for i in left_1_inds]
        right_1_shared_kpnts = [right_key_points_1[i] for i in right_1_inds]
        left_0_points_2D, right_0_points_2D = utils.get_points_from_key_points(left_0_shared_kpnts, right_0_shared_kpnts)
        left_1_points_2D, right_1_points_2D = utils.get_points_from_key_points(left_1_shared_kpnts, right_1_shared_kpnts)
        return left_0_points_2D, left_1_points_2D, right_0_points_2D, right_1_points_2D

    def get_matches(self, left, right):
        left_key_points, right_key_points, left_descriptors, right_descriptors, matches = utils.match_2_images(left, right,
                                                                                                               feature_extractor_name=self.feature_extractor_name)
        good_matches_inds = utils.reject_outliers_using_y_dist(left_key_points, right_key_points, matches,
                                                               self.y_dist_threshold)
        good_matches = [match for match, is_good in zip(matches, good_matches_inds) if is_good]
        matches_dict = self.create_matches_dict(good_matches)
        return left_descriptors, right_descriptors, left_key_points, matches_dict, right_key_points, good_matches

    def Q3(self):
        left_0, right_0 = utils.read_images(0)
        left_1, right_1 = utils.read_images(1)
        left_0_points_2D, left_1_points_2D, right_0_points_2D, right_1_points_2D = self.find_shared_points_across_2_stereo_pairs(left_0, left_1, right_0, right_1)

        Visualizer.display_matches_4_cams(left_0_points_2D, right_0_points_2D,
                                          left_1_points_2D, right_1_points_2D,
                                          image_pair_0=0, image_pair_1=1,
                                          num_points_display=10)

        points_3D_0 = utils.triangulate_points(self.P1, self.P2, left_0_points_2D, right_0_points_2D)
        points_3D_1 = utils.triangulate_points(self.P1, self.P2, left_1_points_2D, right_1_points_2D)

        indices = np.random.randint(0, len(left_0_points_2D), 4)
        left_1_extrinsic_mat = self.get_R_t_using_pnp(points_3D_0[indices], left_1_points_2D[indices], self.K,
                                                      flag=cv2.SOLVEPNP_AP3P)
        right1_to_left1_extrinsic_matrix = self.M2
        right_1_extrinsic_mat = self.do_extrinsic_mat_transform(left_1_extrinsic_mat, right1_to_left1_extrinsic_matrix)
        left_0_extrinsic_mat = self.M1
        right_0_extrinsic_mat = self.M2

        self.display_cameras_given_extrinsic_matrices(left_0_extrinsic_mat, left_1_extrinsic_mat, right_0_extrinsic_mat, right_1_extrinsic_mat)

        left1_P = self.K @ left_1_extrinsic_mat
        right1_P = self.K @ right_1_extrinsic_mat
        left0_P = self.K @ left_0_extrinsic_mat
        right0_P = self.K @ right_0_extrinsic_mat

        inliers_inds, reprojections = self.check_reprojections(left_0_points_2D, right_0_points_2D, left_1_points_2D, right_1_points_2D,
                                                               left0_P, right0_P, left1_P, right1_P, points_3D_0)

        left_0_reprojected, _, left_1_reprojected, _ = reprojections

        Visualizer.plot_matches_and_supporters(left_0_reprojected[inliers_inds], left_1_reprojected[inliers_inds],
                                               left_0_points_2D[inliers_inds], left_1_points_2D[inliers_inds],
                                               left_0, left_1)

        final_left1_extrinsic_mat, max_inliers_indices, _ = self.run_RANSAC(world_3D_points=points_3D_0,
                                                                         left0_2D_points=left_0_points_2D,
                                                                         right0_2D_points=right_0_points_2D,
                                                                         left1_2D_points=left_1_points_2D,
                                                                         right1_2D_points=right_1_points_2D,
                                                                         K=self.K,
                                                                         M_L0=self.M1,
                                                                         M_R0=self.M2)

        T = self.get_T(final_left1_extrinsic_mat)
        points_3D_pair1 = points_3D_1[max_inliers_indices]
        points_3D_pair0 = points_3D_0[max_inliers_indices]
        points_3D_pair0_after_T = self.apply_T_to_points_3D(T, points_3D_pair0)
        dists = np.linalg.norm(points_3D_pair0_after_T - points_3D_pair1, axis=1)
        med = np.median(dists)
        mean = np.mean(dists)
        Visualizer.plot_point_clouds_and_cameras(points_3D_pair1, points_3D_pair0_after_T,
                                                 extrinsic_mat_0=self.M1, extrinsic_mat_1=self.M2)

        Visualizer.plot_left0_left1_inliers_and_outliers(left_0_points_2D, left_1_points_2D, max_inliers_indices, left_0, left_1)

    @staticmethod
    def load_all_ground_truth_camera_matrices(path):
        extrinsic_matrices = []

        with open(path, 'r') as file:
            for line in file:
                numbers = list(map(float, line.strip().split()))
                if len(numbers) == 12:
                    matrix = np.array(numbers).reshape(3, 4)
                    extrinsic_matrices.append(matrix)

        return extrinsic_matrices

    def get_Rt_for_frame(self, frame, K, M1, M2, P1, P2):
        # Initialize feature extractor inside the method
        return self.get_Rt_for_2_stereo_pairs(K, M1, M2, P1, P2, frame)

    def get_Rt_for_2_stereo_pairs(self, K, M1, M2, P1, P2, frame):
        left_0, right_0 = utils.read_images(frame)
        left_1, right_1 = utils.read_images(frame + 1)
        left_0_points_2D, left_1_points_2D, right_0_points_2D, right_1_points_2D = self.find_shared_points_across_2_stereo_pairs(
            left_0, left_1, right_0, right_1)
        points_3D_0 = utils.triangulate_points(P1, P2, left_0_points_2D, right_0_points_2D)
        final_left1_extrinsic_mat, inliers, num_iterations = self.run_RANSAC(world_3D_points=points_3D_0,
                                                             left0_2D_points=left_0_points_2D,
                                                             right0_2D_points=right_0_points_2D,
                                                             left1_2D_points=left_1_points_2D,
                                                             right1_2D_points=right_1_points_2D,
                                                             K=K,
                                                             M_L0=M1,
                                                             M_R0=M2)
        return final_left1_extrinsic_mat, inliers, len(left_0_points_2D), num_iterations

    @staticmethod
    def rotation_matrix_to_axis_angle(matrix):
        # Create a Rotation object from the rotation matrix
        r = R.from_matrix(matrix)

        # Extract the axis and angle from the Rotation object
        axis_angle = r.as_rotvec()

        # The angle is the norm (magnitude) of the rotation vector
        angle = np.linalg.norm(axis_angle)

        # The axis is the normalized rotation vector
        axis = axis_angle / angle

        return axis, angle

    @staticmethod
    def find_rotation_angles_from_relative_Rts(Rts):
        N = len(Rts)
        angles = []
        for i in range(N):
            Rt = Rts[i]
            R, t = utils.extract_R_t(Rt)
            axis, angle = ImageProcessor.rotation_matrix_to_axis_angle(R)
            angle_deg = np.degrees(angle)
            angles.append(angle)
        return np.array(angles)

    def run_all_movie(self):
        directory_path = os.path.join(utils.DATA_PATH, "image_0")
        num_frames = sum(os.path.isfile(os.path.join(directory_path, f)) for f in os.listdir(directory_path))
        Rts = [self.M1]
        print(f"process all {num_frames} frames")
        # num_frames = 20
        start_time = time.time()

        # Use partial to pass the necessary arguments to the worker function
        from functools import partial
        worker_func = partial(self.get_Rt_for_frame, K=self.K, M1=self.M1, M2=self.M2, P1=self.P1, P2=self.P2)

        with mp.Pool(mp.cpu_count()) as pool:
            results = list(tqdm(pool.imap(worker_func, range(num_frames - 1)), total=num_frames - 1))

        Rts.extend([res[0] for res in results])
        ransac_iterations = np.array([res[3] for res in results]).mean()
        angles = ImageProcessor.find_rotation_angles_from_relative_Rts(Rts)
        inliers_num = np.array([len(res[1]) for res in results])

        # plt.plot(angles/angles.max())
        # plt.plot(inliers_num / inliers_num.max())
        # plt.show()

        np.save('relative_Rts.npy', np.array(Rts))
        np.save('inliers_num_oer_frame.npy', inliers_num)
        np.save('angles_rel_Rts.npy', angles)

        end_time = time.time()
        total_time = end_time - start_time
        print(f"the average number of ransac iterations was {ransac_iterations}")
        print(f"done with all {num_frames} frames")
        print(f"Total time {total_time}")

        T_cur = self.get_T(Rts[0])
        all_Ts = [T_cur]
        all_absolute_Rts = [self.M1]
        estimated_camera_centers = [utils.get_camera_center_from_Rt(self.M1)]
        for i in tqdm(range(1, num_frames)):
            T_relative = self.get_T(Rts[i])
            T_cur = T_relative @ T_cur
            Rt_cur = self.get_Rt_from_T(T_cur)
            camera_center_cur = utils.get_camera_center_from_Rt(Rt_cur)
            all_Ts.append(T_cur)
            all_absolute_Rts.append(Rt_cur)
            estimated_camera_centers.append(camera_center_cur)

        true_Rt = self.load_all_ground_truth_camera_matrices(self.GROUND_TRUTH_PATH)[:num_frames]
        ground_truth_centers = [utils.get_camera_center_from_Rt(Rt) for Rt in true_Rt]

        estimated_camera_centers = np.array(estimated_camera_centers)
        ground_truth_centers = np.array(ground_truth_centers)

        Visualizer.plot_trajectories_plt(estimated_camera_centers, ground_truth_centers)
        Visualizer.plot_all_cameras(estimated_camera_centers, no_y_axis=True)
        Visualizer.plot_all_ground_truth_vs_estimated_cameras(estimated_camera_centers, ground_truth_centers,
                                                              no_y_axis=True)
        errors_dists = np.linalg.norm(ground_truth_centers - estimated_camera_centers, axis=-1)
        errors_xyz = np.abs(ground_truth_centers - estimated_camera_centers)
        Visualizer.display_2D(array=errors_xyz,
                              legend=['X', 'Y', 'Z'],
                              save=True,
                              save_name='errors_xyz.png',
                              show=False, title='Ground truth vs estimated camera center Errors\n '
                                                'in X, Y, Z Coordinates',
                              xlabel='Frame Number',
                              ylabel='Error (m)')
        Visualizer.display_2D(array=errors_dists,
                              legend=['L2 distance'],
                              save=True,
                              save_name='errors_l2.png',
                              show=False, title='Ground truth vs estimated camera center Errors\n '
                                                'in absolute distance',
                              xlabel='Frame Number',
                              ylabel='Error (m)')


if __name__ == '__main__':
    feature_extractors = [
        AKAZE,
        # SIFT,
        # ORB,
        # BRISK
    ]
    for feature_extractor_name in feature_extractors:
        processor = ImageProcessor(feature_extractor_name=feature_extractor_name, y_dist_threshold=1, accuracy=1,
                                   min_ransac_iterations=5)
        # processor.Q3()
        processor.run_all_movie()
