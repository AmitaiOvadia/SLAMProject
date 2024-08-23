import gtsam
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.tracking_database import TrackingDB

VISUALIZATION_CONSTANT = 1
INITIAL_POSE_SIGMA = VISUALIZATION_CONSTANT * np.concatenate((np.deg2rad([1, 1, 1]), [0.1, 0.01, 1]))


class Bundelon:
    def __init__(self, tracking_db, frames):
        self.initial_error = None
        self.highest_error = None
        self.highest_error_factor = None
        self.called_creat_factor_graph = False
        self._left_cameras_from_first_frame = None
        self._left_cameras_relative_Rts = None
        self._optimized_values = None
        self._optimizer = None
        self._tracking_db = tracking_db
        self._bundelon_absolute_frames = frames
        self._absolute_last_track_frameId = self._bundelon_absolute_frames[-1]
        self._absolue_first_track_frameId = self._bundelon_absolute_frames[0]
        self._frameId_to_left_camera_gtsam = self.get_left_cameras_from_first_frame_gtsam()
        self._camera_sym = set()
        self._landmark_sym = set()
        self._initial_est = gtsam.Values()
        self._gtsam_calibration = self.get_gtsam_calibration(tracking_db.K, tracking_db.M2)
        self._first_frame = self._bundelon_absolute_frames[0]
        self._bundelon_trackIds_to_frames = self.get_bundelon_tracks()
        self._num_of_tracks = len(self._bundelon_trackIds_to_frames)
        self._graph = gtsam.NonlinearFactorGraph()
        self._is_optimized = False

    def get_all_optimized_camera_poses(self):
        if not self.called_creat_factor_graph and not self._is_optimized:
            raise AssertionError("You must create graph and optimize it first")
        camera_poses_gtsam = [self._optimized_values.atPose3(gtsam.symbol("c", fr)) for fr in self._bundelon_absolute_frames]
        camera_poses = [Bundelon.get_Rt_from_gtsam_Pose3(pose) for pose in camera_poses_gtsam]
        return camera_poses, camera_poses_gtsam

    @staticmethod
    def get_Rt_from_gtsam_Pose3(cam_pose):
        """
        Gets a Pose3 gtsam object and returns an Rt matrix
        also flips from cam -> world (gtsam format) to world -> cam
        """
        R = cam_pose.rotation().matrix()
        t = cam_pose.translation()
        cam_pose = np.column_stack((R, t))
        cam_pose = BundleAdjusment.flip_Rt(cam_pose)  # flip to world -> cam
        return cam_pose

    def get_camera_pose_bundlon_end(self):
        if not self.called_creat_factor_graph and not self._is_optimized:
            raise AssertionError("You must create graph and optimize it first")
        cam_sym = gtsam.symbol("c", self._absolute_last_track_frameId)
        cam_pose_gtsam = self._optimized_values.atPose3(cam_sym)
        cam_pose = Bundelon.get_Rt_from_gtsam_Pose3(cam_pose_gtsam)
        return cam_pose, cam_pose_gtsam

    def get_optimized_landmarks_3d(self):
        if not self.called_creat_factor_graph and not self._is_optimized:
            raise AssertionError("You must create graph and optimize it first")
        landmarks_3d = []
        for landmark_sym in self._landmark_sym:
            landmark = self._optimized_values.atPoint3(landmark_sym)
            landmarks_3d.append(landmark)
        return landmarks_3d

    def get_total_graph_error(self):
        if self._is_optimized:
            return self._graph.error(self._optimized_values)
        else:
            return self._graph.error(self._initial_est)
    #
    # def get_average_graph_error(self):
    #     if self._is_optimized:
    #         return self._graph.error(self._optimized_values) / self._graph.size()
    #     else:
    #         return self._graph.error(self._initial_est) / self._graph.size()

    def get_initial_reprojection_erros(self):
        reprojection_errors = self.get_pose_graph_reprojection_erros(self._graph, self._initial_est)
        return reprojection_errors

    def get_final_reprojection_erros(self):
        if not self._is_optimized:
            assert "optimize graph first"
        reprojection_errors = self.get_pose_graph_reprojection_erros(self._graph, self._optimized_values)
        return reprojection_errors

    def get_pose_graph_reprojection_erros(self, factor_graph, estimations):
        reprojection_errors = []
        for i in range(factor_graph.size()):
            factor = factor_graph.at(i)  # Access the factor at index i

            # Check if the factor is a GenericStereoFactor3D
            if isinstance(factor, gtsam.GenericStereoFactor3D):
                pose_key = factor.keys()[0]
                landmark_key = factor.keys()[1]

                # Get the current estimate for the camera pose and 3D point
                estimated_pose = estimations.atPose3(pose_key)
                estimated_landmark = estimations.atPoint3(landmark_key)

                # Create a StereoCamera object
                stereo_camera = gtsam.StereoCamera(estimated_pose, self._gtsam_calibration)

                # Reproject the 3D point to get the 2D pixel coordinates
                reprojected_stereo_point = stereo_camera.project(estimated_landmark)

                # Get the measured 2D pixel coordinates from the factor
                measured_stereo_point = factor.measured()

                # Calculate the reprojection error in pixels
                error_xl = reprojected_stereo_point.uL() - measured_stereo_point.uL()
                error_xr = reprojected_stereo_point.uR() - measured_stereo_point.uR()
                error_y = reprojected_stereo_point.v() - measured_stereo_point.v()

                # reprojection error
                reprojection_error = np.sqrt(error_xl ** 2 + error_xr ** 2 + 2 * error_y ** 2)
                reprojection_errors.append(reprojection_error)

        reprojection_errors = np.array(reprojection_errors)
        return reprojection_errors

    def get_factor_graph(self):
        if not self.called_creat_factor_graph:
            raise AssertionError("You must create graph first")
        return self._graph

    def get_optimized_values_gtsam_format(self):
        if not self.called_creat_factor_graph and not self._is_optimized:
            raise AssertionError("You must create graph and optimize it first")
        return self._optimized_values

    def get_initial_estimates(self):
        if not self.called_creat_factor_graph:
            raise AssertionError("You must create graph first")
        return self._initial_est

    @staticmethod
    def create_bundelon_factor_graph(
        bundelon_absolute_frames, bundelon_trackIds_to_frames, frameId_to_left_camera_gtsam,
        K, M2, all_links):

        highest_error_factor = None
        highest_error = -1
        called_creat_factor_graph = True
        camera_sym = set()
        initial_est = gtsam.Values()
        graph = gtsam.NonlinearFactorGraph()
        gtsam_calibration = Bundelon.get_gtsam_calibration(K, M2)

        # add initial estimates for poses
        for relative_frame, absolute_frame in enumerate(bundelon_absolute_frames):
            left_pose_sym = gtsam.symbol("c", absolute_frame)  # the symbols are in absolute frames!
            camera_sym.add(left_pose_sym)
            cur_cam_pose = frameId_to_left_camera_gtsam[absolute_frame]
            gtsam_left_cam_pose = gtsam.Pose3(cur_cam_pose)  # supposed to be the origin
            initial_est.insert(left_pose_sym, gtsam_left_cam_pose)
            if relative_frame == 0:  # if we are in the first frame
                sigmas = INITIAL_POSE_SIGMA
                pose_uncertainty = gtsam.noiseModel.Diagonal.Sigmas(sigmas=sigmas)
                factor = gtsam.PriorFactorPose3(left_pose_sym, gtsam_left_cam_pose, pose_uncertainty)
                graph.add(factor)

        # create factor graphs add track
        for trackId, track_frames in bundelon_trackIds_to_frames.items():

            last_track_frame_for_triangulation = track_frames[-1]

            left_camera_last_frame_gtsam = frameId_to_left_camera_gtsam[last_track_frame_for_triangulation]
            gtsam_left_camera_pose = gtsam.Pose3(left_camera_last_frame_gtsam)
            gtsam_frame_to_triangulate_from = gtsam.StereoCamera(gtsam_left_camera_pose, gtsam_calibration)

            track_links = [all_links[(frameId, trackId)] for frameId in track_frames]
            last_link = track_links[-1]
            xl_last, xr_last, y_last = last_link.x_left, last_link.x_right, last_link.y
            gtsam_stereo_point2_for_triangulation = gtsam.StereoPoint2(xl_last, xr_last, y_last)

            gtsam_p3d = gtsam_frame_to_triangulate_from.backproject(gtsam_stereo_point2_for_triangulation)

            if (gtsam_p3d[2] <= 0 or np.linalg.norm(gtsam_p3d) > 200):  # don't let the points be too far, for numerical issues
                continue

            p3d_sym = gtsam.symbol("q", trackId)
            camera_sym.add(p3d_sym)
            initial_est.insert(p3d_sym, gtsam_p3d)

            # add each frame factor for this track
            for track_frameId in track_frames:
                track_frame_link = all_links[(track_frameId, trackId)]
                gtsam_measurment_pt2 = gtsam.StereoPoint2(track_frame_link.x_left,
                                                          track_frame_link.x_right,
                                                          track_frame_link.y)

                projection_uncertainty = gtsam.noiseModel.Isotropic.Sigma(dim=3, sigma=1)

                factor = gtsam.GenericStereoFactor3D(measured=gtsam_measurment_pt2,
                                                     noiseModel=projection_uncertainty,
                                                     poseKey=gtsam.symbol("c", track_frameId),
                                                     landmarkKey=p3d_sym,
                                                     K=gtsam_calibration)

                factor_error = factor.error(initial_est)
                if factor_error > highest_error:
                    highest_error_factor = factor
                    highest_error = factor_error
                graph.add(factor)

        return highest_error_factor, highest_error, called_creat_factor_graph, camera_sym, initial_est, graph

    def create_factor_graph(self):
        (
            self.highest_error_factor,
            self.highest_error,
            self.called_creat_factor_graph,
            self._camera_sym,
            self._initial_est,
            self._graph
        ) = Bundelon.create_bundelon_factor_graph(
            self._bundelon_absolute_frames,
            self._bundelon_trackIds_to_frames,
            self._frameId_to_left_camera_gtsam,
            self._tracking_db.K,
            self._tracking_db.M2,
            self._tracking_db.linkId_to_link
        )
        self.initial_error = self._graph.error(self._initial_est) / self._graph.size()

    def get_initial_error(self):
        return self.initial_error

    def get_error_after_optimization(self):
        if not self._is_optimized:
            assert "optimize first"
        return self._graph.error(self._optimized_values) / self._graph.size()

    def get_error_before_optimization(self):
        return self._graph.error(self._initial_est) / self._graph.size()

    def get_left_cameras_from_first_frame_gtsam(self):
        self._left_cameras_relative_Rts = [self._tracking_db.frameId_to_relative_extrinsic_Rt[frame] for frame in
                                           self._bundelon_absolute_frames]
        self._left_cameras_from_first_frame = BundleAdjusment.composite_relative_Rts(self._left_cameras_relative_Rts)
        left_cameras_from_first_frame_gtsam = [BundleAdjusment.flip_Rt(Rt) for Rt in
                                               self._left_cameras_from_first_frame]
        frameId_to_left_camera_gtsam = {}
        for i, left_cam_gtsam in enumerate(left_cameras_from_first_frame_gtsam):
            frameId_to_left_camera_gtsam[self._bundelon_absolute_frames[i]] = left_cam_gtsam
        return frameId_to_left_camera_gtsam

    def optimize_bundelon(self):
        if not self.called_creat_factor_graph:
            raise AssertionError("You must create graph first")
        self._optimizer = gtsam.LevenbergMarquardtOptimizer(self._graph, self._initial_est)
        self._optimized_values = self._optimizer.optimize()
        self._is_optimized = True

    def get_relative_motion_covariance(self):
        """
        Get the relative covariance between the start and end of the motion
        """
        marginals = gtsam.Marginals(self._graph, self._optimized_values)
        key_0, key_k = self._bundelon_absolute_frames[0], self._bundelon_absolute_frames[-1]
        keys = gtsam.KeyVector()
        keys.append(gtsam.symbol('c', key_0))
        keys.append(gtsam.symbol('c', key_k))
        joint_covariance = np.linalg.inv(marginals.jointMarginalInformation(keys).at(keys[-1], keys[-1]))
        return joint_covariance

    def get_bundelon_tracks(self, min_length=2):
        all_bundelon_tracks = set()
        trackId_to_track_frames = {}
        for absolute_frame in self._bundelon_absolute_frames:
            frame_tracks = self._tracking_db.frameId_to_trackIds_list[absolute_frame]
            all_bundelon_tracks.update(frame_tracks)
        for trackId in all_bundelon_tracks:
            all_track_frames = np.array(self._tracking_db.frames(trackId))
            track_frames = all_track_frames[(all_track_frames >= self._absolue_first_track_frameId) &
                                            (all_track_frames <= self._absolute_last_track_frameId)]
            track_length = len(track_frames)
            if track_length >= min_length:
                trackId_to_track_frames[trackId] = track_frames
        return trackId_to_track_frames

    @staticmethod
    def get_gtsam_calibration(K, M2):
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        s = K[0, 1]
        baseline = -M2[0, -1]
        return gtsam.Cal3_S2Stereo(fx, fy, s, cx, cy, baseline)

class BundleAdjusment:
    def __init__(self, tracking_db, calculate_only_first_bundle=False):
        self.all_cameras = None
        self.all_camera_centers = None
        self._tracking_db = tracking_db
        self.total_num_frames = len(self._tracking_db.frameId_to_trackIds_list)
        self.key_frames = self.choose_key_frames_by_track_length(percentile=40)
        if calculate_only_first_bundle:
            self.key_frames = self.key_frames[:2]
        # self.key_frames = np.arange(0, self.total_num_frames)
        self.all_bundelons = []
        self.key_frames_to_relative_camera_poses = {0: BundleAdjusment.create_origin_Rt()}
        self.key_frames_to_relative_camera_poses_gtsam = {0: gtsam.Pose3()}
        self.key_frames_to_all_camera_poses_per_window = {}
        self.key_frames_to_landmarks_relative_coordinates = {}
        self._is_optimized = False
        self.key_frames_to_absolute_camera_poses = {}
        self.key_frames_to_absolute_camera_poses_gtsam = {}
        self.key_frames_to_landmarks_absolute_coordinates = {}
        self.key_frames_to_camera_centers = {}
        self.key_frames_to_bundelons = {}
        self.key_frames_to_relative_motion_covariance = {}

    @staticmethod
    def create_origin_Rt():
        return np.eye(4)[:3]

    def calculate_key_frames_to_absolute_camera_poses(self):
        if not self._is_optimized:
            assert "need to optimize bundle first"
        key_frames_absolute_Rts = BundleAdjusment.composite_relative_Rts([self.key_frames_to_relative_camera_poses[key_frame]
                                                                   for key_frame in self.key_frames])
        all_camera_poses = []
        for i, key_frame in enumerate(self.key_frames):
            self.key_frames_to_absolute_camera_poses[key_frame] = key_frames_absolute_Rts[i]

        # now do it to the gtsam cameras
        prev_key_frame = 0
        self.key_frames_to_absolute_camera_poses_gtsam[prev_key_frame] = gtsam.Pose3()
        for key_frame in self.key_frames[1:]:
            previous_pose = self.key_frames_to_absolute_camera_poses_gtsam[prev_key_frame]
            cur_pose = self.key_frames_to_relative_camera_poses_gtsam[key_frame]
            cur_pose_global = previous_pose.compose(cur_pose)
            self.key_frames_to_absolute_camera_poses_gtsam[key_frame] = cur_pose_global
            prev_key_frame = key_frame

    def calculate_key_frames_to_landmarks_absolute_coordinates(self):
        """
        Calculate the absolute coordinates of landmarks given the absolute camera poses for each keyframe
        and fill the self.key_frames_to_landmarks_absolute_coordinates dictionary.
        """
        for key_frame, camera_pose in self.key_frames_to_absolute_camera_poses_gtsam.items():
            # Retrieve landmarks in relative coordinates for the current keyframe
            landmarks_relative = self.key_frames_to_landmarks_relative_coordinates.get(key_frame, [])

            # Transform landmarks to global coordinates
            landmarks_global = [camera_pose.transformFrom(landmark) for landmark in landmarks_relative]

            # Store the global landmarks in the dictionary
            self.key_frames_to_landmarks_absolute_coordinates[key_frame] = landmarks_global

    def calculate_key_frames_to_camera_locations(self):
        if not self._is_optimized:
            assert "need to optimize bundle first"
        if len(self.key_frames_to_absolute_camera_poses) == 0:
            assert "run get_key_frames_to_absolute_camera_poses first"
        for key_frame in self.key_frames:
            absolute_camera_pose = self.key_frames_to_absolute_camera_poses[key_frame]
            camera_center = self._tracking_db.get_camera_center_from_Rt(absolute_camera_pose)
            self.key_frames_to_camera_centers[key_frame] = camera_center

    def calculate_all_absolute_camera_poses_per_window(self):
        if len(self.key_frames_to_absolute_camera_poses) == 0:
            assert "run calculate_key_frames_to_absolute_camera_poses first"
        self.all_cameras = []
        self.all_camera_centers = []
        for key_frame in self.key_frames[:-1]:  # the last key frame doesn't hold a window
            absolute_key_frame_T = self.get_ket_frame_T(key_frame)
            for local_camera_pose in self.key_frames_to_all_camera_poses_per_window[key_frame][:-1]:  # overlap
                new_Rt = self.get_global_Rt(absolute_key_frame_T, local_camera_pose)
                camera_center = self._tracking_db.get_camera_center_from_Rt(new_Rt)
                self.all_cameras.append(new_Rt)
                self.all_camera_centers.append(camera_center)
        # deal with last frame
        last_frame_Rt = self.key_frames_to_absolute_camera_poses[self.key_frames[-1]]
        last_frame_camera_center = self._tracking_db.get_camera_center_from_Rt(last_frame_Rt)
        self.all_cameras.append(last_frame_Rt)
        self.all_camera_centers.append(last_frame_camera_center)

        self.all_cameras = np.array(self.all_cameras)
        self.all_camera_centers = np.array(self.all_camera_centers)


    def get_global_Rt(self, absolute_key_frame_T, local_camera_pose):
        local_T = np.row_stack((local_camera_pose, [0, 0, 0, 1]))
        new_local_T = local_T @ absolute_key_frame_T
        new_Rt = new_local_T[:3]
        return new_Rt

    def get_ket_frame_T(self, key_frame):
        absolute_key_frame_camera_pose = self.key_frames_to_absolute_camera_poses[key_frame]
        absolute_key_frame_T = np.row_stack((absolute_key_frame_camera_pose, [0, 0, 0, 1]))
        return absolute_key_frame_T

    def get_all_camera_centers(self):
        if self.all_cameras is None:
            assert "run calculate_all_absolute_camera_poses_per_window"
        return self.all_camera_centers

    def create_and_solve_all_bundle_windows(self):
        for i in tqdm(range(len(self.key_frames) - 1)):
            window_start_frame = self.key_frames[i]
            window_end_frame = self.key_frames[i + 1]
            bundelon_frames = np.arange(window_start_frame, window_end_frame + 1)
            bundelon = Bundelon(tracking_db=self._tracking_db, frames=bundelon_frames)
            # bundelon.create_bundelon_factor_graph()
            bundelon.create_factor_graph()   # todo
            bundelon.optimize_bundelon()
            all_camera_poses, all_camera_poses_gtsam = bundelon.get_all_optimized_camera_poses()
            all_landmarks_relative_coordinates = bundelon.get_optimized_landmarks_3d()
            relative_motion_covariance = bundelon.get_relative_motion_covariance()
            self.key_frames_to_relative_motion_covariance[window_end_frame] = relative_motion_covariance
            self.key_frames_to_bundelons[window_start_frame] = bundelon
            self.key_frames_to_all_camera_poses_per_window[window_start_frame] = all_camera_poses
            self.key_frames_to_relative_camera_poses[window_end_frame] = all_camera_poses[-1]
            self.key_frames_to_relative_camera_poses_gtsam[window_end_frame] = all_camera_poses_gtsam[-1]
            self.key_frames_to_landmarks_relative_coordinates[window_start_frame] = all_landmarks_relative_coordinates
        self._is_optimized = True
        self.calculate_key_frames_to_absolute_camera_poses()
        self.calculate_key_frames_to_landmarks_absolute_coordinates()
        self.calculate_key_frames_to_camera_locations()
        self.calculate_all_absolute_camera_poses_per_window()

    def get_camera_centers_and_landmarks(self):
        if not self._is_optimized:
            assert "you need to optimize first"
        all_landmarks = [self.key_frames_to_landmarks_absolute_coordinates[key_frame]
                         for key_frame in self.key_frames[:-1]]
        all_landmarks = np.concatenate(all_landmarks)
        key_frames_camera_centers = np.array([self.key_frames_to_camera_centers[key_frame] for key_frame in self.key_frames])
        return all_landmarks, key_frames_camera_centers

    def get_key_frames_by_split(self, bundelon_size=12):
        all_frames = np.arange(self.total_num_frames)
        key_frames = all_frames[::bundelon_size]
        if key_frames[-1] != all_frames[-1]:    # add the last frame
            key_frames = np.append(key_frames, all_frames[-1])
        return key_frames

    def choose_key_frames_by_track_length(self, percentile=60):
        """
        Choose keyframes by the median track len's from the last frame
        """
        key_frames = [0]
        n = self.total_num_frames
        frames = np.arange(self.total_num_frames)
        while key_frames[-1] < n - 1:
            last_key_frame = key_frames[-1]
            frame = frames[last_key_frame]
            tracks = np.array(self._tracking_db.frameId_to_trackIds_list[frame])  # all the tracks related to this frame
            tracks = tracks[tracks != -1]  # remove the length 1 tracks

            tracks_lens = []   # get all the lengths of the tracks
            for track in tracks:
                tracks_lens.append(len(self._tracking_db.trackId_to_frames[track]))
            tracks_lens = np.array(tracks_lens)
            percentile_length = np.percentile(tracks_lens, percentile)

            new_key_frame = int(percentile_length + last_key_frame)
            key_frames.append(min(new_key_frame, n - 1))
        # import matplotlib.pyplot as plt
        # diff = np.diff(key_frames)
        # plt.plot(diff)
        # plt.show()
        # meandiff = np.mean(diff)
        # std = np.std(diff)
        return key_frames

    @staticmethod
    def flip_Rt(Rt):
        R, t = Rt[:3,:3], Rt[:, 3]
        new_R = R.T
        new_t = - R.T @ t
        new_Rt = np.column_stack((new_R, new_t))
        return new_Rt

    @staticmethod
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

    @staticmethod
    def composite_relative_Rts(left_cameras_relative_Rts):
        """
        gets camera Rts that are relative to each other and returns their composition: all relative to the first one
        assumes that the first frame is the origin
        """
        T_cur = np.eye(4)  # initialize with R = I and t = 0
        all_Ts = [T_cur]
        all_absolute_Rts = [T_cur[:-1]]  # first frame is the origin
        for i in range(1, len(left_cameras_relative_Rts)):
            T_relative = TrackingDB.get_T(left_cameras_relative_Rts[i])
            T_cur = T_relative @ T_cur
            Rt_cur = TrackingDB.get_Rt_from_T(T_cur)
            all_Ts.append(T_cur)
            all_absolute_Rts.append(Rt_cur)
        return all_absolute_Rts