import numpy as np
from ex3.Ex3 import ImageProcessor
from ex4.Ex4 import connectivity_graph, track_length_histogram
from utils.PoseGraph import PoseGraph
from utils.BundleAdjusment import BundleAdjusment
from utils.tracking_database import TrackingDB
from matplotlib import pyplot as plt
import pickle
import cv2

POSE_GRAPH_PATH = 'ex7/pose_graph'
TRACKING_DB_PATH = 'ex4/tracking_db_1.5_acc_y_1.5_1e-4_blur_1_it_50_akaze'


class PoseGraphProcessor:
    def __init__(self, tracking_db_path):
        self.processor = ImageProcessor()
        self.tracking_db = TrackingDB(self.processor.K, self.processor.M1, self.processor.M2)
        self.tracking_db.load(tracking_db_path)
        self.bundle_object = self.create_bundle_object()
        self.pose_graph = self.load_optimized_pose_graph()
        self.all_frames = np.arange(len(self.tracking_db))
        self.key_frames = self.pose_graph.key_frames

        (self.initial_mean_reprojection_errors_per_key_frame,
         self.initial_median_reprojection_errors_per_key_frame) = self.get_initial_reprojection_errors_per_key_frame()

        (self.final_mean_reprojection_errors_per_key_frame,
         self.final_median_reprojection_errors_per_key_frame) = self.get_final_reprojection_errors_per_key_frame()


        self.initial_mean_factor_errors_per_key_frame = self.get_initial_mean_factor_errors_per_key_frame()
        self.final_mean_factor_errors_per_key_frame = self.get_final_mean_factor_errors_per_key_frame()



        self.all_camera_poses_pnp = self.tracking_db.get_all_camera_poses()
        self.all_camera_poses_after_bundle = self.bundle_object.all_cameras
        self.localizations_pnp = np.array(
            [self.tracking_db.frameId_to_camera_center[frame] for frame in range(len(self.tracking_db))])
        self.localizations_bundle = self.pose_graph.bundle_object.all_camera_centers
        self.camera_poses_loop_closure, self.all_camera_poses_gtsam_loop_closure, self.localizations_loop_closure = self.pose_graph.get_all_cameras()
        self.localizations_ground_truth = self.pose_graph.ground_truth_locations
        self.ground_truth_locations, self.ground_truth_poses = self.pose_graph.get_ground_truth_locations()
        # self.plot_factor_erros()

    def plot_factor_erros(self):
        plt.figure(dpi=600)
        plt.plot(self.initial_mean_factor_errors_per_key_frame, label='Initial mean factor Error')
        plt.plot(self.final_mean_factor_errors_per_key_frame, label='Final mean factor Error')
        plt.xlabel("Key Frame Number")
        plt.ylabel("Mean factor error")
        plt.legend()
        plt.title("Mean factor errors per key frame"
                  "\nbefore and after bundle adjustment")
        plt.show()

    def plot_relative_errors_sub_sections(self):
        """Plot relative errors over different subsections of camera poses for various estimation methods."""
        estimation_methods = {
            'pnp': self.all_camera_poses_pnp,
            'bundle': self.all_camera_poses_after_bundle,
            'loop closure': self.camera_poses_loop_closure
        }
        section_lengths = [100, 400, 800]

        for method_name, camera_poses in estimation_methods.items():
            plt.figure(dpi=600)
            for length in section_lengths:
                relative_angle_errors, relative_movement_errors = self.get_relative_error_sequences(
                    camera_poses=camera_poses,
                    sequence_length=length
                )
                plt.plot(relative_angle_errors, label=f'{length} frames sequence')

            plt.xlabel('frames')
            ylabel = 'angles [deg/m]' if True else 'angles [deg]'  # Assuming per_meter=True here
            plt.ylabel(ylabel)
            plt.legend()
            # plt.yscale('log', base=2)
            plt.title(f'{method_name} Relative Angle Errors')
            plt.savefig(f'{method_name}_relative_angle_errors.png')

            plt.figure(dpi=600)
            for length in section_lengths:
                relative_angle_errors, relative_movement_errors = self.get_relative_error_sequences(
                    camera_poses=camera_poses,
                    sequence_length=length
                )
                plt.plot(relative_movement_errors, label=f'{length} frames sequence')

            plt.xlabel('frames')
            ylabel = 'distance [m/m]' if True else 'distance [m]'  # Assuming per_meter=True here
            plt.ylabel(ylabel)
            plt.legend()
            plt.title(f'{method_name} Relative Movement Errors')
            plt.savefig(f'{method_name}_relative_movement_errors.png')

    def plot_relative_errors_for_all_key_frames(self):
        """Plot relative errors for consecutive key frames across different camera pose estimation methods."""
        estimation_methods = {
            'pnp': self.all_camera_poses_pnp,
            'bundle': self.all_camera_poses_after_bundle,
            'loop closure': self.camera_poses_loop_closure
        }
        for method_name, camera_poses in estimation_methods.items():
            relative_angle_errors, relative_movement_errors = self.get_relative_error_key_frames(camera_poses, diff=1)
            plot_label = f'{method_name} consecutive key frames'
            self.plot_relative_errors(relative_angle_errors, relative_movement_errors, type_of_est=plot_label)

    @staticmethod
    def plot_relative_errors(relative_angles_errors, relative_movement_errors, type_of_est, xlabel='key frames', per_meter=False):
        title = f'{type_of_est} relative angle errors'
        plt.figure(dpi=600)
        plt.plot(relative_angles_errors, label=f'{type_of_est} relative angle errors')
        plt.xlabel(xlabel)
        ylabel = 'angles [deg/m]' if per_meter else 'angles [deg]'
        plt.ylabel(ylabel)
        # plt.legend()
        plt.title(title)
        plt.savefig(f'{title}.png')

        title = f'{type_of_est} relative movement errors'
        plt.figure(dpi=600)
        plt.plot(relative_movement_errors, label=f'{type_of_est} relative movement errors')
        plt.xlabel(xlabel)
        ylabel = 'distance [m/m]' if per_meter else 'distance [m]'
        plt.ylabel(ylabel)
        # plt.legend()
        plt.title(title)
        plt.savefig(f'{title}.png')

    def get_relative_error_key_frames(self, camera_poses, diff=1):
        relative_movement_errors = []
        relative_angles_errors = []
        for i in range(len(self.key_frames) - diff):
            cur_frame = self.key_frames[i]
            next_frame = self.key_frames[i + diff]

            relative_angle_error, relative_movement_error = self.get_relative_angle_and_movement_error(camera_poses,
                                                                                                       cur_frame,
                                                                                                       next_frame)

            relative_movement_errors.append(relative_movement_error)
            relative_angles_errors.append(relative_angle_error)
        return relative_angles_errors, relative_movement_errors

    def get_relative_error_sequences(self, camera_poses, sequence_length=100):
        relative_movement_errors_per_meter = []
        relative_angles_errors_per_meter = []
        for i in range(len(self.all_frames) - sequence_length):
            cur_frame = self.all_frames[i]
            next_frame = self.all_frames[i + sequence_length]

            relative_angle_error, relative_movement_error = self.get_relative_angle_and_movement_error(camera_poses,
                                                                                                       cur_frame,
                                                                                                       next_frame)
            distance_traveled = self.get_total_ground_truth_distance(cur_frame, next_frame)
            relative_angle_error_per_meter = relative_angle_error / distance_traveled
            relative_movement_error_per_meter = relative_movement_error / distance_traveled

            relative_angles_errors_per_meter.append(relative_angle_error_per_meter)
            relative_movement_errors_per_meter.append(relative_movement_error_per_meter)
        return relative_angles_errors_per_meter, relative_movement_errors_per_meter

    def get_total_ground_truth_distance(self, cur_frame, next_frame):
        relevant_locations = self.ground_truth_locations[cur_frame:next_frame]
        relative_distances = []
        for i in range(len(relevant_locations) - 1):
            cur_location = relevant_locations[i]
            next_location = relevant_locations[i + 1]
            distance = np.linalg.norm(cur_location - next_location)
            relative_distances.append(distance)
        accumulative_distance = np.sum(relative_distances)
        return accumulative_distance

    def get_relative_angle_and_movement_error(self, camera_poses, cur_frame, next_frame):
        cur_ground_truth_pose = self.ground_truth_poses[cur_frame]
        next_ground_truth_pose = self.ground_truth_poses[next_frame]
        cur_estimated_pose = camera_poses[cur_frame]
        next_estimated_pose = camera_poses[next_frame]
        # get relative movement
        relative_movement_ground_truth = self.get_relative_movement_2_poses(cur_ground_truth_pose,
                                                                            next_ground_truth_pose)
        relative_movement_estimation = self.get_relative_movement_2_poses(cur_estimated_pose,
                                                                          next_estimated_pose)
        relative_movement_error = np.linalg.norm(relative_movement_ground_truth - relative_movement_estimation)
        # get relative angle
        relative_angle_ground_truth = self.get_relative_angle_2_poses(cur_ground_truth_pose,
                                                                      next_ground_truth_pose)
        relative_angle_estimation = self.get_relative_angle_2_poses(cur_estimated_pose,
                                                                    next_estimated_pose)
        relative_angle_error = np.abs(relative_angle_ground_truth - relative_angle_estimation)
        return relative_angle_error, relative_movement_error

    @staticmethod
    def get_relative_movement_2_poses(pose1, pose2):
        c1 = TrackingDB.get_camera_center_from_Rt(pose1)
        c2 = TrackingDB.get_camera_center_from_Rt(pose2)
        relative_movement = c1 - c2
        return relative_movement

    def get_relative_angle_2_poses(self, pose1, pose2):
        R1, t1 = self.tracking_db.extract_R_t(pose1)
        R2, t2 = self.tracking_db.extract_R_t(pose2)
        relative_R = R1.T @ R2
        rvec, _ = cv2.Rodrigues(relative_R)
        relative_angle_deg = np.rad2deg(np.linalg.norm(rvec))
        return relative_angle_deg

    def get_initial_reprojection_errors_per_key_frame(self):
        initial_mean_reprojection_errors_per_key_frame = []
        initial_median_reprojection_errors_per_key_frame = []
        for key_frame in self.key_frames[:-1]:
            key_frame_errors = self.bundle_object.key_frames_to_bundelons[key_frame].get_initial_reprojection_erros()
            mean_key_frames_errors = np.mean(key_frame_errors)
            median_keyframe_errors = np.median(key_frame_errors)
            initial_mean_reprojection_errors_per_key_frame.append(mean_key_frames_errors)
            initial_median_reprojection_errors_per_key_frame.append(median_keyframe_errors)

        return np.array(initial_mean_reprojection_errors_per_key_frame), np.array(initial_median_reprojection_errors_per_key_frame)

    def get_final_reprojection_errors_per_key_frame(self):
        final_mean_reprojection_errors_per_key_frame = []
        final_median_reprojection_errors_per_key_frame = []
        for key_frame in self.key_frames[:-1]:
            key_frame_errors = self.bundle_object.key_frames_to_bundelons[key_frame].get_final_reprojection_erros()
            mean_key_frames_errors = np.mean(key_frame_errors)
            median_keyframe_errors = np.median(key_frame_errors)
            final_mean_reprojection_errors_per_key_frame.append(mean_key_frames_errors)
            final_median_reprojection_errors_per_key_frame.append(median_keyframe_errors)
        return np.array(final_mean_reprojection_errors_per_key_frame), np.array(final_median_reprojection_errors_per_key_frame)

    def get_initial_mean_factor_errors_per_key_frame(self):
        initial_mean_factor_errors_per_key_frame = []
        for key_frame in self.key_frames[:-1]:
            key_frame_factor_error = self.bundle_object.key_frames_to_bundelons[key_frame].get_error_before_optimization()
            initial_mean_factor_errors_per_key_frame.append(key_frame_factor_error)
        return np.array(initial_mean_factor_errors_per_key_frame)

    def get_final_mean_factor_errors_per_key_frame(self):
        final_mean_factor_errors_per_key_frame = []
        for key_frame in self.key_frames[:-1]:
            key_frame_factor_error = self.bundle_object.key_frames_to_bundelons[
                key_frame].get_error_after_optimization()
            final_mean_factor_errors_per_key_frame.append(key_frame_factor_error)
        return np.array(final_mean_factor_errors_per_key_frame)

    def create_bundle_object(self):
        """Create and optimize the bundle object."""
        bundle_object = BundleAdjusment(self.tracking_db)
        bundle_object.create_and_solve_all_bundle_windows()
        return bundle_object

    def load_optimized_pose_graph(self):
        """Load and return the optimized pose graph."""
        pose_graph = PoseGraph(self.bundle_object, self.processor, do_loop_closure=False)
        pose_graph.load(POSE_GRAPH_PATH)
        return pose_graph

    def present_statistics(self):
        """Present various statistics related to the tracking database."""
        self.tracking_db.present_tracking_statistics()
        self.tracking_db.plot_num_matches_and_inlier_percentage()
        connectivity_graph(self.tracking_db)
        track_length_histogram(self.tracking_db)

    def plot_trajectories(self, pnp=True, bundle=True, loop_closure=True,
                          ground_truth=True, only_key_frames=False,
                          title='Localizations vs Ground Truth', mark_size=0.1):
        """Plot the trajectories from various localization methods."""
        plt.figure(dpi=600)
        marker = 'o'
        # Scatter plot the x and z coordinates for each localization array with smaller markers
        frames_to_take = self.key_frames if only_key_frames else self.all_frames
        if pnp:
            plt.scatter(self.localizations_pnp[frames_to_take, 0],
                        self.localizations_pnp[frames_to_take, 2],
                        label='PnP Localizations',
                    color='orange', s=mark_size, marker=marker)
        if bundle:
            plt.scatter(self.localizations_bundle[frames_to_take, 0],
                        self.localizations_bundle[frames_to_take, 2],
                        label='Bundle Adjustment',
                    color='green', s=mark_size, marker=marker)
        if loop_closure:
            plt.scatter(self.localizations_loop_closure[frames_to_take, 0],
                        self.localizations_loop_closure[frames_to_take, 2],
                        label='Loop Closure',
                    color='blue', s=mark_size, marker=marker)
        if ground_truth:
            plt.scatter(self.localizations_ground_truth[frames_to_take, 0],
                        self.localizations_ground_truth[frames_to_take, 2],
                        label='Ground Truth',
                    color='red', s=mark_size, marker=marker)

        # Labels and title
        plt.xlabel('X Coordinate')
        plt.ylabel('Z Coordinate')
        plt.title(title)

        # Legend to identify each localization type
        plt.legend()

        # Show the plot
        plt.savefig('localizations.png', dpi=600)
        plt.show()

    def absolute_pnp_estimation_error(self):
        title = "Absolute PNP Estimation Error"
        pose_graph_processor.present_localization_error(self.localizations_ground_truth,
                                                        self.localizations_pnp,
                                                        title=title)

    def absolute_pose_graph_estimation_error_before_loop_closure(self):
        title = "Absolute pose graph Estimation Error after bundle adjustment"
        pose_graph_processor.present_localization_error(self.localizations_ground_truth,
                                                        self.localizations_bundle,
                                                        title=title)

    def absolute_pose_graph_estimation_error_after_loop_closure(self):
        title = "Absolute pose graph Estimation Error after loop closure"
        pose_graph_processor.present_localization_error(self.localizations_ground_truth,
                                                        self.localizations_loop_closure,
                                                        title=title)

    @staticmethod
    def present_localization_error(ground_truth, localizations_to_compare, title="Localization Error"):
        distance_per_axis = np.abs(localizations_to_compare - ground_truth)
        distance_l2 = np.linalg.norm(distance_per_axis, axis=1)
        plt.figure(dpi=600)
        plt.plot(distance_per_axis[:, 0], label='x')
        plt.plot(distance_per_axis[:, 1], label='y')
        plt.plot(distance_per_axis[:, 2], label='z')
        plt.plot(distance_l2, label='l2')
        plt.xlabel("Frames")
        plt.ylabel("Error [meters]")
        plt.legend()
        plt.title(title)
        plt.savefig(f'{title}.png', dpi=600)

    def get_relative_error_consecutive_key_frames(self):
        pass


if __name__ == "__main__":
    pose_graph_processor = PoseGraphProcessor(TRACKING_DB_PATH)
    pose_graph_processor.plot_relative_errors_sub_sections()
    pose_graph_processor.plot_relative_errors_for_all_key_frames()
    pose_graph_processor.absolute_pnp_estimation_error()
    pose_graph_processor.absolute_pose_graph_estimation_error_before_loop_closure()
    pose_graph_processor.absolute_pose_graph_estimation_error_after_loop_closure()
    pose_graph_processor.present_statistics()
    pose_graph_processor.plot_trajectories(pnp=False, bundle=True,
                                           loop_closure=False, ground_truth=False,
                                           only_key_frames=True, title='Division to Key frames', mark_size=0.5)
