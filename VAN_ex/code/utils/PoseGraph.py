import gtsam
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import networkx as nx
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.tracking_database import TrackingDB, Link
from utils import utils
from utils.BundleAdjusment import BundleAdjusment, Bundelon
from utils.visualize import Visualizer
import pickle
MAHALANOBIS_THERSH = 750
PNP_INLIERS_THRESHOLD = 50
VISUALIZATION_CONSTANT = 1
INITIAL_POSE_SIGMA = VISUALIZATION_CONSTANT * np.concatenate((np.deg2rad([1, 1, 1]), [0.1, 0.01, 1]))


class PoseGraph:
    def __init__(self, bundle_object, processor, do_loop_closure=False):
        self.loop_closure_pair_to_camera_centers = None
        self.loop_closure_pair_to_marginals = None
        self.loop_closure_pair_to_values = None
        self.loop_closure_keyframes_indices = None
        self.loop_closure_pair_to_pose_graph = None
        self.all_loop_closure_frames = None
        self.loop_closure_pair_to_optimized_values = dict()
        self._is_optimized = False
        self._optimized_values = None

        self.optimizer = None
        self.marginals = None
        self.loop_closure_graph = None
        self.processor = processor
        self.do_loop_closure = do_loop_closure
        if not bundle_object._is_optimized:
            assert "optimize the bundle first"
        self._camera_sym = set()
        self._initial_est = gtsam.Values()
        self.bundle_object = bundle_object
        self.tracking_db = self.bundle_object._tracking_db
        self.key_frames = self.bundle_object.key_frames
        self.key_frames_to_relative_motion_covariance = self.bundle_object.key_frames_to_relative_motion_covariance
        self.key_frames_to_relative_camera_poses_gtsam = self.bundle_object.key_frames_to_relative_camera_poses_gtsam
        self.key_frames_to_absolute_camera_poses_gtsam = self.bundle_object.key_frames_to_absolute_camera_poses_gtsam
        self.ground_truth_locations, self.ground_truth_poses = self.get_ground_truth_locations()
        self._pose_graph = self.create_pose_graph()
        self.initial_location_uncertainty_per_frame = self.get_location_uncertainties_for_entire_graph()
        self.optimize()
        if self.do_loop_closure:
            self.total_num_loop_closures = 0
            self.loop_closure_frames_counter = 0
            self.create_initial_loop_closure_graph()
            self.optimize_pose_graph_with_loop_closures()
            self.final_location_uncertainty_per_frame = self.get_location_uncertainties_for_entire_graph()

    def get_all_cameras(self):
        all_camera_poses_gtsam = []
        for key_frame in self.key_frames[:-1]:
            global_pose = self._optimized_values.atPose3(gtsam.symbol("c", key_frame))
            bundelon = self.bundle_object.key_frames_to_bundelons[key_frame]
            _, relative_poses_gtsam = bundelon.get_all_optimized_camera_poses()
            global_poses = [global_pose.compose(relative_pose) for relative_pose in relative_poses_gtsam]
            all_camera_poses_gtsam += global_poses[:-1]
        all_camera_poses_gtsam += [self._optimized_values.atPose3(gtsam.symbol("c", self.key_frames[-1]))]
        all_camera_centers = [pose.translation() for pose in all_camera_poses_gtsam]
        all_camera_poses = [Bundelon.get_Rt_from_gtsam_Pose3(pose) for pose in all_camera_poses_gtsam]
        return np.array(all_camera_poses), np.array(all_camera_poses_gtsam), np.array(all_camera_centers)

    def get_ground_truth_locations(self):
        true_Rt = self.processor.load_all_ground_truth_camera_matrices(self.processor.GROUND_TRUTH_PATH)
        ground_truth_centers = [utils.get_camera_center_from_Rt(Rt) for Rt in true_Rt]
        return np.array(ground_truth_centers), np.array(true_Rt)

    def get_location_uncertainties_for_entire_graph(self):
        all_uncertainty_sizes = []
        self.marginals = self.get_marginals()
        for key_frame in self.key_frames:

            # the covariance of the key frame relative to the starting frame
            keys = gtsam.KeyVector()
            keys.append(gtsam.symbol('c', 0))
            keys.append(gtsam.symbol('c', key_frame))
            cov = np.linalg.inv(self.marginals.jointMarginalInformation(keys).at(keys[-1], keys[-1]))

            location_cov = cov[3:, 3:]  # take the bottom left 3 by 3 matrix
            volume = np.sqrt(np.linalg.det(location_cov))
            uncertainty_size = volume
            all_uncertainty_sizes.append(uncertainty_size)
        return np.array(all_uncertainty_sizes)

    def save_cur_camera_locations_status_vs_ground_truth(self):
        cur_camera_centers = self.get_camera_centers_from_gtsam_graph()
        self.all_loop_closure_frames, self.loop_closure_keyframes_indices = self.get_loop_closure_frames()
        self.loop_closure_keyframes_indices = np.where(np.isin(self.key_frames, self.all_loop_closure_frames))[0]
        margin = 40

        # Calculate limits
        xmin = self.ground_truth_locations[:, 0].min() - margin
        xmax = self.ground_truth_locations[:, 0].max() + margin
        zmin = self.ground_truth_locations[:, 2].min() - margin
        zmax = self.ground_truth_locations[:, 2].max() + margin

        fig, ax = plt.subplots()

        cur_camera_centers_2D = cur_camera_centers[:, [0, 2]]
        ground_truth_2D = self.ground_truth_locations[:, [0, 2]]

        # Scatter plots
        ax.scatter(cur_camera_centers_2D[:, 0], cur_camera_centers_2D[:, 1], s=1, color='blue',
                   label='estimated key-frames centers')
        ax.scatter(ground_truth_2D[:, 0], ground_truth_2D[:, 1], s=1, color='red', label='ground truth')

        # Highlight loop closure frames
        loop_closure_2D = cur_camera_centers[self.loop_closure_keyframes_indices][:, [0, 2]]
        ax.scatter(loop_closure_2D[:, 0], loop_closure_2D[:, 1], s=10, color='green', label='loop closure frames')

        # Set axis limits
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([zmin, zmax])

        # Add grid
        ax.grid(True)

        # Save the plot in high resolution
        plt.legend()
        plt.savefig(f'all_loop_closure_stages/stage_{self.loop_closure_frames_counter}.png', dpi=600)
        plt.close()

    def get_loop_closure_frames(self):
        all_pairs_as_keys = [self.loop_closure_pair_to_optimized_values[i][1] for i in
                             range(len(self.loop_closure_pair_to_optimized_values))]
        all_loop_closure_frames = []
        for pairs in all_pairs_as_keys:
            for i in range(len(pairs)):
                all_loop_closure_frames.append(int(list(pairs)[i][0]))
                all_loop_closure_frames.append(int(list(pairs)[i][1]))
        all_loop_closure_frames = np.array(all_loop_closure_frames)
        loop_closure_keyframes_indices = np.where(np.isin(self.key_frames,all_loop_closure_frames))[0]
        return all_loop_closure_frames, loop_closure_keyframes_indices

    def get_camera_centers_from_gtsam_graph(self):
            camera_centers = []
            for key_frame in self.key_frames:
                optimized_pose = self._optimized_values.atPose3(gtsam.symbol('c', key_frame))
                camera_center = optimized_pose.translation()  # gtsam coordinates: t is is camera lovarin
                camera_centers.append(camera_center)
            camera_centers = np.array(camera_centers)
            return camera_centers

    def optimize_pose_graph_with_loop_closures(self,
                                               look_back_frames=60,
                                               candidates_per_frame=4):

        self.save_cur_camera_locations_status_vs_ground_truth()
        self.loop_closure_pair_to_optimized_values = {0: (self._optimized_values, dict(), self.get_marginals(), self.get_camera_centers_from_gtsam_graph())}
        self.loop_closure_pair_to_values = {0: self._optimized_values}
        self.loop_closure_pair_to_marginals = {0: self.get_marginals()}
        self.loop_closure_pair_to_camera_centers = {0: self.get_camera_centers_from_gtsam_graph()}
        self.loop_closure_pair_to_pose_graph = {0: self._pose_graph}
        for pose_ind in range(look_back_frames, len(self.key_frames)):
            cur_key_frame = int(self.key_frames[pose_ind])
            cur_pose_vector = self.get_key_frame_pose_vector(cur_key_frame)
            loop_closer_candidates = []
            for prev_pose_ind in range(pose_ind - look_back_frames):  # look back only more than look_back_frames frames
                prev_key_frame = self.key_frames[prev_pose_ind]
                shortest_path = nx.shortest_path(self.loop_closure_graph, source=prev_key_frame, target=cur_key_frame,
                                                 weight='weight')

                # Retrieve edge attributes for the shortest path
                cov_approximation = self.get_covariance_approximation(shortest_path)
                prev_pose_vector = self.get_key_frame_pose_vector(prev_key_frame)
                mahalanobis_distance = PoseGraph.get_mahalanobis(v1=prev_pose_vector,
                                                                 v2=cur_pose_vector,
                                                                 cov=cov_approximation)
                if mahalanobis_distance < MAHALANOBIS_THERSH:
                    # print(f"from {prev_key_frame} to {cur_key_frame} path length is {len(shortest_path)}" )
                    loop_closer_candidates.append((prev_key_frame, mahalanobis_distance))
            if len(loop_closer_candidates) > 0:
                loop_closer_candidates = np.array(loop_closer_candidates)
                loop_closer_candidates = loop_closer_candidates[loop_closer_candidates[:, 1].argsort()][:5]
                # print(f"for frame {cur_key_frame} the cancicates were: {loop_closer_candidates}")
                loop_closure_info = {}
                for closure_candidate, dist in loop_closer_candidates:
                    try:
                        pose_covariance, optimized_pose, _ = self.get_bundel_pose_and_covariance(
                                                                               closure_candidate, cur_key_frame)
                        # add a pose to the pose graph
                        loop_closure_info[(cur_key_frame, closure_candidate)] = [optimized_pose, pose_covariance]
                    except:
                        # print("pnp failed")
                        continue
                if len(loop_closure_info) > 0:
                    print(f"added {loop_closure_info.keys()}")
                    self.update_gtsam_pose_graph(loop_closure_info)
                    self.add_edges_to_loop_closure_graph(loop_closure_info)
                    self.update_weights_and_cov_loop_closure_graph()
                    self.loop_closure_frames_counter += 1
                    self.loop_closure_pair_to_optimized_values[self.loop_closure_frames_counter] = (self._optimized_values,
                                                                                             loop_closure_info.keys(),
                                                                                             self.get_marginals(),
                                                                                             self.get_camera_centers_from_gtsam_graph())
                    self.loop_closure_pair_to_values[self.loop_closure_frames_counter] = self._optimized_values
                    self.loop_closure_pair_to_marginals[self.loop_closure_frames_counter] = self.get_marginals()
                    self.loop_closure_pair_to_camera_centers[self.loop_closure_frames_counter] = self.get_camera_centers_from_gtsam_graph()
                    self.loop_closure_pair_to_pose_graph[self.loop_closure_frames_counter] = self._pose_graph
                    self.save_cur_camera_locations_status_vs_ground_truth()


    def add_edges_to_loop_closure_graph(self, loop_closer_info):
        # add edges
        for key in loop_closer_info:
            cur_key_frame, loop_closer_frame = key
            pose, covariance = loop_closer_info[key]
            self.loop_closure_graph.add_edge(cur_key_frame, loop_closer_frame, weight=1, covariance=covariance)

    def update_weights_and_cov_loop_closure_graph(self):
        for i in range(self._pose_graph.size()):
            factor = self._pose_graph.at(i)
            if isinstance(factor, gtsam.NoiseModelFactor):
                keys = factor.keys()
                # Since the graph only contains camera symbols, we assume all keys are camera symbols.
                if len(keys) == 2:  # Only consider factors with two keys (i.e., edges)
                    frame_vertex_1 = gtsam.symbolIndex(keys[0])
                    frame_vertex_2 = gtsam.symbolIndex(keys[1])
                    weight, cov = self.get_covariance_weight(frame_vertex_1, frame_vertex_2)
                    self.loop_closure_graph.add_edge(frame_vertex_1, frame_vertex_2, weight=weight, covariance=cov)
        for key_frame in self.key_frames:
            current_pose = self._optimized_values.atPose3(gtsam.symbol("c", key_frame))
            self.loop_closure_graph.add_node(key_frame, pose=current_pose)


    def update_gtsam_pose_graph(self, loop_closer_info):
        for key in loop_closer_info:
            cur_key_frame, loop_closer_frame = key
            optimized_pose, pose_covariance = loop_closer_info[key]
            prev_pose_sym = gtsam.symbol("c", int(loop_closer_frame))
            cur_pose_sym = gtsam.symbol("c", cur_key_frame)
            noise_model = gtsam.noiseModel.Gaussian.Covariance(pose_covariance)
            factor = gtsam.BetweenFactorPose3(cur_pose_sym, prev_pose_sym, optimized_pose,
                                              noise_model)  # here between cur to prev
            self._pose_graph.add(factor)
        self.optimize()

    def get_bundel_pose_and_covariance(self, closure_candidate, cur_key_frame):
        closure_candidate = int(closure_candidate)
        linkId_to_link, new_pose, num_inliers = self.get_initial_pose_estimate_and_links(closure_candidate, cur_key_frame)
        num_links = len(linkId_to_link.keys()) // 2
        frames = np.array([cur_key_frame, closure_candidate])
        track_Id_to_frames = {i: np.array([cur_key_frame, closure_candidate]) for i in range(num_links)}
        frameId_to_left_camera_gtsam = {cur_key_frame: gtsam.Pose3(), closure_candidate: new_pose}
        _, _, _, _, initial_est, graph_2_poses = Bundelon.create_bundelon_factor_graph(frames,
                                                                               track_Id_to_frames,
                                                                               frameId_to_left_camera_gtsam,
                                                                               self.tracking_db.K,
                                                                               self.tracking_db.M2,
                                                                               linkId_to_link)
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph_2_poses, initial_est)
        optimized_values = optimizer.optimize()
        marginals = gtsam.Marginals(graph_2_poses, optimized_values)
        keys = gtsam.KeyVector()
        keys.append(gtsam.symbol('c', cur_key_frame))
        keys.append(gtsam.symbol('c', closure_candidate))
        joint_covariance = np.linalg.inv(
            marginals.jointMarginalInformation(keys).at(keys[-1], keys[-1]))
        optimized_pose = optimized_values.atPose3(gtsam.symbol('c', closure_candidate))
        original_pose = gtsam.Pose3(new_pose)
        return joint_covariance, optimized_pose, original_pose

    def get_initial_pose_estimate_and_links(self, closure_candidate, cur_key_frame):
        left_0, right_0 = utils.read_images(cur_key_frame)
        left_1, right_1 = utils.read_images(closure_candidate)
        left_0_points_2D, left_1_points_2D, right_0_points_2D, right_1_points_2D = self.processor.find_shared_points_across_2_stereo_pairs(
            left_0, left_1, right_0, right_1)
        P1 = self.tracking_db.K @ self.tracking_db.M1
        P2 = self.tracking_db.K @ self.tracking_db.M2
        points_3D_0 = utils.triangulate_points(P1, P2, left_0_points_2D, right_0_points_2D)
        final_left1_extrinsic_mat, pnp_inliers, num_iterations = self.processor.run_RANSAC(world_3D_points=points_3D_0,
                                                                                           left0_2D_points=left_0_points_2D,
                                                                                           right0_2D_points=right_0_points_2D,
                                                                                           left1_2D_points=left_1_points_2D,
                                                                                           right1_2D_points=right_1_points_2D,
                                                                                           K=self.tracking_db.K,
                                                                                           M_L0=self.tracking_db.M1,
                                                                                           M_R0=self.tracking_db.M2)
        # print(len(pnp_inliers), flush=True)
        if len(pnp_inliers) < PNP_INLIERS_THRESHOLD:
            raise ValueError(F"the number of inliers must be above PNP_INLIERS_THRESHOLD = {PNP_INLIERS_THRESHOLD}")

        self.total_num_loop_closures += 1
        all_indices = np.arange(len(points_3D_0))
        pnp_outliers = np.setdiff1d(all_indices, pnp_inliers)
        left_0_outliers_2D = left_0_points_2D[pnp_outliers]
        left_1_outliers_2D = left_1_points_2D[pnp_outliers]

        left_0_points_2D, left_1_points_2D, right_0_points_2D, right_1_points_2D = (left_0_points_2D[pnp_inliers],
                                                                                    left_1_points_2D[pnp_inliers],
                                                                                    right_0_points_2D[pnp_inliers],
                                                                                    right_1_points_2D[pnp_inliers])
        Visualizer.display_key_points_with_inliers_outliers(left_0, left_1, left_0_points_2D, left_1_points_2D,
                                                            left_0_outliers_2D, left_1_outliers_2D,
                                                             gap=10, label_width=50, title="consensus matches",
                                                                  save_path=f"consensus matches/consensus matches for frames "
                                                                            f"{cur_key_frame} and {closure_candidate}.png")

        new_pose = BundleAdjusment.flip_Rt(final_left1_extrinsic_mat)
        links_0 = Link.create_links_from_points(left_0_points_2D, right_0_points_2D)
        link_0_dict = {(cur_key_frame, i): links_0[i] for i in range(len(links_0))}
        links_1 = Link.create_links_from_points(left_1_points_2D, right_1_points_2D)
        link_1_dict = {(closure_candidate, i): links_1[i] for i in range(len(links_1))}
        linkId_to_link = {**link_0_dict, **link_1_dict}
        return linkId_to_link, new_pose, len(pnp_inliers)

    def get_left_features(self, frame):
        left_features = self.tracking_db.get_features_left(frame)
        return left_features

    def do_pnp(self, frame_a, frame_b):
        K = self.bundle_object._tracking_db.K
        M1 = self.bundle_object._tracking_db.M1
        M2 = self.bundle_object._tracking_db.M2
        ransac_output = self.processor.get_Rt_2_frames(K, M1, M2, frame_a, frame_b)
        if type(ransac_output) == int:
            return -1, -1
        final_left1_extrinsic_mat, inliers, num_iterations = ransac_output
        return final_left1_extrinsic_mat, inliers


    def get_key_frame_pose_vector(self, cur_key_frame):
        cur_pose = self.loop_closure_graph.nodes[cur_key_frame].get("pose", None)
        cur_pose_vector = PoseGraph.gtsam_pose3_to_vector(cur_pose)
        return cur_pose_vector

    def get_covariance_approximation(self, shortest_path):
        all_intermediate_covariances = []
        for i in range(len(shortest_path) - 1):
            edge_data = self.loop_closure_graph.get_edge_data(shortest_path[i], shortest_path[i + 1])
            cov = edge_data['covariance']
            all_intermediate_covariances.append(cov)
        cov_approximation = np.sum(all_intermediate_covariances, axis=0)
        return cov_approximation

    @staticmethod
    def get_mahalanobis(v1, v2, cov):
        return np.sqrt((v1 - v2).T @ np.linalg.inv(cov) @ (v1 - v2))


    def get_key_frame_pose(self, key_frame):
        return self._optimized_values.atPose3(gtsam.symbol("c", key_frame))

    def create_initial_loop_closure_graph(self):
        self.loop_closure_graph = nx.Graph()
        # key_frame_pose = self.get_key_frame_pose(0)
        # self.loop_closure_graph.add_node(0, pose=key_frame_pose)
        # for i in range(1, len(self.key_frames)):
        #     cur_key_frame = self.key_frames[i]
        #     self.loop_closure_graph.add_node(cur_key_frame, pose=self.get_key_frame_pose(cur_key_frame))
        self.update_weights_and_cov_loop_closure_graph()


    def get_covariance_weight(self, key_frame_a, key_frame_b):
        cov = self.get_key_frames_poses_covariance(key_frame_a, key_frame_b)
        volume = np.sqrt(np.linalg.det(cov))
        return volume, cov

    def get_key_frames_poses_covariance(self, key_frame_a, key_frame_b):
        keys = gtsam.KeyVector()
        keys.append(gtsam.symbol('c', key_frame_a))
        keys.append(gtsam.symbol('c', key_frame_b))
        joint_covariance = np.linalg.inv(self.marginals.jointMarginalInformation(keys).at(keys[-1], keys[-1]))
        return joint_covariance

    def create_pose_graph(self):
        pose_graph = gtsam.NonlinearFactorGraph()
        # create pose for the first frame
        cur_pose_sym = gtsam.symbol("c", 0)
        self._camera_sym.add(cur_pose_sym)
        cur_cam_pose_gtsam = self.key_frames_to_absolute_camera_poses_gtsam[0]  # should be gtsam.Pose3()
        self._initial_est.insert(cur_pose_sym, cur_cam_pose_gtsam)
        pose_uncertainty = gtsam.noiseModel.Diagonal.Sigmas(sigmas=INITIAL_POSE_SIGMA)
        prior_factor = gtsam.PriorFactorPose3(cur_pose_sym, cur_cam_pose_gtsam, pose_uncertainty)
        pose_graph.add(prior_factor)

        prev_pose_sym = cur_pose_sym
        for key_frame in self.key_frames[1:]:
            cur_pose_sym = gtsam.symbol("c", key_frame)
            self._camera_sym.add(cur_pose_sym)

            # get the relative camera pose and covariance from this key frame to the previous one
            cur_relative_pose_gtsam = self.key_frames_to_relative_camera_poses_gtsam[key_frame]
            pose_covariance = self.key_frames_to_relative_motion_covariance[key_frame]

            # add the factor
            noise_model = gtsam.noiseModel.Gaussian.Covariance(pose_covariance)
            factor = gtsam.BetweenFactorPose3(prev_pose_sym, cur_pose_sym, cur_relative_pose_gtsam, noise_model)
            pose_graph.add(factor)

            # add the initial estimates: the absolute camera poses from the bundle
            cur_absolute_camera_pose = self.key_frames_to_absolute_camera_poses_gtsam[key_frame]
            self._initial_est.insert(cur_pose_sym, cur_absolute_camera_pose)

            # update the previous pose symbol
            prev_pose_sym = cur_pose_sym
        return pose_graph

    def optimize(self):
        if not self._is_optimized:
            self.optimizer = gtsam.LevenbergMarquardtOptimizer(self._pose_graph, self._initial_est)
        else:
            self.optimizer = gtsam.LevenbergMarquardtOptimizer(self._pose_graph, self._optimized_values)
        result = self.optimizer.optimize()
        self._optimized_values = result
        self._is_optimized = True
        self.marginals = self.get_marginals()

    def get_total_graph_error(self):
        if self._is_optimized:
            return self._pose_graph.error(self._optimized_values)
        elif not self._is_optimized:
            return self._pose_graph.error(self._initial_est)

    def get_marginals(self):
        if self._is_optimized:
            marginals = gtsam.Marginals(self._pose_graph, self._optimized_values)
        else:
            marginals = gtsam.Marginals(self._pose_graph, self._initial_est)
        return marginals

    def serialize(self, base_filename):
        data = {
            'loop_closure_keyframes_indices': self.loop_closure_keyframes_indices,
            'all_loop_closure_frames': self.all_loop_closure_frames,
            '_is_optimized': self._is_optimized,
            '_optimized_values': self._optimized_values,
            '_initial_est': self._initial_est,
            'key_frames': self.key_frames,
            'ground_truth_locations': self.ground_truth_locations,
            '_pose_graph': self._pose_graph,
            'initial_location_uncertainty_per_frame': self.initial_location_uncertainty_per_frame,
            'final_location_uncertainty_per_frame': getattr(self, 'final_location_uncertainty_per_frame', None),
            'loop_closure_graph': self.loop_closure_graph,
            'total_num_loop_closures': self.total_num_loop_closures,
            'loop_closure_frames_counter': self.loop_closure_frames_counter,
            'loop_closure_pair_to_values': self.loop_closure_pair_to_values,
            'loop_closure_pair_to_camera_centers': self.loop_closure_pair_to_camera_centers,
            'loop_closure_pair_to_pose_graph' : self.loop_closure_pair_to_pose_graph,
        }
        filename = base_filename + '.pkl'
        with open(filename, "wb") as file:
            pickle.dump(data, file)
        print('PoseGraph serialized to', filename)

    def load(self, base_filename):
        # Load the serialized data
        filename = base_filename + '.pkl'
        with open(filename, 'rb') as file:
            data = pickle.load(file)

            self.loop_closure_keyframes_indices = data['loop_closure_keyframes_indices']
            self.all_loop_closure_frames = data['all_loop_closure_frames']
            self._is_optimized = data['_is_optimized']
            self._optimized_values = data['_optimized_values']
            self._initial_est = data['_initial_est']
            self.key_frames = data['key_frames']
            self.ground_truth_locations = data['ground_truth_locations']
            self._pose_graph = data['_pose_graph']
            self.initial_location_uncertainty_per_frame = data['initial_location_uncertainty_per_frame']
            self.final_location_uncertainty_per_frame = data['final_location_uncertainty_per_frame']
            self.loop_closure_graph = data['loop_closure_graph']
            self.total_num_loop_closures = data['total_num_loop_closures']
            self.loop_closure_frames_counter = data['loop_closure_frames_counter']
            self.loop_closure_pair_to_values = data['loop_closure_pair_to_values']
            self.loop_closure_pair_to_camera_centers = data['loop_closure_pair_to_camera_centers']
            self.loop_closure_pair_to_pose_graph = data['loop_closure_pair_to_pose_graph']

        print('PoseGraph loaded from', filename)

    @staticmethod
    def gtsam_pose3_to_vector(pose):
        """
        Convert a gtsam.Pose3 object to a vector of size 6 containing the 3 Euler angles and the [x, y, z] translation using SciPy.

        Parameters:
        pose (gtsam.Pose3): The Pose3 object to be converted.

        Returns:
        np.ndarray: A vector of size 6 containing [roll, pitch, yaw, x, y, z].
        """
        # Extract rotation and translation from the Pose3 object
        rotation_matrix = pose.rotation().matrix()  # 3x3 rotation matrix
        translation = pose.translation()

        # Convert the rotation matrix to Euler angles (roll, pitch, yaw)
        r = R.from_matrix(rotation_matrix)
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)

        # Combine the Euler angles and translation into a single vector
        euler_and_translation_vector = np.array([roll, pitch, yaw, translation[0], translation[1], translation[2]])

        return euler_and_translation_vector
