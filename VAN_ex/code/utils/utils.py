
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os

NUMBER_OF_FEATURES = 128
IMAGE_HEIGHT = 376
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'VAN_ex', 'dataset', 'sequences', '00')
AKAZE_THRESH = 1e-4
DO_BLUR = True
AKAZE = 'AKAZE'
SIFT = 'SIFT'
ORB = 'ORB'
BRISK = 'BRISK'


def read_images(idx):
    """
    Reads a pair of images from the dataset.

    Parameters:
    idx (int): Index of the image pair.

    Returns:
    tuple: A tuple containing the left and right images.
    """
    img_name = '{:06d}.png'.format(idx)
    path1 = os.path.join(DATA_PATH,'image_0', img_name)
    path2 = os.path.join(DATA_PATH,'image_1', img_name)
    # print(f"{path1}\n{path2}")
    img1 = cv2.imread(path1, 0)
    img2 = cv2.imread(path2, 0)
    return img1, img2


def get_points_from_key_points(key_points_a, key_points_b):
    points_a = []
    points_b = []
    for i in range(len(key_points_a)):
        (x1, y1) = key_points_a[i].pt
        (x2, y2) = key_points_b[i].pt
        points_a.append((x1, y1))
        points_b.append((x2, y2))
    points_a = np.array(points_a)
    points_b = np.array(points_b)
    return points_a, points_b


def read_cameras():
     path = os.path.join(DATA_PATH, 'calib.txt')
     with open(path) as f:
         l1 = f.readline().split()[1:] # skip first token
         l2 = f.readline().split()[1:] # skip first token
         l1 = [float(i) for i in l1]
         m1 = np.array(l1).reshape(3, 4)
         l2 = [float(i) for i in l2]
         m2 = np.array(l2).reshape(3, 4)
         k = m1[:, :3]
         m1 = np.linalg.inv(k) @ m1
         m2 = np.linalg.inv(k) @ m2
     return k, m1, m2


def detect_features(image1, image2, feature_extractor=cv2.AKAZE_create(), blur=False):
    """
    Detects and computes features in two images.

    Parameters:
    image1 (ndarray): The first image.
    image2 (ndarray): The second image.
    feature_extractor (cv2.Feature2D): Feature extractor.

    Returns:
    tuple: Keypoints and descriptors for both images.
    """
    if blur:
        sigma = 1.0
        kernel_size = (int(6 * sigma + 1) | 1, int(6 * sigma + 1) | 1)  # Ensure the kernel size is odd
        image1 = cv2.GaussianBlur(image1, kernel_size, sigma)
        image2 = cv2.GaussianBlur(image2, kernel_size, sigma)
    kp1, des1 = feature_extractor.detectAndCompute(image1, None)
    kp2, des2 = feature_extractor.detectAndCompute(image2, None)
    return kp1, des1, kp2, des2


def find_closest_features(des1, des2, distance_func=cv2.NORM_L2):
    """
    Finds the closest features between two sets of descriptors.

    Parameters:
    des1 (ndarray): Descriptors of the first image.
    des2 (ndarray): Descriptors of the second image.
    distance_func (int): Distance function to use.

    Returns:
    list: Sorted matches between descriptors.
    """
    bf = cv2.BFMatcher(distance_func, crossCheck=True)
    matches = bf.match(des1, des2)
    # matches = sorted(matches, key=lambda x: x.distance)
    return matches


def get_y_distance_between_points(matches, key_points_1, key_points_2):
    all_y_dists = []
    for match in matches:
        x1, x2, y1, y2 = get_pixels_from_matches(key_points_1, key_points_2, match)
        y_dist = np.abs(y1 - y2)
        all_y_dists.append(y_dist)
    all_y_dists = np.array(all_y_dists)
    return all_y_dists


def reject_outliers_using_y_dist(key_points_1, key_points_2, matches, threshold):
    all_y_dists = get_y_distance_between_points(matches, key_points_1, key_points_2)
    good_matches_inds = np.where(all_y_dists < threshold, 1, 0)
    return good_matches_inds


def match_pair(image_pair_ind):
    image1, image2 = read_images(image_pair_ind)
    key_points_1, key_points_2, descriptors_1, descriptors_2, matches = match_2_images(image1, image2)
    return key_points_1, key_points_2, matches


def match_2_images(image1, image2, feature_extractor_name='AKAZE'):
    if feature_extractor_name == AKAZE:
        feature_extractor = cv2.AKAZE_create(threshold=AKAZE_THRESH)  # Lower threshold for more sensitivity
    elif feature_extractor_name == SIFT:
        feature_extractor = cv2.SIFT.create()
    elif feature_extractor_name == ORB:
        feature_extractor = cv2.ORB.create()
    elif feature_extractor_name == BRISK:
        feature_extractor = cv2.BRISK_create()
    key_points_1, descriptors_1, key_points_2, descriptors_2 = detect_features(image1, image2,
                                                                               feature_extractor=feature_extractor,
                                                                               blur=DO_BLUR)
    matches = find_closest_features(descriptors_1, descriptors_2)
    return key_points_1, key_points_2, descriptors_1, descriptors_2, matches


def get_pixels_from_matches(key_points_1, key_points_2, match):
    img1_idx = match.queryIdx
    img2_idx = match.trainIdx
    (x1, y1) = key_points_1[img1_idx].pt
    (x2, y2) = key_points_2[img2_idx].pt
    return x1, x2, y1, y2


def triangulate_pair(Pa, Pb, point_a, point_b):
    A = np.zeros((4, 4))
    p1a, p2a, p3a = Pa[0, :], Pa[1, :], Pa[2, :]
    p1b, p2b, p3b = Pb[0, :], Pb[1, :], Pb[2, :]
    A[0, :] = point_a[0] * p3a - p1a
    A[1, :] = point_a[1] * p3a - p2a
    A[2, :] = point_b[0] * p3b - p1b
    A[3, :] = point_b[1] * p3b - p2b
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X[:-1] / X[-1]
    return X


def triangulate_points_per_pair(Pa, Pb, points_a, points_b):
    N = len(points_a)
    Xs = []
    for i in range(N):
        Xs.append(triangulate_pair(Pa, Pb, points_a[i], points_b[i]))
    return Xs


def triangulate_points_without_opencv(Pa, Pb, points_a, points_b):
    N = points_a.shape[0]
    A = np.zeros((N, 4, 4))
    p1a, p2a, p3a = Pa[0, :], Pa[1, :], Pa[2, :]
    p1b, p2b, p3b = Pb[0, :], Pb[1, :], Pb[2, :]
    A[:, 0] = points_a[:, 0].reshape(-1, 1) * p3a - p1a
    A[:, 1] = points_a[:, 1].reshape(-1, 1) * p3a - p2a
    A[:, 2] = points_b[:, 0].reshape(-1, 1) * p3b - p1b
    A[:, 3] = points_b[:, 1].reshape(-1, 1) * p3b - p2b

    # Perform SVD on each A matrix
    _, _, Vt = np.linalg.svd(A)
    # take last row of Vt that corresponds to last column of V
    X = Vt[:, -1, :]
    X = X[:, :-1] / X[:, -1:]
    return X

def extract_R_t(extrinsic_matrix):
    R = extrinsic_matrix[:3, :3]
    t = extrinsic_matrix[:3, 3]
    return R, t


def get_camera_center_from_Rt(Rt):
    R, t = extract_R_t(Rt)
    center = -R.T @ t
    return center


def triangulate_points(Pa, Pb, points_a, points_b):
    x_3d_opencv = cv2.triangulatePoints(Pa, Pb, points_a.T, points_b.T).T
    x_3d_opencv = x_3d_opencv[:, :-1] / x_3d_opencv[:, -1:]
    return x_3d_opencv


