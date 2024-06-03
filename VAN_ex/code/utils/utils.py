
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

NUMBER_OF_FEATURES = 128
NUMBER_OF_FEATURES_TO_SHOW = 2000
IMAGE_HEIGHT = 376
RATIO = 0.4
DATA_PATH = r'C:\Users\amita\OneDrive\Desktop\master\year 2\SLAM\SLAMProject\VAN_ex\dataset\sequences\00'


def read_images(idx):
    """
    Reads a pair of images from the dataset.

    Parameters:
    idx (int): Index of the image pair.

    Returns:
    tuple: A tuple containing the left and right images.
    """
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(DATA_PATH + '\\image_0\\' + img_name, 0)
    img2 = cv2.imread(DATA_PATH + '\\image_1\\' + img_name, 0)
    return img1, img2


def read_cameras():
     with open(DATA_PATH + '\calib.txt') as f:
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

def detect_features(image1, image2, feature_extractor=cv2.AKAZE_create()):
    """
    Detects and computes features in two images.

    Parameters:
    image1 (ndarray): The first image.
    image2 (ndarray): The second image.
    feature_extractor (cv2.Feature2D): Feature extractor.

    Returns:
    tuple: Keypoints and descriptors for both images.
    """
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
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def match_pair(image_pair_ind):
    image1, image2 = read_images(image_pair_ind)
    key_points_1, descriptors_1, key_points_2, descriptors_2 = detect_features(image1, image2)
    matches = find_closest_features(descriptors_1, descriptors_2)
    return key_points_1, key_points_2, matches


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


def triangulate_points(Pa, Pb, points_a, points_b):
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


def triangulate_using_opencv(Pa, Pb, points_a, points_b):
    x_3d_opencv = cv2.triangulatePoints(Pa, Pb, points_a.T, points_b.T).T
    x_3d_opencv = x_3d_opencv[:, :-1] / x_3d_opencv[:, -1:]
    return x_3d_opencv


def display(Left_img, right_img, keypoints1, keypoints2, good_matches=None,
            bad_matches=None, gap=10, label_width=50,
            number_of_features_to_show=NUMBER_OF_FEATURES_TO_SHOW,
            show_matches=False, line_thickness=1, show_lines=False,
            title="", save_name=""):

    left_label = np.ones((Left_img.shape[0], label_width), dtype=np.uint8) * 255
    right_label = np.ones((right_img.shape[0], label_width), dtype=np.uint8) * 255
    Left_0_with_label = np.hstack((left_label, Left_img))
    Right_0_with_label = np.hstack((right_label, right_img))
    max_width = max(Left_0_with_label.shape[1], Right_0_with_label.shape[1])
    gap_array = np.ones((gap, max_width), dtype=np.uint8) * 255

    if Left_0_with_label.shape[1] < max_width:
        padding = np.ones((Left_0_with_label.shape[0], max_width - Left_0_with_label.shape[1]), dtype=np.uint8) * 255
        Left_0_with_label = np.hstack((Left_0_with_label, padding))

    if Right_0_with_label.shape[1] < max_width:
        padding = np.ones((Right_0_with_label.shape[0], max_width - Right_0_with_label.shape[1]), dtype=np.uint8) * 255
        Right_0_with_label = np.hstack((Right_0_with_label, padding))

    combined_image = np.vstack((Left_0_with_label, gap_array, Right_0_with_label))
    combined_image_color = cv2.cvtColor(combined_image, cv2.COLOR_GRAY2BGR)

    if show_matches:
        for match in good_matches[:number_of_features_to_show]:
            img1_idx = match.queryIdx
            img2_idx = match.trainIdx
            (x1, y1) = keypoints1[img1_idx].pt
            (x2, y2) = keypoints2[img2_idx].pt
            color = (255, 165, 0)
            cv2.circle(combined_image_color, (int(x1) + label_width, int(y1)), 3, color, -1)
            cv2.circle(combined_image_color, (int(x2) + label_width, int(y2) + Left_0_with_label.shape[0] + gap), 3, color, -1)
            if show_lines:
                cv2.line(combined_image_color, (int(x1) + label_width, int(y1)),
                     (int(x2) + label_width, int(y2) + Left_0_with_label.shape[0] + gap), color, line_thickness)

        if bad_matches:
            for match in bad_matches[:number_of_features_to_show]:
                img1_idx = match.queryIdx
                img2_idx = match.trainIdx
                (x1, y1) = keypoints1[img1_idx].pt
                (x2, y2) = keypoints2[img2_idx].pt
                color = (0, 165, 255)
                cv2.circle(combined_image_color, (int(x1) + label_width, int(y1)), 3, color, -1)
                cv2.circle(combined_image_color, (int(x2) + label_width, int(y2) + Left_0_with_label.shape[0] + gap), 3,
                           color, -1)
                if show_lines:
                    cv2.line(combined_image_color, (int(x1) + label_width, int(y1)),
                         (int(x2) + label_width, int(y2) + Left_0_with_label.shape[0] + gap), color, line_thickness)
    else:
        for kp in keypoints1[:number_of_features_to_show]:
            x, y = kp.pt
            cv2.circle(combined_image_color, (int(x) + label_width, int(y)), 3, (0, 255, 0), -1)

        for kp in keypoints2[:number_of_features_to_show]:
            x, y = kp.pt
            cv2.circle(combined_image_color, (int(x) + label_width, int(y) + Left_0_with_label.shape[0] + gap), 3, (0, 255, 0), -1)

    plt.figure(figsize=(12, 8))
    plt.imshow(combined_image_color)
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
    plt.close()
