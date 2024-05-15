import cv2
import numpy as np
import matplotlib.pyplot as plt

NUMBER_OF_FEATURES = 128
NUMBER_OF_FEATURES_TO_SHOW = 10
RATIO = 0.2
DATA_PATH = r'C:\Users\amita\OneDrive\Desktop\master\year 2\SLAM\SLAMProject\VAN_ex\dataset\sequences\00'

def read_images(idx):
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(DATA_PATH + '\image_0\\' + img_name, 0)
    img2 = cv2.imread(DATA_PATH + '\image_1\\' + img_name, 0)
    return img1, img2

def display_2_images(Left_img, right_img, gap=100, label_width=50):
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
    fig, ax = plt.subplots()
    ax.imshow(combined_image, cmap='gray')
    ax.axis('off')
    # Add text labels, adjusting the x-coordinate to move them further to the left
    plt.text(0, Left_img.shape[0] // 2, 'L', verticalalignment='center', color='black')
    plt.text(0, Left_img.shape[0] + gap + right_img.shape[0] // 2, 'R', verticalalignment='center', color='black')
    plt.show()


def detect_features(image1, image2, feature_extractor):
    kp1, des1 = feature_extractor.detectAndCompute(image1, None)
    kp2, des2 = feature_extractor.detectAndCompute(image2, None)
    return kp1, des1, kp2, des2


def display_2_images_with_matches(Left_img, right_img, keypoints1, keypoints2, good_matches, bad_matches=None, gap=10, label_width=50):
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

    # Draw good matches in green
    for match in good_matches[:NUMBER_OF_FEATURES_TO_SHOW]:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        (x1, y1) = keypoints1[img1_idx].pt
        (x2, y2) = keypoints2[img2_idx].pt
        color = (0, 255, 0)  # Green color for lines
        cv2.line(combined_image_color, (int(x1) + label_width, int(y1)),
                 (int(x2) + label_width, int(y2) + Left_0_with_label.shape[0] + gap), color, 1)

    # Draw bad matches in red
    if bad_matches:
        for match in bad_matches[:NUMBER_OF_FEATURES_TO_SHOW]:
            img1_idx = match.queryIdx
            img2_idx = match.trainIdx
            (x1, y1) = keypoints1[img1_idx].pt
            (x2, y2) = keypoints2[img2_idx].pt
            color = (0, 0, 255)  # Red color for lines
            cv2.line(combined_image_color, (int(x1) + label_width, int(y1)),
                     (int(x2) + label_width, int(y2) + Left_0_with_label.shape[0] + gap), color, 1)

    cv2.imshow('Matches', combined_image_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_closest_features(des1, des2, distance_func=cv2.NORM_L2):
    bf = cv2.BFMatcher(distance_func, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def reject_matches_using_significance_test(des1, des2, ratio=RATIO):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    bad_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
        else:
            bad_matches.append(m)
    return good_matches, bad_matches


def run_ex1():
    image1, image2 = read_images(0)
    alg = cv2.AKAZE_create()
    key_points_1, descriptors_1, key_points_2, descriptors_2 = detect_features(image1, image2, alg)
    good_matches, bad_matches = reject_matches_using_significance_test(descriptors_1, descriptors_2)
    display_2_images_with_matches(image1, image2, key_points_1, key_points_2, good_matches, bad_matches)


if __name__ == "__main__":
    run_ex1()
