import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

NUMBER_OF_FEATURES = 128
NUMBER_OF_FEATURES_TO_SHOW = 500
RATIO = 0.4
DATA_PATH = r'C:\Users\amita\OneDrive\Desktop\master\year 2\SLAM\SLAMProject\VAN_ex\dataset\sequences\00'


def read_images(idx):
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(DATA_PATH + '\image_0\\' + img_name, 0)
    img2 = cv2.imread(DATA_PATH + '\image_1\\' + img_name, 0)
    return img1, img2


def detect_features(image1, image2, feature_extractor):
    kp1, des1 = feature_extractor.detectAndCompute(image1, None)
    kp2, des2 = feature_extractor.detectAndCompute(image2, None)
    return kp1, des1, kp2, des2


def display(Left_img, right_img, keypoints1, keypoints2, good_matches=None,
                                  bad_matches=None, gap=10, label_width=50,
                                  number_of_features_to_show=NUMBER_OF_FEATURES_TO_SHOW,
                                  show_matches=False, line_thickness=1,
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

    # inds_to_show = np.random.randint(0, len(good_matches), size=number_of_features_to_show)
    # good_matches = [good_matches[ind] for ind in inds_to_show]

    if show_matches:
        # Draw good matches in green
        for match in good_matches[:number_of_features_to_show]:
            img1_idx = match.queryIdx
            img2_idx = match.trainIdx
            (x1, y1) = keypoints1[img1_idx].pt
            (x2, y2) = keypoints2[img2_idx].pt
            color = (0, 255, 0)  # Green color for lines
            cv2.line(combined_image_color, (int(x1) + label_width, int(y1)),
                     (int(x2) + label_width, int(y2) + Left_0_with_label.shape[0] + gap), color, line_thickness)

        # Draw bad matches in red
        if bad_matches:
            for match in bad_matches[:number_of_features_to_show]:
                img1_idx = match.queryIdx
                img2_idx = match.trainIdx
                (x1, y1) = keypoints1[img1_idx].pt
                (x2, y2) = keypoints2[img2_idx].pt
                color = (255, 0, 0)  # Red color for lines
                cv2.line(combined_image_color, (int(x1) + label_width, int(y1)),
                         (int(x2) + label_width, int(y2) + Left_0_with_label.shape[0] + gap), color, line_thickness)
    else:
        # Draw keypoints on the left image
        for kp in keypoints1[:number_of_features_to_show]:
            x, y = kp.pt
            cv2.circle(combined_image_color, (int(x) + label_width, int(y)), 3, (0, 255, 0), -1)  # Green circles

        # Draw keypoints on the right image
        for kp in keypoints2[:number_of_features_to_show]:
            x, y = kp.pt
            cv2.circle(combined_image_color, (int(x) + label_width, int(y) + Left_0_with_label.shape[0] + gap), 3, (0, 255, 0), -1)  # Green circles

    # Save the image using matplotlib
    plt.figure(figsize=(12, 8))
    plt.imshow(combined_image_color)
    if show_matches:
        plt.title(f"{number_of_features_to_show} features matched between left and right images")
    else:
        plt.title(title)
    plt.axis('off')
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
    plt.close()


def find_closest_features(des1, des2, distance_func=cv2.NORM_L2):
    bf = cv2.BFMatcher(distance_func, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def reject_matches_using_significance_test(des1, des2, ratio=RATIO):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    good_ratios = []
    bad_matches = []
    bad_ratios = []
    for first_match, second_match in matches:
        matches_ratio = first_match.distance / second_match.distance
        if matches_ratio < ratio:
            good_matches.append(first_match)
            good_ratios.append(matches_ratio)
        else:
            bad_matches.append(first_match)
            bad_ratios.append(matches_ratio)
    return good_matches, good_ratios, bad_matches, bad_ratios


def Q11():
    image1, image2 = read_images(0)
    alg = cv2.AKAZE_create()
    key_points_1, descriptors_1, key_points_2, descriptors_2 = detect_features(image1, image2, alg)
    display(image1, image2, key_points_1, key_points_2,
                                  title=f"Keypoints on left and right images", save_name="Keypoints")


def Q12():
    image1, image2 = read_images(0)
    alg = cv2.AKAZE_create()
    key_points_1, descriptors_1, key_points_2, descriptors_2 = detect_features(image1, image2, alg)
    print(f"first feature descriptor: \n{descriptors_1[0]}\n")
    print(f"second feature descriptor: \n{descriptors_1[2]}\n")


def Q13():
    image1, image2 = read_images(0)
    alg = cv2.AKAZE_create()
    key_points_1, descriptors_1, key_points_2, descriptors_2 = detect_features(image1, image2, alg)
    matches = find_closest_features(descriptors_1, descriptors_2)
    random_20 = random.sample(matches, 20)
    display(image1, image2, key_points_1, key_points_2, good_matches=random_20, number_of_features_to_show=20, show_matches=True,
            title=f"Matching key-points between left and right images", save_name="Matches", line_thickness=2)


def Q14():
    ratio = 0.4
    image1, image2 = read_images(0)
    alg = cv2.AKAZE_create()
    key_points_1, descriptors_1, key_points_2, descriptors_2 = detect_features(image1, image2, alg)
    good_matches, good_ratios, bad_matches, bad_ratios = reject_matches_using_significance_test(descriptors_1, descriptors_2, ratio=ratio)
    discarded_good_match = bad_matches[4]
    discarded_good_match_ratio = bad_ratios[4]
    display(image1, image2, key_points_1, key_points_2, good_matches=good_matches[:20], bad_matches=[discarded_good_match],
            number_of_features_to_show=20, show_matches=True,
            title=f"Matching key-points between left and right images with significance test",
            save_name="Matches_with_significance_test.png", line_thickness=2)
    print("ratio score of discarded good match: ", discarded_good_match_ratio)
    print(f"Ratio used for significance test: {ratio}")
    print(f"Ratio of discarded matches: {len(bad_matches)}/{len(good_matches) + len(bad_matches)}")


# Usage
def run_ex1():
    Q11()
    Q12()
    Q13()
    Q14()


if __name__ == "__main__":
    run_ex1()

