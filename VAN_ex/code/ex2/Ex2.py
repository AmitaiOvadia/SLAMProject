from utils import utils
from utils import visualize
import numpy as np
from matplotlib import pyplot as plt
import time
import plotly.graph_objects as go

Y_DISTANCE_THRESHOLD = 1
IMAGE_HEIGHT = 376


def get_points_from_matches(good_matches, key_points_a, key_points_b):
    points_a = []
    points_b = []
    for pair in range(len(good_matches)):
        x1, x2, y1, y2 = utils.get_pixels_from_matches(key_points_a, key_points_b, good_matches[pair])
        points_a.append((x1, y1))
        points_b.append((x2, y2))
    points_a = np.array(points_a)
    points_b = np.array(points_b)
    return points_a, points_b


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


def display_points_cloud(X, case):
    # Extract x, y, z coordinates
    x_coords = X[:, 0]
    y_coords = X[:, 1]
    z_coords = X[:, 2]

    # Create a 3D scatter plot
    fig = go.Figure(data=[
        go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=4,
                color=z_coords,  # Color points by z value
                colorscale='Viridis',
                opacity=0.8
            )
        ),
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode='markers',
            marker=dict(
                size=10,  # Bigger size for the origin point
                color='red',  # Different color for visibility
                opacity=1.0
            ),
            name='Origin (0,0,0)'
        )
    ])

    # Update layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', ),
            yaxis=dict(title='Y', ),
            zaxis=dict(title='Z', )
        ),
        title=f'3D Point Cloud Visualization with Origin {case}'
    )

    # Show the plot
    # fig.show()

    # Save the plot to an HTML file
    fig.write_html(f"3d_point_cloud_with_origin {case}.html")


def find_reprojection_error(Pa, Pb, X, points_a, points_b):
    N = len(points_a)
    points_a_rep = get_reprojection(Pa, X)
    points_b_rep = get_reprojection(Pb, X)
    reprojection_error_a = np.linalg.norm(points_a - points_a_rep, axis=-1)
    reprojection_error_b = np.linalg.norm(points_b - points_b_rep, axis=-1)
    reprojection_errors = (reprojection_error_a + reprojection_error_b) / 2
    return reprojection_errors


def get_reprojection(Pa, X):
    N = len(X)
    X_hom = np.column_stack((X, np.ones(N, )))
    points_rep = (Pa @ X_hom.T).T
    points_rep = points_rep[:, :-1] / points_rep[:, -1:]
    return points_rep


def check_times():
    k, m1, m2 = utils.read_cameras()
    P1, P2 = k @ m1, k @ m2
    points_a, points_b = get_matching_points_from_pair(0)

    num_iterations = 1000
    # Test 1: My implementation
    start_time = time.time()
    for _ in range(num_iterations):
        X_3d_per_pair = utils.triangulate_points_per_pair(P1, P2, points_a, points_b)
    end_time = time.time()
    time_naive = (end_time - start_time) / num_iterations

    # Test 2: No loops implementation
    start_time = time.time()
    for _ in range(num_iterations):
        x_3d_test = utils.triangulate_points(P1, P2, points_a, points_b)
    end_time = time.time()
    time_mine = (end_time - start_time) / num_iterations

    # Test 3: OpenCV implementation
    start_time = time.time()
    for _ in range(num_iterations):
        x_3d_opencv = utils.triangulate_using_opencv(P1, P2, points_a, points_b)
    end_time = time.time()
    time_opencv = (end_time - start_time) / num_iterations

    print(f"Time for (1) naive implementation: {time_naive :8f} seconds")
    print(f"Time for (2) no loops implementation: {time_mine :8f} seconds")
    print(f"Time for (3) OpenCV implementation: {time_opencv :8f} seconds")


def evaluate_random_y_distribution(key_points_1, key_points_2, matches):
    image1_inliers, image2_inliers, image1_outliers, image2_outliers = [], [], [], []
    for match in matches:
        _, y1, _, _ = utils.get_pixels_from_matches(key_points_1, key_points_2, match)
        y2 = np.random.uniform(0.0, IMAGE_HEIGHT)
        img1_idx, img2_idx = match.queryIdx, match.trainIdx
        if abs(y2 - y1) > 1:
            image1_outliers.append(key_points_1[img1_idx].pt)
            image2_outliers.append(key_points_2[img2_idx].pt)
        else:
            image1_inliers.append(key_points_1[img1_idx].pt)
            image2_inliers.append(key_points_2[img2_idx].pt)

    return np.array(image1_inliers), np.array(image2_inliers), np.array(image1_outliers), np.array(image2_outliers)


def verify_uniformity(kp1, kp2, match_pairs):
    total_inliers = 0
    total_outliers = 0
    num_iterations = 1000

    for _ in range(num_iterations):
        inliers_img1, _, outliers_img1, _ = evaluate_random_y_distribution(kp1, kp2, match_pairs)
        total_inliers += len(inliers_img1)
        total_outliers += len(outliers_img1)

    avg_inliers = total_inliers // num_iterations
    avg_outliers = total_outliers // num_iterations

    print(f"Average inliers and outliers with uniform y distribution over {num_iterations} iterations:")
    print(f"Inliers: {avg_inliers} Outliers: {avg_outliers}")


def get_matching_points_from_pair(pair_ind):
    key_points_1, key_points_2, matches = utils.match_pair(pair_ind)
    good_matches_inds = utils.reject_outliers_using_y_dist(key_points_1, key_points_2, matches, Y_DISTANCE_THRESHOLD)
    good_matches = [match for match, is_good in zip(matches, good_matches_inds) if is_good]
    points_a, points_b = get_points_from_matches(good_matches, key_points_1, key_points_2)
    return points_a, points_b


def Q21():
    """
    We are working with a pair of rectified stereo images.
    • Explain the special pattern of correct matches on such images. What is the cause of this
      pattern?
    • Create a histogram of the deviations from this pattern for all the matches.
    • Print the percentage of matches that deviate by more than 2 pixels.
    :return:
    """
    image_pair_ind = 0
    key_points_1, key_points_2, matches = utils.match_pair(image_pair_ind)
    all_y_dists = utils.get_y_distance_between_points(matches, key_points_1, key_points_2)
    num_of_matches = len(matches)
    y_dist_more_then_2 = len(all_y_dists[all_y_dists > 2])
    plt.hist(all_y_dists, bins=75)
    plt.title("Histogram of y distance between matched points pairs")
    plt.xlabel("Y Distance")
    plt.ylabel("number of points")
    plt.show()
    print(f"the percentage of matches that deviate by more than 2 is "
          f"{((y_dist_more_then_2 / num_of_matches) * 100):.2f}%")


def Q22():
    """
    2.2 Use the rectified stereo pattern to reject matches.
    • Present all the resulting matches as dots on the image pair. Accepted matches (inliers)
      in orange and rejected matches (outliers) in cyan.
    • How many matches were discarded?
    • Assuming erroneous matches are distributed uniformly across the image, what ratio of
      them would you expect to be rejected by this rejection policy? To how many erroneous
      matches that are wrongly accepted this would translate in the case of the current
      image?
    • Is this assumption (uniform distribution) realistic?
      Would you expect the actual number of accepted erroneous matches to be higher or
      lower? Why?``

    :return:
    """
    ind = 0
    left_img, right_img = utils.read_images(ind)
    key_points_1, key_points_2, matches = utils.match_pair(ind)
    good_matches_inds = utils.reject_outliers_using_y_dist(key_points_1, key_points_2, matches, Y_DISTANCE_THRESHOLD)
    good_matches = [match for match, is_good in zip(matches, good_matches_inds) if is_good]
    bad_matches = [match for match, is_good in zip(matches, good_matches_inds) if not is_good]
    print('percentage of discarded matches: {:.4g}'.format(100 * len(bad_matches) / len(matches)))
    visualize.Visualizer.display_key_points(left_img, right_img, key_points_1, key_points_2, good_matches, bad_matches,
                  show_matches=True, show_lines=False, save_name="y distance outliers rejection",
                  title='Accepted matches (inliers) '
                        'in orange and rejected matches (outliers) '
                        f'in cyan\nmatches criteria: less then {Y_DISTANCE_THRESHOLD} pixels y distance')

    verify_uniformity(key_points_1, key_points_2, matches)


def Q23():
    """
    Read the relative camera matrices of the stereo cameras from ‘calib.txt’.
    Use the matches and the camera matrices to define and solve a linear least squares
    triangulation problem. Do not use the opencv triangulation function.
    • Present a 3D plot of the calculated 3D points.
    • Repeat the triangulation using ‘cv2.triangulatePoints’ and compare the results.
        o Display the point cloud obtained from opencv and print the median distance
        between the corresponding 3d points.
    """
    k, m1, m2 = utils.read_cameras()
    P1, P2 = k @ m1, k @ m2
    pair_ind = 0
    points_a, points_b = get_matching_points_from_pair(pair_ind)

    x_3d_mine = utils.triangulate_points(P1, P2, points_a, points_b)
    display_points_cloud(x_3d_mine, case="my triangulation function")

    x_3d_opencv = utils.triangulate_using_opencv(P1, P2, points_a, points_b)
    display_points_cloud(x_3d_opencv, case="opencv triangulation function")

    dists = np.linalg.norm(x_3d_mine - x_3d_opencv, axis=-1)
    median = np.median(dists, axis=0)
    print(f"The median distance between the 3D points triangulated using\n"
          f"OpenCV function and my function is {median} meters")


def Q24():
    """
    2.4 Run this process (matching and triangulation) over a few pairs of images.
    • Look at the matches and 3D points, can you spot any 3D points that have an obviously
      erroneous location? (if not, look for errors at different image pairs)
    • What in your opinion is the reason for the error?
    • Can you think of a relevant criterion for outlier removal?
    """
    k, m1, m2 = utils.read_cameras()
    P1, P2 = k @ m1, k @ m2
    for pair_ind in range(5, 10):
        points_a, points_b = get_matching_points_from_pair(pair_ind)
        x_3d_mine = utils.triangulate_points(P1, P2, points_a, points_b)
        display_points_cloud(x_3d_mine, case=f"image pair {pair_ind}")


if __name__ == "__main__":
    Q21()
    Q22()
    Q23()
    Q24()
