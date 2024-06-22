from ex3 import Ex3
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    path_angles = r"C:\Users\amita\OneDrive\Desktop\temp\Slam\angles_rel_Rts.npy"
    path_inliers = r"C:\Users\amita\OneDrive\Desktop\temp\Slam\inliers_num_oer_frame.npy"
    inliers_num = np.load(path_inliers)
    angles = np.load(path_angles)[1:]


    inliers_num = inliers_num / inliers_num.max()
    # angles = angles / angles.max()
    # plt.plot(inliers_num)
    plt.plot(angles)
    plt.show()
    # plt.savefig('angles_vs_inliers.png')