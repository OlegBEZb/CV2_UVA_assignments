
import numpy as np
import cv2 as cv
import random
import matplotlib.pyplot as plt

    # sift = cv.SIFT_create()
    # kp, des = sift.detectAndCompute(img, None)
    # if bool_plot:
    #     plot = cv.drawKeypoints(img, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #     plt.plot(plot)
    #     plt.axis(False)
    #     plt.show()

def keypoint_matcher(img1, img2, n):
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=1)

    # get random subset of matches list
    subset = random.sample(matches, n)

    # show_matches = cv.drawMatchesKnn(img1, kp1, img2, kp2, subset, None)

    # plt.imshow(show_matches)
    # plt.axis(False)
    # plt.show()
    points1 = []
    points2 = []
    for m in matches:
        # if m.distance < 0.8*n.distance:
        points1.append(kp1[m[0].queryIdx].pt)
        points2.append(kp2[m[0].trainIdx].pt)

    return subset, points1, points2, kp1, kp2

def get_fundamental_matrix(matches):
    A = []
    for match in matches:
        # original point in image1
        p1 = kp1[match[0].queryIdx].pt
        # matched point in image2
        p2 = kp2[match[0].trainIdx].pt

        A.append([p1[0]*p2[0], p1[0]*p2[1], p1[0],
                  p1[1]*p2[0], p1[1]*p2[1], p1[1],
                  p2[0], p2[1], 1])
    A = np.array(A)
    U, D, V_t = np.linalg.svd(A)

    # slide 17: https://inst.eecs.berkeley.edu/~ee290t/fa19/lectures/lecture9-4-computing-the-fundamental-matrix.pdf
    tmp = V_t.T[:,n-1]
    print(tmp.shape)
    F = tmp.reshape((3,3))
    FU, FD, FV_t = np.linalg.svd(F)
    FD_prime = FD.copy()
    FD_prime[np.argmin(FD)] = 0
    new_F = np.dot(np.dot(FU, np.diag(FD_prime)),FV_t)

    return new_F

def draw_epipolar_lines(image1, image2, matches, kp1, kp2, F):
    _,c,_ = image1.shape

    for match in matches:
        # original point in image1
        p1 = kp1[match[0].queryIdx].pt
        cv.circle(image1, (int(p1[0]),int(p1[1])), 10, (255,0,0),4)
        p1 = np.array([p1[0],p1[1],1])

        r = np.dot(F,p1)
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        cv.line(image2, (x0,y0),(x1,y1),(255,0,0),3)

        # matched point in image2
        p2 = kp2[match[0].trainIdx].pt
        cv.circle(image2, (int(p2[0]),int(p2[1])), 10, (0,255,0),4)
        p2 = np.array([p2[0],p2[1],1])

        r = np.dot(F.T,p2)
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        cv.line(image1, (x0,y0),(x1,y1),(0,255,0),3)

    # ax2.imshow(image2)
    # ax1.imshow(image1)
    show_matches = cv.drawMatchesKnn(image1, kp1, image2, kp2, matches, None)

    plt.imshow(show_matches)
    plt.axis(False)
    plt.show()

if __name__ == '__main__':
    image1 = cv.imread("../Data/House/House/frame00000001.png")
    image2 = cv.imread("../Data/House/House/frame00000049.png")

    n = 8
    matches, pts1, pts2, kp1, kp2 = keypoint_matcher(image1, image2, n)

    F = get_fundamental_matrix(matches)
    draw_epipolar_lines(image1, image2, matches, kp1, kp2, F)



