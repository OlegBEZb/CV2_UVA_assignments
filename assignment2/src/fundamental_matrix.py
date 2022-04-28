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

def keypoint_matcher(img1, img2, random_n=None, filter_neighbours=True, draw_matches=False):
    sift = cv.SIFT_create()
    kp1, descriptors1 = sift.detectAndCompute(img1, None)
    kp2, descriptors2 = sift.detectAndCompute(img2, None)

    # TODO: if works slow, replace with KD
    matcher = cv.BFMatcher()
    # Once it is created, two important methods are BFMatcher.match() and BFMatcher.knnMatch(). First one returns the
    # best match. Second method returns k best matches where k is specified by the user. It may be useful when we need
    # to do additional work on that.
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    if random_n is not None:
        # get random subset of matches list
        matches = random.sample(matches, random_n)

    if draw_matches:
        # cv.drawMatches() draws the matches. It stacks two images horizontally and draw lines from first image to
        # second image showing best matches. There is also cv.drawMatchesKnn which draws all the k best matches. If k=2,
        # it will draw two match-lines for each keypoint.
        plt.figure(figsize=(15, 15))
        show_matches = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None)
        plt.imshow(show_matches)

    if filter_neighbours:
        # in some cases, the second closest-match may be very near to the first. It may happen due to noise or some
        # other reasons. In that case, ratio of closest-distance to second-closest distance is taken. If it is greater
        # than 0.8, they are rejected. It eliminates around 90% of false matches while discards only 5% correct matches,
        # as per the paper.
        len_before = len(matches)
        # With lower thresholds it's even better
        matches = [m for m in matches if m[0].distance / m[1].distance < 0.8]
        print(f'Before filtering neighbours: {len_before}. After: {len(matches)}')

    if draw_matches:
        # cv.drawMatches() draws the matches. It stacks two images horizontally and draw lines from first image to
        # second image showing best matches. There is also cv.drawMatchesKnn which draws all the k best matches. If k=2,
        # it will draw two match-lines for each keypoint.
        plt.figure(figsize=(15, 15))
        show_matches = cv.drawMatchesKnn(img1, kp1, img2, kp2,
                                         [(m[0], ) for m in matches],  # matches,
                                         None)
        plt.imshow(show_matches)

    # matches = sorted(matches, key=lambda x: x[0].distance)
    # if draw_matches:
    #     # cv.drawMatches() draws the matches. It stacks two images horizontally and draw lines from first image to
    #     # second image showing best matches. There is also cv.drawMatchesKnn which draws all the k best matches. If k=2,
    #     # it will draw two match-lines for each keypoint.
    #     plt.figure(figsize=(15, 15))
    #     show_matches = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches[:10], None)
    #     plt.imshow(show_matches)

    # extracting coordinates of matches points
    points1 = []
    points2 = []
    for m in matches:
        # according to the doc, queryIdx refers to the first keypoints and trainIdx refers to second keypoints
        # here we just take the closest point from all neighbours
        points1.append(kp1[m[0].queryIdx].pt)
        points2.append(kp2[m[0].trainIdx].pt)

    return matches, points1, points2, kp1, kp2


def get_fundamental_matrix(matches):
    A = []
    for match in matches:
        # original point in image1
        p1 = kp1[match[0].queryIdx].pt
        # matched point in image2
        p2 = kp2[match[0].trainIdx].pt

        A.append([p1[0] * p2[0], p1[0] * p2[1], p1[0],
                  p1[1] * p2[0], p1[1] * p2[1], p1[1],
                  p2[0], p2[1], 1])
    A = np.array(A)
    U, D, V_t = np.linalg.svd(A)

    # slide 17: https://inst.eecs.berkeley.edu/~ee290t/fa19/lectures/lecture9-4-computing-the-fundamental-matrix.pdf
    tmp = V_t.T[:, n - 1]
    print(tmp.shape)
    F = tmp.reshape((3, 3))
    FU, FD, FV_t = np.linalg.svd(F)
    FD_prime = FD.copy()
    FD_prime[np.argmin(FD)] = 0
    new_F = np.dot(np.dot(FU, np.diag(FD_prime)), FV_t)

    return new_F


def draw_epipolar_lines(image1, image2, matches, kp1, kp2, F):
    _, c, _ = image1.shape

    for match in matches:
        # original point in image1
        p1 = kp1[match[0].queryIdx].pt
        cv.circle(image1, (int(p1[0]), int(p1[1])), 10, (255, 0, 0), 4)
        p1 = np.array([p1[0], p1[1], 1])

        r = np.dot(F, p1)
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        cv.line(image2, (x0, y0), (x1, y1), (255, 0, 0), 3)

        # matched point in image2
        p2 = kp2[match[0].trainIdx].pt
        cv.circle(image2, (int(p2[0]), int(p2[1])), 10, (0, 255, 0), 4)
        p2 = np.array([p2[0], p2[1], 1])

        r = np.dot(F.T, p2)
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        cv.line(image1, (x0, y0), (x1, y1), (0, 255, 0), 3)

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
