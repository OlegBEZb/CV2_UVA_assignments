import numpy as np
import cv2 as cv
import random
import matplotlib.pyplot as plt


def keypoint_matcher(img1, img2, n_points=8, distance_threshold=None, random_selection=False, filter_neighbours=True,
                     draw_matches=False):
    """

    :param img1:
    :param img2:
    :param n_points: the number of matched points to keep finally. -1 means to keep all. 8 by default
    :param distance_threshold: in case of consecutive images the distance should be quite low as the points do not change
    significantly their location in the space. This really helps filtering outliers
    :param random_selection:
    :param filter_neighbours: as per Lowe's paper
    :param draw_matches:
    :return:
    """
    # TODO: this may be done once instead of repetition in the chaining part
    sift = cv.SIFT_create()
    kp1, descriptors1 = sift.detectAndCompute(img1, None)
    kp2, descriptors2 = sift.detectAndCompute(img2, None)

    # TODO: if works slow, replace with KD
    matcher = cv.BFMatcher(normType=cv.NORM_L2)
    # Once it is created, two important methods are BFMatcher.match() and BFMatcher.knnMatch(). First one returns the
    # best match. Second method returns k best matches where k is specified by the user. It may be useful when we need
    # to do additional work on that.
    # TODO: check why contains duplicated pairs. It doesn't affect chaining
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    if filter_neighbours:
        # in some cases, the second closest-match may be very near to the first. It may happen due to noise or some
        # other reasons. In that case, ratio of closest-distance to second-closest distance is taken. If it is greater
        # than 0.8, they are rejected. It eliminates around 90% of false matches while discards only 5% correct matches,
        # as per Lowe's paper.
        len_before = len(matches)
        # With lower thresholds it's even better
        matches = [m for m in matches if m[0].distance / m[1].distance < 0.8]
        # print(f'Before filtering neighbours: {len_before}. After: {len(matches)}')

    if distance_threshold is not None:
        # Yes. You have the usual threshold of m.distance < n.distance in sift. That calculates distance between
        # descriptor vectors. But since the consecutive views don???t really change much, the actual location of features
        # can???t really change that much as well in the two views.
        matches_filtered_by_dist = []
        len_before = len(matches)
        for m in matches:
            # according to the doc, queryIdx refers to the first keypoints and trainIdx refers to second keypoints
            # here we just take the closest point from all neighbours
            point_1 = kp1[m[0].queryIdx].pt
            point_2 = kp2[m[0].trainIdx].pt

            if np.linalg.norm(np.array(point_1) - np.array(point_2)) < distance_threshold:
                matches_filtered_by_dist.append(m)
        matches = matches_filtered_by_dist
        # print(f'Before filtering by L2 norm with threshold {distance_threshold}: {len_before}. After: {len(matches)}')

    assert len(matches) > 8, 'not enough point for 8-points algorithm'
    # if len(matches) < n_points:
    #     print(f'Have {len(matches)} matches while asking for n_points={n_points}')

    if n_points != -1:
        if random_selection:
            # get random subset of matches list
            assert len(matches) >= n_points, "'random.sample' fails if 'n_points' is larger than the number of matches"
            matches = random.sample(matches, n_points)
        else:
            # Sort them in the order of distance between descriptors.
            matches = sorted(matches, key=lambda x: x[0].distance)[:n_points]

    if draw_matches:
        # cv.drawMatches() draws the matches. It stacks two images horizontally and draw lines from first image to
        # second image showing best matches. There is also cv.drawMatchesKnn which draws all the k best matches. If k=2,
        # it will draw two match-lines for each keypoint.
        plt.figure(figsize=(15, 15))
        show_matches = cv.drawMatchesKnn(img1, kp1, img2, kp2,
                                         [(m[0],) for m in matches[:8]],  # the nearest neighbour of 8 matches
                                         None)
        plt.imshow(show_matches)

    # extracting coordinates of matches points
    matched_points1 = []
    matched_points2 = []
    for m in matches:
        # according to the doc, queryIdx refers to the first keypoints and trainIdx refers to second keypoints
        # here we just take the closest point from all neighbours
        point_1 = kp1[m[0].queryIdx].pt
        point_2 = kp2[m[0].trainIdx].pt

        matched_points1.append(point_1)
        matched_points2.append(point_2)

    return matches, matched_points1, matched_points2, kp1, kp2


def normalize_points(matched_points):
    mx = np.mean([p[0] for p in matched_points])
    my = np.mean([p[1] for p in matched_points])
    d = sum([np.sqrt((p[0] - mx) ** 2 + (p[1] - my) ** 2) for p in matched_points]) / len(matched_points)
    print(f"mean x: {round(mx, 3)}, mean y: {round(my, 3)}, average distance to the mean: {round(d, 3)}")

    coef = np.sqrt(2) / d

    T = np.array([[coef, 0, -mx * coef],
                  [0, coef, -my * coef],
                  [0, 0, 1]])

    result_list = []
    for p_i in matched_points:
        p_i = np.array([p_i[0], p_i[1], 1])
        p_i_hat = np.dot(T, p_i)
        result_list.append((p_i_hat[0], p_i_hat[1]))

    return result_list, T


def sampson_distance(p1, p2, F):
    p1 = np.array([p1[0], p1[1], 1])
    p2 = np.array([p2[0], p2[1], 1])

    Fp1 = np.dot(F, p1)
    Fp2 = np.dot(F.T, p2)
    denom = Fp1[0] ** 2 + Fp1[1] ** 2 + Fp2[0] ** 2 + Fp2[1] ** 2
    err = np.dot(np.dot(p2.T, F), p1) ** 2 / denom
    return err


def get_fundamental_matrix(matched_points1, matched_points2, normalize=True):
    assert len(matched_points1) == len(matched_points2)
    n = len(matched_points1)

    if normalize:
        matched_points1, T = normalize_points(matched_points1)
        matched_points2, T_prime = normalize_points(matched_points2)

    A = []
    for p1, p2 in zip(matched_points1, matched_points2):
        x1, y1 = p1
        x2, y2 = p2
        A.append([x1 * x2, x1 * y2, x1,
                  y1 * x2, y1 * y2, y1,
                  x2, y2, 1])
    A = np.array(A)
    assert A.shape == (n, 9)
    U, D, V_t = np.linalg.svd(A)
    print('U, D, V_t = np.linalg.svd(A)', U.shape, D.shape, V_t.shape)

    # slide 17: https://inst.eecs.berkeley.edu/~ee290t/fa19/lectures/lecture9-4-computing-the-fundamental-matrix.pdf
    # The entries of F are the components of the column of V corresponding to the smallest singular value.
    tmp = V_t.T[:, -1]
    print(f'V_t.T[:, n - 1]\n{tmp}\n{tmp.shape}')
    F = tmp.reshape((3, 3))
    print(f'F\n{F}\n{F.shape}')
    FU, FD, FV_t = np.linalg.svd(F)
    FD_prime = FD.copy()
    print('FD_prime', FD_prime)
    FD_prime[np.argmin(FD)] = 0
    print('FD_prime with the smallest singular value zeroed', FD_prime)
    new_F = np.dot(np.dot(FU, np.diag(FD_prime)), FV_t)

    if normalize:
        new_F = np.dot(np.dot(T_prime.T, new_F), T)

    return new_F


def get_fundamental_matrix_ransac(matched_points1, matched_points2, threshold=1, iterations=100):
    # TODO: make recursive

    best_F = None
    best_inliers_num = 0

    for i in range(iterations):
        import random
        iter_points1, iter_points2 = zip(*random.sample(list(zip(matched_points1, matched_points2)), 8))
        F = get_fundamental_matrix(iter_points1, iter_points2, normalize=True)
        inliers = len(
            [1 for p1, p2 in zip(matched_points1, matched_points2) if sampson_distance(p1, p2, F) < threshold])
        if inliers > best_inliers_num:
            best_inliers_num = inliers
            print('new best number of inliers', best_inliers_num)
            best_F = F

    return best_F


def drawlines(img1, img2, lines, pts1, pts2, colors):
    ''' img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines '''

    img1 = img1.copy()
    img2 = img2.copy()

    r, c, _ = img1.shape
    for r, pt1, pt2, color in zip(lines, pts1, pts2, colors):
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 2)
        img1 = cv.circle(img1, (int(pt1[0]), int(pt1[1])), 5, color, 5)
        img2 = cv.circle(img2, (int(pt2[0]), int(pt2[1])), 5, color, 5)
    return img1, img2


def random_colors(lines):
    n = len(lines)
    colors = []
    for i in range(n):
        colors.append(tuple(np.random.randint(0, 255, 3).tolist()))
    return colors


if __name__ == '__main__':
    image1 = cv.imread("../Data/House/frame00000001.png")
    image2 = cv.imread("../Data/House/frame00000049.png")

    n = 8
    matches, matched_points1, matched_points2, kp1, kp2 = keypoint_matcher(image1, image2)

    F = get_fundamental_matrix(matched_points1, matched_points2)
    print(F)
