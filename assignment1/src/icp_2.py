import numpy as np
import open3d as o3d
import os
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from copy import deepcopy

# globals.
DATA_DIR = '../Data'  # This depends on where this file is located. Change for your needs.


# == Load data ==
def open3d_example():
    pcd = o3d.io.read_point_cloud("Data/data/0000000000.pcd")
    # ## convert into ndarray

    pcd_arr = np.asarray(pcd.points)

    # ***  you need to clean the point cloud using a threshold ***
    pcd_arr_cleaned = pcd_arr

    # visualization from ndarray
    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.points = o3d.utility.Vector3dVector(pcd_arr_cleaned)
    o3d.visualization.draw_geometries([vis_pcd])


def open_wave_data():
    target = np.load(os.path.join(DATA_DIR, 'wave_target.npy'))
    source = np.load(os.path.join(DATA_DIR, 'wave_source.npy'))
    return source, target


def open_bunny_data():
    target = np.load(os.path.join(DATA_DIR, 'bunny_target.npy'))
    source = np.load(os.path.join(DATA_DIR, 'bunny_source.npy'))
    return source, target


############################
#     ICP                  #
############################

###### 0. (adding noise)

def icp(source, target, method='bf', epsilon=0.0001, n_space=3, kernel=lambda diff: 1.0):
    # 1. initialize R = I , t = 0
    R = np.eye(n_space)
    R_list = [R]
    t = np.zeros((n_space, 1))
    t_list = [t]

    RMSE_old = np.inf
    RMSE = mean_squared_error(source, target, squared=False)
    RMSE_list = [RMSE]
    print('Initial RMSE:', RMSE)

    source_updated = deepcopy(source)

    # unless RMS is unchanged(<= epsilon)
    while np.abs(RMSE_old - RMSE) > epsilon and RMSE != 0.0:
        RMSE_old = RMSE

        ###### 2. using different sampling methods

        # 3. transform point cloud with R and t
        # transformed = (source_updated.T @ R + t).T
        transformed = R.dot(source_updated) + t

        # matched is analogue for target but with points selected correspondingly to source
        if method == 'bf':
            # matched = brute_force_cloud_matching(source=transformed, target=target)
            correspondences = get_correspondence_indices(transformed, target)
        else:
            raise ValueError(f"'method' should be one of ['bf']. Given: {method}")

        # 5. Calculate RMSE
        # RMSE_old = np.sqrt(np.mean(np.linalg.norm(matched - transformed)**2))
        # RMSE = mean_squared_error(transformed, matched, squared=False)
        # RMSE_list.append(RMSE)
        # print('RMSE:', RMSE)

        print('RMSE with original:', mean_squared_error(source, transformed, squared=False))

        # R, t = calc_transformation(transformed, matched, target, kernel)
        R, t, RMSE = calc_transformation(transformed, target, correspondences, kernel)
        RMSE_list.append(RMSE)
        print('RMSE:', RMSE)
        R_list.append(R)
        t_list.append(t)
        print('R\n', R)
        print('t\n', t)

        source_updated = transformed

    plt.plot(RMSE_list)

    R_final, t_final = calc_transformation(source, transformed, target, kernel)

    return transformed, R_list, t_list, R_final, t_final


# def compute_cross_covariance(source, target, kernel=lambda diff: 1.0):
#     dim = source.shape[0]
#     cov = np.zeros((dim, dim))
#     exclude_indices = []
#     for i, points in enumerate(zip(source.T, target.T)):
#         p_point, q_point = points
#         weight = kernel(p_point - q_point)
#         if weight < 0.01: exclude_indices.append(i)
#         cov += weight * q_point.dot(p_point.T)
#     return cov, exclude_indices

def compute_cross_covariance(P, Q, correspondences, kernel=lambda diff: 1.0):
    dim = P.shape[0]
    cov = np.zeros((dim, dim))
    exclude_indices = []
    for i, j in correspondences:
        p_point = P[:, [i]]
        q_point = Q[:, [j]]
        weight = kernel(p_point - q_point)
        if weight < 0.01: exclude_indices.append(i)
        cov += weight * q_point.dot(p_point.T)
    return cov, exclude_indices


def get_correspondence_indices(P, Q):
    """For each point in P find closest one in Q."""
    p_size = P.shape[1]
    q_size = Q.shape[1]
    correspondences = []
    for i in range(p_size):
        p_point = P[:, i]
        min_dist = np.inf
        chosen_idx = -1
        for j in range(q_size):
            q_point = Q[:, j]
            dist = np.linalg.norm(q_point - p_point)
            if dist < min_dist:
                min_dist = dist
                chosen_idx = j
        correspondences.append((i, chosen_idx))
    return correspondences


def center_data(data, exclude_indices=[]):
    reduced_data = np.delete(data, exclude_indices, axis=1)
    center = np.array([reduced_data.mean(axis=1)]).T
    return center, data - center


def calc_transformation(source, target, correspondences, kernel):
    # 6. Refine R and t using SVD
    # TODO: make weighted
    source_center, source_centered = center_data(source)
    print('source_center, source_centered', source_center, source_centered.shape)
    target_center, target_centered = center_data(target)
    print('target_center, target_centered', target_center, target_centered.shape)

    # original_target_center, original_target_centered = center_data(original_target)
    # print('original_target_center', original_target_center)

    # TODO: add weights
    # S = centered_y.T @ centered_x
    # cov, _ = compute_cross_covariance(source_centered, target_centered, kernel)
    cov, _ = compute_cross_covariance(source_centered, target_centered, correspondences, kernel)
    print('cov', cov, cov.shape)
    U, Sig, V_T = np.linalg.svd(cov)
    print('U', U, U.shape)
    print('V_T', V_T, V_T.shape)

    dim = source.shape[0]
    diag = np.append(np.ones(dim - 1), np.linalg.det(U @ V_T))
    R = U @ np.diag(diag) @ V_T
    # t = target_center - R.dot(source_center)
    t = target_center - R.dot(source_center)

    RMSE = mean_squared_error(source_centered, target_centered, squared=False)

    return R, t, RMSE  # rotated weridly without T


def brute_force_cloud_matching(source, target):
    # 4. Find the closest point for each point in A1 based on A2 using brute-force approach
    assert source.shape[0] == target.shape[0]

    matched = np.empty(target.shape)
    # TODO: parallelize this
    for i, point in enumerate(source.T):
        best_diff = np.inf
        closest = 0
        for target_point in target.T:
            diff = np.linalg.norm(point - target_point)
            if diff < best_diff:
                best_diff = diff
                closest = target_point
        matched[:, i] = closest
    return matched


def list2homogenous(R_list, t_list):
    tr_list = []
    for R_i, t_i in zip(R_list, t_list):
        tr = np.eye(R_i.shape[0] + 1)
        tr[0:R_i.shape[0], 0:R_i.shape[0]] = R_i
        tr[0:R_i.shape[0], [-1]] = t_i
        tr_list.append(tr)
    return tr_list


def combined_homogenous(R_list, t_list):
    tr_list = list2homogenous(R_list, t_list)

    tr_comb = tr_list[0]
    for tr in tr_list[1:]:
        np.dot(tr_comb, tr)
    return tr_comb


def apply_R_t_lists(source, R_list, t_list):
    source_sequential = source.copy()
    for R_i, t_i in zip(R_list, t_list):
        source_sequential = R_i.dot(source_sequential) + t_i
    return source_sequential


def get_point_cloud(arr, c=[0, 0, 0]):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(arr)
    point_cloud.paint_uniform_color(c)
    return point_cloud


############################
#   Merge Scene            #
############################

#  Estimate the camera poses using two consecutive frames of given data.

#  Estimate the camera pose and merge the results using every 2nd, 4th, and 10th frames.

#  Iteratively merge and estimate the camera poses for the consecutive frames.


############################
#  Additional Improvements #
############################

if __name__ == "__main__":

    bunny_source, bunny_target = open_bunny_data()
    # downsampling both point clouds for the brute force search for correspondences
    step = 50
    bunny_source = bunny_source[:, ::step]
    bunny_target = bunny_target[:, ::step]

    # The most straight-forward approach to deal with number mismatch between source and target
    if bunny_source.shape != bunny_target.shape:
        so = bunny_source.shape
        ta = bunny_target.shape

        if (so[1] < ta[1]):
            bunny_target = bunny_target[:, :so[1]]
        elif so[1] > ta[1]:
            bunny_source = bunny_source[:, :ta[1]]

    transformed, R_list, t_list, R_final, t_final = icp(source=bunny_source, target=bunny_target, method='bf',
                                                        epsilon=1e-4, n_space=3)

    source = get_point_cloud(bunny_source.T, [1, 0, 0])  # red
    target = get_point_cloud(bunny_target.T, [0, 1, 0])  # green

    # constructed_final = np.add(np.dot(R_final, bunny_source).T, t_final)
    # constructed_final = get_point_cloud(constructed_final.T, [1, 1, 0])  # yellow

    transformed = get_point_cloud(transformed.T, [0, 0, 1])  # blue

    sequential = apply_R_t_lists(bunny_source, R_list[1:-1], t_list[1:-1])  # make the same as the last iteration
    # sequential = sequential + 1e-5
    sequential = get_point_cloud(sequential.T, [1, 0, 1])  # purple

    o3d.visualization.draw_geometries([
        source,
        # constructed_final,
        # transformed,
        sequential,
        target
    ])
