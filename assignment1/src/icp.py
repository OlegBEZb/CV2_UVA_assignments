import time
import numpy as np
import open3d as o3d
import os
from sklearn.metrics import mean_squared_error
from copy import deepcopy
from sklearn.neighbors import KDTree
import pandas as pd
from data_utils import open_bunny_data, open_wave_data

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


############################
#     ICP                  #
############################

###### 0. (adding noise)

def icp(source, target, method='bf', sampling='none', epsilon=0.0001, n_space=3, **sampling_kwargs):
    # 1. initialize R = I , t = 0
    R = np.eye(n_space)
    R_list = [R]
    t = np.zeros(n_space)
    t_list = [t]

    RMSE_old = np.inf
    RMSE = 100  # just a initial value
    RMSE_list = []
    # RMSE = mean_squared_error(source, target, squared=False)
    print('Initial RMSE:', RMSE)

    source_updated = deepcopy(source)
    target_updated = deepcopy(target)

    # unless RMS is unchanged(<= epsilon)
    iter = 0
    while np.abs(RMSE_old - RMSE) > epsilon and RMSE != 0.0:
        RMSE_old = RMSE

        ###### 2. using different sampling methods

        if sampling == 'rand':
            modified_source = apply_R_t_lists(source.copy(), R_list, t_list)
            source_updated, target_updated = uniform_sampling(modified_source, target.copy(), **sampling_kwargs)
        elif sampling == 'multires':
            X = int(
                np.log(source.shape[1] / source_updated.shape[1]) / np.log(4))  # number of times already divided by 4
            if source_updated.shape[1] / 4 > 50:
                modified_source = apply_R_t_lists(source.copy(), R_list, t_list)
                source_updated, target_updated = uniform_sampling(modified_source, target.copy(), take_n=4 ** (X + 1))
                print('after multires')
                print('source_updated', source_updated.shape)
                print('target_updated', target_updated.shape)

        # 3. transform point cloud with R and t
        source_updated = (source_updated.T @ R + t).T

        # 4. Find the closest point for each point in A1 based on A2 using brute-force approach
        if method == 'bf':
            matched = brute_force_cloud_matching(source=source_updated, target=target_updated)
        elif method == 'kd':
            matched = kd_tree_cloud_matching(source=source_updated, target=target_updated)
        elif method == 'zbuf':
            source_updated, matched = z_buffer_cloud_matching(source=source_updated, target=target_updated, R=R, t=t)
            # Oleg added 27.04.2022 because of the empty array returned
            if source_updated.shape[1] == 0 or matched.shape[1] == 0:
                break
        else:
            raise ValueError(f"'method' should be one of ['bf']. Given: {method}")

        # 5. Calculate RMSE
        RMSE = mean_squared_error(matched, source_updated, squared=False)
        print('RMSE:', RMSE)
        print('RMSE with original:', mean_squared_error(source_updated, source_updated, squared=False))

        # 6. Refine R and t using SVD
        R, t = calc_transformation(source_updated, matched)
        R_list.append(R)
        t_list.append(t)
        print('R\n', R)
        print('t\n', t)

        iter += 1
        if iter >= 15:
            break

    return source_updated, R_list, t_list, RMSE_list


def calc_transformation(source, target):
    # TODO: make weighted
    p_line = np.mean(source, axis=1)
    q_line = np.mean(target, axis=1)

    centered_x = source.T - p_line.T
    centered_y = target.T - q_line.T

    # TODO: add weights
    S = centered_y.T @ centered_x
    U, Sig, V_t = np.linalg.svd(S)

    R = V_t.T @ np.diag(np.array([1, 1, np.linalg.det(V_t.T @ U.T)])) @ U.T
    t = q_line - R @ p_line
    return R, t


def brute_force_cloud_matching(source, target):
    matched = np.empty(target.T.shape)
    for i, point in enumerate(source.T):
        best_diff = np.inf
        closest = 0
        for target_point in target.T:
            diff = np.linalg.norm(point - target_point)
            if diff < best_diff:
                best_diff = diff
                closest = target_point
        matched[i, :] = closest
    matched = matched.T
    return matched


def kd_tree_cloud_matching(source, target):
    tree = KDTree(target.T)
    _, indices = tree.query(source.T)
    matched = np.empty(source.T.shape)
    for i, index in enumerate(indices):
        matched[i, :] = target.T[index]
    matched = matched.T
    return matched


def z_buffer_cloud_matching(source, target, R, t):
    # transform souce to target coordinate system
    source = (source.T @ R + t).T
    union = np.unique(np.append(source, target, axis=1).T, axis=1).T
    bbox = np.min(union[0]), np.max(union[0]), np.min(union[1]), np.max(union[1])
    H = 10
    W = 10
    # divide the bbox in H x W cells
    hs = np.linspace(bbox[2], bbox[3], H)
    ws = np.linspace(bbox[0], bbox[1], W)
    source_buf = np.zeros((H, W))
    target_buf = np.zeros((H, W))

    # for every point in source, find to which cell of the buffer it belongs, and store index
    for source_idx, point in enumerate(source.T):
        x, y, z = point
        idxs = [0, 0]
        for i, w in enumerate(ws):
            if (i == len(ws) - 1 and w < x) or (w < x and ws[i + 1] > x):
                idxs[0] = i
        for j, h in enumerate(hs):
            if (j == len(hs) - 1 and h < y) or (h < y and hs[j + 1] > y):
                idxs[1] = j
        # if cell is already taken, overwrite if point is closer to xy plane
        if source_buf[idxs[1], idxs[0]] != 0:
            cur = source.T[int(source_buf[idxs[1], idxs[0]])]
            if np.abs(cur[2]) > np.abs(point[2]):
                source_buf[idxs[1], idxs[0]] = int(
                    source_idx + 1)  # +1 needed to identify when cell is not taken (then it's 0)
        else:
            source_buf[idxs[1], idxs[0]] = int(source_idx + 1)

    # for every point in target, find to which cell of the buffer it belongs, and store index
    for target_idx, point in enumerate(target.T):
        x, y, z = point
        idxs = [0, 0]
        for i, w in enumerate(ws):
            if (i == len(ws) - 1 and w < x) or (w < x and ws[i + 1] > x):
                idxs[0] = i
        for j, h in enumerate(hs):
            if (j == len(hs) - 1 and h < y) or (h < y and hs[j + 1] > y):
                idxs[1] = j
        # if cell is already taken, overwrite if point is closer to xy plane (smaller z)
        if target_buf[idxs[1], idxs[0]] != 0:
            cur = target.T[int(target_buf[idxs[1], idxs[0]])]
            if np.abs(cur[2]) > np.abs(point[2]):
                target_buf[idxs[1], idxs[0]] = int(target_idx + 1)
        else:
            target_buf[idxs[1], idxs[0]] = int(target_idx + 1)

    # find correspondences for every source buffer cell -> look in 3x3 window around target buffer cell
    correspondences = []
    for j in range(H):
        for i in range(W):
            if source_buf[j, i] != 0:
                source_idx = source_buf[j, i] - 1  # -1 needed to make up for +1 earlier
                target_idxs = []
                for i_win in [-1, 0, 1]:
                    for j_win in [-1, 0, 1]:
                        if (i + i_win >= 0 and i + i_win < W and j + j_win >= 0 and j + j_win < H):
                            if target_buf[j + j_win, i + i_win] != 0:
                                target_idxs.append(int(target_buf[j + j_win, i + i_win] - 1))  # -1 needed
                # if there is no match, point in source is not interesting
                if target_idxs != []:
                    closest = np.inf
                    best_target_idx = 0
                    # if there are multiple target buffer cells matched, find closest (euclidean distance)
                    for idx in target_idxs:
                        target_point = target.T[idx]
                        source_point = source.T[int(source_idx)]
                        if np.linalg.norm(target_point - source_point) < closest:
                            closest = np.linalg.norm(target_point - source_point)
                            best_target_idx = idx
                    # so correspondence is list with tuple of matched (source_index, target_index)
                    correspondences.append((int(source_idx), int(best_target_idx)))

    transformed = np.empty((len(correspondences), 3))
    matched = np.empty((len(correspondences), 3))
    # transform the correspondences list to new source and target matrix
    for i, corr in enumerate(correspondences):
        transformed[i, :] = source.T[corr[0]]
        matched[i, :] = target.T[corr[1]]

    return transformed.T, matched.T


def get_point_cloud(arr, c=[0, 0, 0]):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(arr)
    point_cloud.paint_uniform_color(c)
    return point_cloud


def apply_R_t_lists(source, R_list, t_list):
    source_sequential = source.copy()
    for R_i, t_i in zip(R_list[:-1], t_list[:-1]):
        source_sequential = (source_sequential.T @ R_i + t_i).T
    return source_sequential


def uniform_sampling(source, target, take_n=1000):
    c = min(source.shape[1], target.shape[1], take_n)
    source = np.random.permutation(source.T)[:c, :].T
    target = np.random.permutation(target.T)[:c, :].T
    return source, target


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
    res_df = pd.DataFrame(columns=['icp_time', 'method', 'sampling', 'epsilon', 'rmse_list'])
    for i in range(5):
        for sampling_method in [
            # 'all',
            'uni', 'rand', 'multires',
            # 'grad'
        ]:
            for matching_method in [
                # 'bf',
                # 'kd',
                'zbuf'
            ]:
                for epsilon in [
                    1e-3,
                    # 1e-4,
                    #             1e-5, 1e-6, 1e-7
                ]:
                    print('running', sampling_method, matching_method, epsilon)

                    orig_bunny_source, orig_bunny_target = open_bunny_data(DATA_DIR)
                    bunny_source, bunny_target = orig_bunny_source.copy(), orig_bunny_target.copy()

                    start = time.time()

                    if sampling_method == 'all':
                        if matching_method == 'bf':
                            bunny_source, bunny_target = uniform_sampling(bunny_source, bunny_target, take_n=1500)

                        # The most straight-forward approach to deal with number mismatch between source and target
                        if bunny_source.shape != bunny_target.shape:
                            bunny_source, bunny_target = uniform_sampling(bunny_source, bunny_target, take_n=np.inf)

                    elif sampling_method in ['uni', 'multires']:
                        # downsampling both point clouds using uniform sampling
                        bunny_source, bunny_target = uniform_sampling(bunny_source, bunny_target, take_n=1000)
                    elif sampling_method == 'grad':
                        grad = np.gradient(bunny_source, axis=1)  # x,y,z together along all the points
                        grad = np.linalg.norm(grad, ord=2, axis=0)  # for each point we have grad
                        # quartile = np.percentile(grad, 75)  # Q3
                        # high_grad_idx_src = np.squeeze(np.argwhere(grad >= quartile))
                        high_grad_idx_src = grad.argsort()[::-1][:1000]
                        bunny_source = bunny_source[:, high_grad_idx_src]

                        # for target we just take first n max grad points, whene n is the number of points for source
                        grad = np.gradient(bunny_target, axis=1)  # x,y,z together along all the points
                        grad = np.linalg.norm(grad, ord=2, axis=0)  # for each point we have grad
                        high_grad_idx_tgt = grad.argsort()[::-1][:len(high_grad_idx_src)]
                        bunny_target = bunny_target[:, high_grad_idx_tgt]

                    print('bunny_source', bunny_source.shape)
                    print('bunny_target', bunny_target.shape)

                    # source = get_point_cloud(bunny_source.T, [1, 0, 0])  # red
                    # target = get_point_cloud(bunny_target.T, [0, 1, 0])  # green
                    # o3d.visualization.draw_geometries([target, source])

                    transformed, R_list, t_list, RMSE_list = icp(source=bunny_source, target=bunny_target,
                                                                 method=matching_method,
                                                                 sampling=sampling_method,
                                                                 epsilon=epsilon, n_space=3,
                                                                 take_n=1000)

                    end = time.time()
                    icp_time = end - start

                    constructed_comb = apply_R_t_lists(orig_bunny_source.copy(), R_list, t_list)  # make the same as the last iteration
                    constructed_comb = get_point_cloud(constructed_comb.T, [1, 0, 1])  # purple

                    source = get_point_cloud(orig_bunny_source.T, [1, 0, 0])  # red
                    target = get_point_cloud(orig_bunny_target.T, [0, 1, 0])  # green

                    transformed = get_point_cloud(transformed.T, [0, 0, 1])  # blue
                    o3d.visualization.draw_geometries([target, source,
                                                       constructed_comb,
                                                       transformed
                                                       ])

                    results = {'icp_time': icp_time,
                               'method': matching_method,
                               'sampling': sampling_method,
                               'epsilon': epsilon,
                               'rmse_list': RMSE_list}
                    print('results', results)
                    res_df = res_df.append(results, ignore_index=True)
                    print('res_df\n', res_df)
                    res_df.to_csv('../results/results_zbuf.csv', index=False)
