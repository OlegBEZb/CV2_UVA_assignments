import numpy as np
import open3d as o3d
import os
from sklearn.metrics import mean_squared_error
from copy import deepcopy
from sklearn.neighbors import KDTree

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

def icp(source, target, method='bf', sampling='none', epsilon=0.0001, n_space=3, **sampling_kwargs):
    # 1. initialize R = I , t = 0
    R = np.eye(n_space)
    R_list = [R]
    t = np.zeros(n_space)
    t_list = [t]

    RMSE_old = np.inf
    RMSE = 100  # just a initial value
    # RMSE = mean_squared_error(source, target, squared=False)
    print('Initial RMSE:', RMSE)

    source_updated = deepcopy(source)
    target_updated = deepcopy(target)

    # unless RMS is unchanged(<= epsilon)
    while np.abs(RMSE_old - RMSE) > epsilon and RMSE != 0.0:
        RMSE_old = RMSE

        ###### 2. using different sampling methods

        if sampling == 'rand':
            modified_source = apply_R_t_lists(source.copy(), R_list, t_list)
            source_updated, target_updated = uniform_sampling(modified_source, target.copy(), **sampling_kwargs)
        elif sampling == 'multires':
            X = int(np.log(source.shape[1] / source_updated.shape[1]) / np.log(4))  # number of times already divided by 4
            if source_updated.shape[1] / 4 > 50:
                modified_source = apply_R_t_lists(source.copy(), R_list, t_list)
                source_updated, target_updated = uniform_sampling(modified_source, target.copy(), part=4 ** (X + 1))

        # 3. transform point cloud with R and t
        transformed = (source_updated.T @ R + t).T

        # 4. Find the closest point for each point in A1 based on A2 using brute-force approach
        if method == 'bf':
            matched = brute_force_cloud_matching(source=transformed, target=target_updated)
        elif method == 'kd':
            matched = kd_tree_cloud_matching(source=transformed, target=target_updated)
        else:
            raise ValueError(f"'method' should be one of ['bf']. Given: {method}")

        # 5. Calculate RMSE
        RMSE = mean_squared_error(matched, transformed, squared=False)
        print('RMSE:', RMSE)
        print('RMSE with original:', mean_squared_error(source_updated, transformed, squared=False))

        # 6. Refine R and t using SVD
        R, t = calc_transformation(transformed, matched)
        R_list.append(R)
        t_list.append(t)
        print('R\n', R)
        print('t\n', t)

        source_updated = transformed

    return transformed, R_list, t_list


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


def uniform_sampling(source, target, part=50):
    c = int(min(bunny_source.shape[1], bunny_target.shape[1]) / part)
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
    SAMPLING_METHOD = 'multires'  # all, uni, rand, multires, grad

    orig_bunny_source, orig_bunny_target = open_bunny_data()
    bunny_source, bunny_target = orig_bunny_source.copy(), orig_bunny_target.copy()

    if SAMPLING_METHOD == 'all':
        # downsampling both point clouds for the brute force search for correspondences
        step = 1 # 100
        bunny_source = bunny_source[:, ::step]
        bunny_target = bunny_target[:, ::step]

        # The most straight-forward approach to deal with number mismatch between source and target
        if bunny_source.shape != bunny_target.shape:
            so = bunny_source.shape
            ta = bunny_target.shape

            # TODO: replace with sampling
            if (so[1] < ta[1]):
                bunny_target = bunny_target[:, :so[1]]
            elif so[1] > ta[1]:
                bunny_source = bunny_source[:, :ta[1]]
    elif SAMPLING_METHOD == 'uni':
        # downsampling both point clouds using uniform sampling
        bunny_source, bunny_target = uniform_sampling(bunny_source, bunny_target, part=50)
    elif SAMPLING_METHOD == 'grad':
        grad = np.gradient(bunny_source, axis=1)  # x,y,z together along all the points
        grad = np.linalg.norm(grad, ord=2, axis=0)  # for each point we have grad
        quartile = np.percentile(grad, 75)  # Q1
        high_grad_idx_src = np.squeeze(np.argwhere(grad >= quartile))
        bunny_source = bunny_source[:, high_grad_idx_src]

        # for target we just take first n max grad points, whene n is the number of points for source
        grad = np.gradient(bunny_target, axis=1)  # x,y,z together along all the points
        grad = np.linalg.norm(grad, ord=2, axis=0)  # for each point we have grad
        high_grad_idx_tgt = grad.argsort()[::-1][:len(high_grad_idx_src)]
        bunny_target = bunny_target[:, high_grad_idx_tgt]

    source = get_point_cloud(bunny_source.T, [1, 0, 0])  # red
    target = get_point_cloud(bunny_target.T, [0, 1, 0])  # green
    o3d.visualization.draw_geometries([target, source])

    transformed, R_list, t_list = icp(source=bunny_source, target=bunny_target, method='kd',
                                      sampling=SAMPLING_METHOD, epsilon=0.00001, n_space=3, part=50)

    constructed_comb = apply_R_t_lists(orig_bunny_source.copy(), R_list, t_list)  # make the same as the last iteration
    constructed_comb = get_point_cloud(constructed_comb.T, [1, 0, 1])  # purple

    source = get_point_cloud(orig_bunny_source.T, [1, 0, 0])  # red
    target = get_point_cloud(orig_bunny_target.T, [0, 1, 0])  # green

    transformed = get_point_cloud(transformed.T, [0, 0, 1])  # blue
    o3d.visualization.draw_geometries([target, source,
                                       constructed_comb,
                                       transformed
                                       ])
