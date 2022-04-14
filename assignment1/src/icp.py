import numpy as np
import open3d as o3d
import os
from sklearn.metrics import mean_squared_error
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

def icp(source, target, method='bf', epsilon=0.0001, n_space=3):
    # 1. initialize R = I , t = 0
    R = np.eye(n_space)
    R_list = [R]
    t = np.zeros(n_space)
    t_list = [t]

    RMSE_old = np.inf
    RMSE = mean_squared_error(source, target, squared=False)
    print('Initial RMSE:', RMSE)

    source_updated = deepcopy(source)

    # unless RMS is unchanged(<= epsilon)
    while np.abs(RMSE_old - RMSE) > epsilon and RMSE != 0.0:
        RMSE_old = RMSE

        ###### 2. using different sampling methods

        # 3. transform point cloud with R and t
        transformed = (source_updated.T @ R + t).T

        # matched is analogue for target but with points selected correspondingly to source
        if method == 'bf':
            matched = brute_force_cloud_matching(source=transformed, target=target)
        else:
            raise ValueError(f"'method' should be one of ['bf']. Given: {method}")

        # 5. Calculate RMSE
        # RMSE_old = np.sqrt(np.mean(np.linalg.norm(matched - transformed)**2))
        RMSE = mean_squared_error(matched, transformed, squared=False)
        print('RMSE:', RMSE)

        print('RMSE with original:', mean_squared_error(source, transformed, squared=False))

        R, t = calc_transformation(transformed, matched)
        R_list.append(R)
        t_list.append(t)
        print('R\n', R)
        print('t\n', t)

        source_updated = transformed

    R_final, t_final = calc_transformation(source, transformed)
    R_comb, t_comb = R_list[0], t_list[0]
    for R_i, t_i in zip(R_list[1:-1], t_list[1:-1]):
        R_comb = np.dot(R_comb, R_i) # becomes exactly the same as final
        t_comb = t_comb + t_i
    # print('R\n', R_comb)
    # print('t\n', t_comb)
    return transformed, R_comb, t_comb, R_final, t_final


def calc_transformation(source, target):
    # 6. Refine R and t using SVD
    # TODO: make weighted
    p_line = np.mean(source, axis=1)
    q_line = np.mean(target, axis=1)

    centered_x = source.T - p_line.T
    centered_y = target.T - q_line.T

    # TODO: add weights
    S = centered_y.T @ centered_x
    U, Sig, V_t = np.linalg.svd(S)

    # Oleg's question
    # U, Sig, V_t = np.linalg.svd(S.T)

    # Original Emily's version
    # R = U @ np.diag(np.array([1, 1, np.linalg.det(U @ V_t)])) @ V_t

    # Oleg's question/suggestion
    # R = V_t.T @ np.diag(np.array([1, 1, np.linalg.det(U @ V_t)])) @ U.T
    R = V_t.T @ U.T
    t = q_line - p_line @ R
    return R, t


def brute_force_cloud_matching(source, target):
    # 4. Find the closest point for each point in A1 based on A2 using brute-force approach
    matched = np.empty(target.T.shape)
    # TODO: parallelize this
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
    step = 20
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

    transformed, R_comb, t_comb, R_final, t_final = icp(source=bunny_source, target=bunny_target, method='bf',
                                                              epsilon=0.00001, n_space=3)

    # constructed = np.add(np.dot(R, bunny_source).T, t).T
    # transformed = (bunny_source.T @ R + t).T

    constructed_final = np.add(np.dot(R_final, bunny_source).T, t_final).T
    constructed_comb = np.add(np.dot(R_comb, bunny_source).T, t_comb).T

    source = get_point_cloud(bunny_source.T, [1, 0, 0])  # red
    target = get_point_cloud(bunny_target.T, [0, 1, 0])  # green

    constructed_final = get_point_cloud(constructed_final.T, [1, 1, 0])  # yellow
    constructed_comb = get_point_cloud(constructed_comb.T, [1, 0, 1])  # purple


    transformed = get_point_cloud(transformed.T, [0, 0, 1])  # blue
    o3d.visualization.draw_geometries([target, source,
                                       constructed_final,
                                       constructed_comb,
                                       transformed
                                       ])
