import numpy as np
from data_utils import open_bunny_data, open_wave_data
import open3d as o3d
from sklearn.metrics import mean_squared_error


DATA_DIR = '../../Data'  # This depends on where this file is located. Change for your needs.


def brute_force_cloud_matching(P, Q):
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
    # print('correspondences', correspondences[:5])
    return correspondences


def kd_tree_cloud_matching(P, Q):
    """For each point in P find closest one in Q."""
    from sklearn.neighbors import KDTree
    tree = KDTree(Q.T)
    ind = tree.query(P.T, k=1, return_distance=False)
    correspondences = [(i, chosen_idx[0]) for i, chosen_idx in enumerate(ind)]
    # print('correspondences', correspondences[:5])
    return correspondences


def center_data(data, exclude_indices=[]):
    reduced_data = np.delete(data, exclude_indices, axis=1)
    center = np.array([reduced_data.mean(axis=1)]).T
    return center, data - center


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


def icp_svd(source, target, method='bf', epsilon=0.0001, kernel=lambda diff: 1.0):
    """
    Perform ICP using SVD.

    :param source:
    :param target:
    :param method:
    :param epsilon: either this parameter or iterations have to be defined. iterations is dominant
    :param kernel:
    :return:
    """

    center_of_Q, Q_centered = center_data(target)

    RMSE_old = np.inf
    RMSE = mean_squared_error(source, target, squared=False)
    RMSE_list = [RMSE]
    print('Initial RMSE:', RMSE)

    P_values = [source.copy()]
    P_copy = source.copy()
    exclude_indices = []
    R_list, t_list = [], []

    while np.abs(RMSE_old - RMSE) > epsilon and RMSE != 0.0:
        center_of_P, P_centered = center_data(P_copy, exclude_indices=exclude_indices)

        # matched is analogue for target but with points selected correspondingly to source
        if method == 'bf':
            # matched = brute_force_cloud_matching(source=transformed, target=target)
            correspondences = brute_force_cloud_matching(P_centered, Q_centered)
        elif method == 'kd':
            correspondences = kd_tree_cloud_matching(P_centered, Q_centered)
        else:
            raise ValueError(f"'method' should be one of ['bf']. Given: {method}")

        RMSE_old = RMSE
        RMSE = mean_squared_error(P_centered, Q_centered, squared=False)
        print('RMSE:', RMSE)
        RMSE_list.append(RMSE)

        cov, exclude_indices = compute_cross_covariance(P_centered, Q_centered, correspondences, kernel)
        U, _, V_T = np.linalg.svd(cov)
        R = U.dot(V_T)
        R_list.append(R)
        t = center_of_Q - R.dot(center_of_P)
        t_list.append(t)
        print('R\n', R)
        print('t\n', t)

        P_copy = R.dot(P_copy) + t
        P_values.append(P_copy)
    return P_values, RMSE_list, R_list, t_list


def get_point_cloud(arr, c=[0, 0, 0]):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(arr)
    point_cloud.paint_uniform_color(c)
    return point_cloud


def apply_R_t_lists(source, R_list, t_list):
    source_sequential = source.copy()
    for R_i, t_i in zip(R_list, t_list):
        source_sequential = R_i.dot(source_sequential) + t_i
    return source_sequential


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
        tr_comb = np.dot(tr_comb, tr)
    return tr_comb


if __name__ == "__main__":

    bunny_source, bunny_target = open_bunny_data(DATA_DIR)
    # downsampling both point clouds for the brute force search for correspondences
    step = 1
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

    P_values, norm_values, R_list, t_list = icp_svd(source=bunny_source, target=bunny_target, method='kd',
                                                    epsilon=1e-5)

    source = get_point_cloud(bunny_source.T, [1, 0, 0])  # red
    target = get_point_cloud(bunny_target.T, [0, 1, 0])  # green

    # T = combined_homogenous(R_list, t_list)
    # R_final = T[:bunny_source.shape[0], :bunny_source.shape[0]]
    # t_final = T[0:bunny_source.shape[0], [-1]]
    # print('R_final\n', R_final)
    # print('t_final\n', t_final)
    # # constructed_final = np.add(np.dot(R_final, bunny_source), t_final)
    # constructed_final = np.dot(R_final, bunny_source) + t_final
    # constructed_final = get_point_cloud(constructed_final.T, [0, 0, 1])  # blue

    # transformed = get_point_cloud(P_values[-1].T, [0, 0, 1])  # blue. exactly matches with the sequential one

    sequential = apply_R_t_lists(bunny_source, R_list, t_list)  # make the same as the last iteration
    # sequential = sequential + 1e-5
    sequential = get_point_cloud(sequential.T, [1, 0, 1])  # purple

    o3d.visualization.draw_geometries([
        source,
        # constructed_final,
        # transformed,
        # sequential,
        target
    ])
