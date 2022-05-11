import numpy as np
import cv2 as cv
import open3d as o3d
from pandas import factorize
from chaining import chaining, get_row_value_idx


def normalize(column):
    return column - np.mean(column)


def factorize_and_stitch(pvm):
    # PVM = chaining()
    # Two_M_all, N_all = PVM.shape

    # print(PVM[:4,:8]) # first dense block, but too small

    # gt_pvm = np.loadtxt('../PointViewMatrix.txt')

    # 1. Select a dense block and construct D
    # D = gt_pvm.copy()
    D = pvm.copy()
    print('2m: ', D.shape[0])
    print('n: ', D.shape[1])

    # 2 Normalize the point coordinates by translating them to mean
    norm_D = np.apply_along_axis(normalize, 1, D)

    # 3 Apply SVD
    U, W, V_T = np.linalg.svd(norm_D)
    W = np.diag(W)

    # 4 Get measurement and shape matrix
    W3 = W[:3, :3]
    U3 = U[:, :3]
    V_T3 = V_T[:3, :]
    M = U3 @ np.sqrt(W3)
    S = np.sqrt(W3) @ V_T3

    # 5 Eliminate affine ambiguity (additional improvement)

    # 6 Use Procrustes analysis to find transformation between corresponding 3D points

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(S.T)
    # o3d.visualization.draw_geometries([pcd])

    return S


if __name__ == '__main__':
    factorize_and_stitch()
