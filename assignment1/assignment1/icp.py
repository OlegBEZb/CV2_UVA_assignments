import numpy as np
import open3d as o3d
import os

# globals.
DATA_DIR = 'Data'  # This depends on where this file is located. Change for your needs.

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
bunny_source, bunny_target = open_bunny_data()
# downsampling both point clouds for the brute force search for correspondences
bunny_source = bunny_source[:, ::50]
bunny_target = bunny_target[:, ::50]

# The most straight-forward approach to deal with number mismatch between source and target
if bunny_source.shape != bunny_target.shape:
    so = bunny_source.shape
    ta = bunny_target.shape

    if(so[1] < ta[1]):
        bunny_target = bunny_target[:,:so[1]]
    elif so[1] > ta[1]:
        bunny_source = bunny_source[:,:ta[1]]

###### 0. (adding noise)

###### 1. initialize R= I , t= 0
R = np.eye(3)
t = np.zeros(3)

###### go to 2. unless RMS is unchanged(<= epsilon)
epsilon = 0.00001
RMS_old = 200
RMS = 100
new = bunny_source
while np.abs(RMS_old - RMS) > epsilon and RMS != 0.0:
    RMS_old = RMS
###### 2. using different sampling methods

###### 3. transform point cloud with R and t
    transformed = (new.T @ R + t).T
###### 4. Find the closest point for each point in A1 based on A2 using brute-force approach
    matched = np.empty(transformed.T.shape)

    for i, point in enumerate(transformed.T):
        best_diff = 100
        closest = 0
        for target_point in bunny_target.T:
            diff = np.linalg.norm(point - target_point)
            if diff < best_diff:
                best_diff = diff
                closest = target_point
        matched[i,:] = closest
    matched = matched.T
###### 5. Calculate RMS
    RMS = np.sqrt(np.mean(np.linalg.norm(matched - transformed)**2))
    print(RMS)

###### 6. Refine R and t using SVD
    p_line = np.mean(transformed, axis=1)
    q_line = np.mean(matched, axis=1)
    centered_x = transformed.T - p_line.T
    centered_y = matched.T - q_line.T
    S = centered_y.T @ centered_x
    U, Sig, V_t = np.linalg.svd(S)
    R = U @ np.diag(np.array([1,1,np.linalg.det(U @ V_t)])) @ V_t
    t = q_line - p_line @ R

    new = transformed

constructed = np.add(np.dot(R, bunny_source).T, t).T

source = o3d.geometry.PointCloud()
source.points = o3d.utility.Vector3dVector(bunny_source.T)
source.paint_uniform_color([1,0,0]) #red

target = o3d.geometry.PointCloud()
target.points = o3d.utility.Vector3dVector(bunny_target.T)
target.paint_uniform_color([0,1,0]) #green

const = o3d.geometry.PointCloud()
const.points = o3d.utility.Vector3dVector(constructed.T)
const.paint_uniform_color([0,0,1]) #blue

o3d.visualization.draw_geometries([target, source, const])

############################
#   Merge Scene            #
############################

#  Estimate the camera poses using two consecutive frames of given data.

#  Estimate the camera pose and merge the results using every 2nd, 4th, and 10th frames.

#  Iteratively merge and estimate the camera poses for the consecutive frames.


############################
#  Additional Improvements #
############################
