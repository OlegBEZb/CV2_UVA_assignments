import numpy as np
import open3d as o3d
import final_icp
from tqdm import tqdm
import os

DATA_DIR = '../Data'  # This depends on where this file is located. Change for your needs.


def get_pcd_array_picture(no):
    if no < 10:
        filename = DATA_DIR + f"/data/000000000{no}.pcd"
    else:
        filename = DATA_DIR + f"/data/00000000{no}.pcd"
    
    pcd = o3d.io.read_point_cloud(filename)
    
    # ## convert into ndarray
    pcd_arr = np.asarray(pcd.points)

    # ***  you need to clean the point cloud using a threshold ***
    bools = pcd_arr < 2
    pcd_arr_cleaned = pcd_arr[bools[:,-1]]

    return pcd_arr_cleaned

def visualize_pcd_array(array, name):
    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.points = o3d.utility.Vector3dVector(array)
    o3d.visualization.draw_geometries([vis_pcd])

    # vis = o3d.visualization.Visualizer()
    ## somewhere we should add the picture to this Visualizer?
    # vis.capture_screen_image(os.path.join(DATA_DIR+"/videos/", f"video_{name}" ))

def estimate_poses(N=1):
    R_list = []
    t_list = []

    merged_clouds = get_pcd_array_picture(99).T

    for picture_no in tqdm(range(99, N-1, -N)):
        orig_source = get_pcd_array_picture(picture_no-N)
        orig_target = get_pcd_array_picture(picture_no)

        source = orig_source.T # our icp algorithm needs dim x n
        target = orig_target.T # our icp algorithm needs dim x n

        _, R, t = final_icp.icp(source, target, method='kd', sampling='multires')
        R_list.append(R)
        t_list.append(t)

        source_reconstructed = source
        for i in range(len(R_list),0):
            source_reconstructed = final_icp.apply_R_t_lists(source_reconstructed,R_list[i-1],t_list[i-1])

        merged_clouds = np.concatenate((merged_clouds, source_reconstructed), axis=1)

    part = 1 # no sampling
    no_samples = int(merged_clouds.shape[1] / part) 
    sampled_merged_clouds = np.random.permutation(merged_clouds.T)[:no_samples, :]

    visualize_pcd_array(sampled_merged_clouds, N)
        

if __name__ == "__main__":
    estimate_poses(N=10)
    # visualize_pcd_array(get_pcd_array_picture(0), 'trial')
