# mean and std both .5 normalization
import h5py
import numpy as np
import dlib
# import openface
import cv2

# import sys
# sys.path.insert(1, '~/Documents/MSc AI/ComputerVision2/assignment3/cv2_2022_assignment3/supplemental_code/supplemental_code')
# import supplemental_code

bfm = h5py.File("../model2017-1_face12_nomouth.h5", 'r')

# Select a specific weight from BFM
id_mean = np.asarray(bfm['shape/model/mean'], dtype=np.float32) # 3N
# Sometimes you will need to reshape it to a proper shape
id_mean = np.reshape(id_mean, (-1,3))

id_pcabasis = np.asarray(bfm['shape/model/pcaBasis'], dtype=np.float32) # 3N x 199
id_pcabasis = np.reshape(id_pcabasis, (-1,3,199))
id_pcabasis30 = id_pcabasis[:,:,:30]

id_pcavariance = np.asarray(bfm['shape/model/pcaVariance'], dtype=np.float32) #199
id_pcavariance30 = id_pcavariance[:30]

expr_mean = np.asarray(bfm['expression/model/mean'], dtype=np.float32) # 3N
expr_mean = np.reshape(expr_mean, (-1,3))

expr_pcabasis = np.asarray(bfm['expression/model/pcaBasis'], dtype=np.float32) # 3N x 100
expr_pcabasis = np.reshape(expr_pcabasis, (-1,3,100))
expr_pcabasis20 = expr_pcabasis[:,:,:20]

expr_pcavariance = np.asarray(bfm['expression/model/pcaVariance'], dtype=np.float32) # 100
expr_pcavariance20 = expr_pcavariance[:20]

alpha = np.random.uniform(-1,1,30)
delta = np.random.uniform(-1,1,20)

G = id_mean + id_pcabasis30 @ (alpha * np.sqrt(id_pcavariance30)) + expr_mean + expr_pcabasis20 @ (delta * np.sqrt(expr_pcavariance20))
print(G.shape)
color = np.asarray(bfm['color/model/mean'], dtype=np.float32)
color = np.reshape(color, (-1,3))

triangles = np.asarray(bfm['shape/representer/cells'], dtype=np.float32).T # 3xK
print(triangles.shape)

print(G)

# MOETEN WE NOG FIXEN MET EEN IMPORT
def save_obj(file_path, shape, color, triangles):
    assert len(shape.shape) == 2
    assert len(color.shape) == 2
    assert len(triangles.shape) == 2
    print(shape.shape[1], color.shape[1],triangles.shape[1])
    assert shape.shape[1] == color.shape[1] == triangles.shape[1] == 3
    assert np.min(triangles) == 0
    assert np.max(triangles) < shape.shape[0]

    with open(file_path, 'wb') as f:
        data = np.hstack((shape, color))

        np.savetxt(
            f, data,
            fmt=' '.join(['v'] + ['%.5f'] * data.shape[1]))

        np.savetxt(f, triangles + 1, fmt='f %d %d %d')

save_obj('pointcloud.obj',G,color,triangles)

def rotation(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

rot10 = rotation(1/36 * np.pi)
rot_10 = rotation(-1/36 * np.pi)

G_rot10 = (rot10 @ G.T).T
print(G_rot10)
print(G_rot10.shape)
G_rot_10 = (rot_10 @ G.T).T
print(G_rot_10)
print(G_rot_10.shape)

save_obj('rot10.obj',G_rot10,color,triangles)
save_obj('rot_10.obj',G_rot_10,color,triangles)

G_new = (rot10 @ G.T).T + np.array([0, 0, -500])
print("G_new shape", G_new.shape)

# V =
# P = 
# projected = V @ P @ G_new

# Face Detector
detector = dlib.get_frontal_face_detector()
# Landmark Detector
predictor = openface.AlignDlib("../Landmarks68_model2017-1_face12_nomouth.anl")

def predict_landmarks(img):
  dets = detector(img, 1)
  if len(dets) < 1:
    return None # Face Not Found

  print("Found %d faces" % len(dets))
  d = dets[0]
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  landmarks = predictor.findLandmarks(gray_img, d)
  return np.asarray(landmarks)




# MISSCHIEN KUNNEN WE DIT GEBRUIKEN ALS WE EENMAAL DE CAMERA MATRIX HEBBEN
def project_to_image(point_cloud, p):
    """ Projects a 3D point cloud to 2D points for plotting

    :param point_cloud: 3D point cloud (3, N)
    :param p: Camera matrix (3, 4)

    :return: pts_2d: the image coordinates of the 3D points in the shape (2, N)
    """

    pts_2d = np.dot(p, np.append(point_cloud,
                                 np.ones((1, point_cloud.shape[1])),
                                 axis=0))

    pts_2d[0, :] = pts_2d[0, :] / pts_2d[2, :]
    pts_2d[1, :] = pts_2d[1, :] / pts_2d[2, :]
    pts_2d = np.delete(pts_2d, 2, 0)
    return pts_2d 

