# mean and std both .5 normalization
from tkinter import Variable
import h5py
import numpy as np
import dlib
import openface
import cv2
import matplotlib.pyplot as plt
import torch

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


# ANGELA
## -------------------------- Use pinhole camera model to obtain 2D
G_new = (rot10 @ G.T).T + np.array([0, 0, -500])

def convertTo2D(G_new):
    width = torch.max(G_new[:,0]) - torch.min(G_new[:,0])
    heigth = torch.max(G_new[:,1]) - torch.min(G_new[:,1])
    depth = torch.max(G_new[:,2]) - torch.min(G_new[:,2])

    # Source: https://www3.ntu.edu.sg/home/ehchua/programming/opengl/CG_BasicsTheory.html
    V = torch.Tensor([[width/2, 0, 0, torch.min(G_new[:,0])+width/2],
                [0, -heigth/2, 0, torch.min(G_new[:,1])+heigth/2],
                [0, 0, depth, torch.min(G_new[:,2])],
                [0, 0, 0, 1]])

    angleOfView = torch.tensor(90)
    n = torch.tensor(0.1)
    print(n)
    f = torch.tensor(100)
    print(f)
    imageAspectRatio = width/ heigth
    scale = torch.tan(angleOfView * 0.5 * torch.pi / 180)*n
    r = imageAspectRatio * scale
    l = -r
    t =  scale
    b = -t

    P = torch.Tensor([[2*n/(r-l), 0, 0, 0],
                [0, 2*n/(t-b), 0, 0],
                [(r+1)/(r-1), (t+b)/(t-b), -(f+n)/(f-n), -1],
                [0, 0, -2*f*n/(f-n), 0]])

    vec = torch.ones((G_new.shape[0],1))
    G_new = torch.cat((G_new,vec),1)
    G_new = V @ P @ G_new.T

    G_new[0, :] = G_new[0, :] / G_new[3, :]
    G_new[1, :] = G_new[1, :] / G_new[3, :]
    # G_new = torch.delete(G_new, obj=[2,3], axis=0)
    return G_new[:2,:] # 2xN


#TODO: Ask question on how to go from 2D point cloud to image: rounding to integers decreases the resolution
#How to go from 2xN to an image that is plottable
def makeImag(G_new):
    minX = np.amin(G_new[0,:])
    maxX = np.amax(G_new[0,:])
    minY = np.amin(G_new[1,:])
    maxY = np.amax(G_new[1,:])
    img = np.zeros(shape=(int(maxX-minX)+1, int(maxY-minY)+1))
    for i in range(G_new.shape[1]):
        x, y = G_new[:,i]
        img[int(x+np.abs(minX)),int(y+np.abs(minY))] = 1

    plt.imshow(img)
    plt.show()

    return img



def zbuffer_approach(G_new):
    # G_new = G_new.detach().numpy().T
    bbox = np.min(G_new.T[0]), np.max(G_new.T[0]), np.min(G_new.T[1]), np.max(G_new.T[1])
    H = 600
    W = 400
    # divide the bbox in H x W cells
    hs = np.linspace(bbox[2],bbox[3],H)
    ws = np.linspace(bbox[0],bbox[1],W)
    img_idx = np.zeros((H,W))

    # for every point in source, find to which cell of the buffer it belongs, and store index
    for idx, point in enumerate(G_new):
        # print(point)
        x,y, z = point
        idxs = [0,0]
        for i, w in enumerate(ws):
            if (i==len(ws)-1 and w < x) or (w < x and ws[i+1] > x):
                idxs[0] = i
        for j, h in enumerate(hs):
            if (j==len(hs)-1 and h < y) or (h < y and hs[j+1] > y):
                idxs[1] = j
        # if cell is already taken, overwrite if point is closer to xy plane
        if img_idx[idxs[1],idxs[0]] != 0:
            cur = G_new[int(img_idx[idxs[1],idxs[0]])]
            # if np.abs(cur[2]) > np.abs(point[2]):
            #     img_idx[idxs[1],idxs[0]] = int(idx + 1) # +1 needed to identify when cell is not taken (then it's 0)
        else:
            img_idx[idxs[1],idxs[0]] = int(idx + 1)

    img_idx[img_idx > 0] = 1

    plt.imshow(img_idx)
    plt.show()

    return img_idx

# img_G = zbuffer_approach(G_new)

# _ = zbuffer_approach(G)
# _ = zbuffer_approach(G_rot10)
# _ = zbuffer_approach(G_rot_10)
# G_new_image = zbuffer_approach(G_new)

# Face Detector
detector = dlib.get_frontal_face_detector()
# Landmark Detector
predictor = openface.AlignDlib("shape_predictor_68_face_landmarks.dat")

def predict_landmarks(img):
  img = (img*255).astype('uint8')
  dets = detector(img, 1)
  if len(dets) < 1:
    return None # Face Not Found

  print("Found %d faces" % len(dets))
  d = dets[0]
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  landmarks = predictor.findLandmarks(gray_img, d)
  return np.asarray(landmarks)

def visualize_landmarks(img, landmarks, radius=2):
  new_img = np.copy(img)
  h, w = new_img.shape
  for x, y in landmarks:
    x = int(x)
    y = int(y)
    new_img[max(0,y-radius):min(h-1,y+radius),max(0,x-radius):min(w-1,x+radius)] = (255, 0, 0)
  plt.imshow(new_img[...,::-1])
  plt.axis(False)
  plt.show()

# landmarks = predict_landmarks(img_G)
# visualize_landmarks(img_G, landmarks, 4)

#### LATENT PARAMETERS ESTIMATION
# person = cv2.imread("oleg.jpg")
person = cv2.imread("person.jpeg")
ground_truth = predict_landmarks(person)
# visualize_landmarks(person,ground_truth,radius=4)

alpha = torch.Tensor(torch.randn(30,))
delta = torch.Tensor(torch.randn(20,))
omega = torch.Tensor(torch.randn(3,3))
t = torch.Tensor([0,0,-500])
optim = torch.optim.Adam([alpha, delta, omega, t], lr=0.02)

for i in range(10):
    G = torch.Tensor(id_mean) + torch.Tensor(id_pcabasis30) @ (alpha * torch.sqrt(torch.Tensor(id_pcavariance30))) + torch.Tensor(expr_mean) + torch.Tensor(expr_pcabasis20) @ (delta * torch.sqrt(torch.Tensor(expr_pcavariance20)))

    G_new = (omega @ G.T).T + t

    G_2D = convertTo2D(G_new)
    # TODO: MAKE IMAGE OF 2D POINTS
    G_image = zbuffer_approach(G_2D)
    predicted = predict_landmarks(G_image)
    print("prediction #",i, predicted)

    loss_lan = torch.mean(torch.linalg.norm(predicted - ground_truth)**2)
    lambda_alpha = 0.5
    lambda_delta = 0.5
    loss_reg = lambda_alpha * torch.sum(alpha**2) + lambda_delta * torch.sum(delta**2)

    loss_reg.backward()
    optim.step()

print("alpha", alpha)
print("delta", delta)
print("omega", omega)
print("t", t)

G_def = id_mean + id_pcabasis30 @ (alpha * np.sqrt(id_pcavariance30)) + expr_mean + expr_pcabasis20 @ (delta * np.sqrt(expr_pcavariance20))

G_new_def = (omega @ G.T).T + t

save_obj("pytorch.obj",G_new_def,color,triangles)



