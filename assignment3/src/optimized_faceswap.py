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

bfm = h5py.File("model2017-1_face12_nomouth.h5", 'r')

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

color = np.asarray(bfm['color/model/mean'], dtype=np.float32)
color = np.reshape(color, (-1,3))

triangles = np.asarray(bfm['shape/representer/cells'], dtype=np.float32).T # 3xK

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

def rotation_y(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def general_rotation(omega):
    alpha, beta, gamma = omega
    rot = torch.zeros((3,3))
    rot[0,0] = torch.cos(alpha)*torch.cos(beta)
    rot[0,1] = torch.cos(alpha)*torch.sin(beta)*torch.sin(gamma)-torch.sin(alpha)*torch.cos(gamma)
    rot[0,2] = torch.cos(alpha)*torch.sin(beta)*torch.cos(gamma)+torch.sin(alpha)*torch.sin(gamma)
    rot[1,0] = torch.sin(alpha)*torch.cos(beta)
    rot[1,1] = torch.sin(alpha)*torch.sin(beta)*torch.sin(gamma)+torch.cos(alpha)*torch.cos(gamma)
    rot[1,2] = torch.sin(alpha)*torch.sin(beta)*torch.cos(gamma)-torch.cos(alpha)*torch.sin(gamma)
    rot[2,0] = -torch.sin(beta)
    rot[2,1] = torch.cos(beta)*torch.sin(gamma)
    rot[2,2] = torch.cos(beta)*torch.cos(gamma)
    return rot

rot10 = rotation_y(1/36 * np.pi)
rot_10 = rotation_y(-1/36 * np.pi)

G_rot10 = (rot10 @ G.T).T
G_rot_10 = (rot_10 @ G.T).T

save_obj('rot10.obj',G_rot10,color,triangles)
save_obj('rot_10.obj',G_rot_10,color,triangles)

## -------------------------- Use pinhole camera model to obtain 2D
G_new = (rot10 @ G.T).T + np.array([0, 0, -500])

def convertTo2D(G_new):
    # width = torch.max(G_new[:,0]) - torch.min(G_new[:,0])
    # heigth = torch.max(G_new[:,1]) - torch.min(G_new[:,1])
    # depth = torch.max(G_new[:,2]) - torch.min(G_new[:,2])
    width = 128
    heigth = 128
    depth = 1/2

    # Source: https://www3.ntu.edu.sg/home/ehchua/programming/opengl/CG_BasicsTheory.html
    V = torch.Tensor([[width/2, 0, 0, torch.min(G_new[:,0])+width/2],
                      [0, -heigth/2, 0, torch.min(G_new[:,1])+heigth/2],
                      [0, 0, depth, torch.min(G_new[:,2])],
                      [0, 0, 0, 1]])

    angleOfView = torch.tensor(90)
    n = torch.tensor(0.1)
    f = torch.tensor(100)
    imageAspectRatio = width/ heigth
    # scale = torch.tan(angleOfView * 0.5 * torch.pi / 180)*n
    fov_y = torch.tensor(0.5)
    scale = torch.tan(fov_y/2)*n
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

    G_def = torch.zeros((2,G_new.shape[1]))
    G_def[0, :] = G_new[0, :] / G_new[3, :]
    G_def[1, :] = G_new[1, :] / G_new[3, :]
    # G_new = torch.delete(G_new, obj=[2,3], axis=0)
    return G_def[:2,:].T # 2xN, only first 2 rows (x,y)

def convertTo2D_version2(G_new):
    vec = torch.ones((G_new.shape[0],1))
    G_conv = torch.cat((G_new,vec),1)
    n = torch.tensor(0.1)
    f = torch.tensor(100)
    W = 128
    H = 128
    fov_y = torch.tensor(0.5)

    aspect = W/H

    t = torch.tan(fov_y/2)*n
    b = -t
    r = t * aspect
    l = -t*aspect

    P = torch.Tensor([[2*n/(r-l), 0, (r+1)/(r-1), 0],
                [0, 2*n/(t-b), (t+b)/(t-b), 0],
                [0, 0, -(f+n)/(f-n), -2*f*n/(f-n)],
                [0, 0, -1, 0]])

    V = torch.Tensor([[(r-l)/2, 0, 0, (r+l)/2],
                      [0, (t-b)/2, 0, (t+b)/2],
                      [0, 0, 1/2, 1/2],
                      [0, 0, 0, 1]])

    PI = V.T @ P
    tmp = G_conv @ PI
    return (tmp[:,:2].T / tmp[:,3]).T

def zbuffer_approach(G_new):
    G_new = G_new.detach().numpy()
    # bbox = np.min(G_new.T[0]), np.max(G_new.T[0]), np.min(G_new.T[1]), np.max(G_new.T[1])
    H = 128
    W = 128
    # divide the bbox in H x W cells
    hs = np.linspace(-128,128,H)
    ws = np.linspace(-128,128,W)
    img_idx = np.zeros((H,W,3),dtype=np.uint8)

    # for every point in source, find to which cell of the buffer it belongs, and store index
    for idx, point in enumerate(G_new):
        x,y = point
        idxs = [0,0]
        for i, w in enumerate(ws):
            if (i==len(ws)-1 and w < x) or (w < x and ws[i+1] > x):
                idxs[0] = i
        for j, h in enumerate(hs):
            if (j==len(hs)-1 and h < y) or (h < y and hs[j+1] > y):
                idxs[1] = j
        # if cell is already taken, overwrite if point is closer to xy plane
        img_idx[idxs[1],idxs[0]] = (color[idx] * 255).astype(np.uint8)

    plt.imshow(img_idx)
    plt.show()

    return torch.from_numpy(img_idx)

# img_G = makeImag(G_new)

# Face Detector
detector = dlib.get_frontal_face_detector()
# Landmark Detector
predictor = openface.AlignDlib("shape_predictor_68_face_landmarks.dat")

def predict_landmarks(img):
  img = img.detach().numpy()
  dets = detector(img, 1)
  if len(dets) < 1:
    return None # Face Not Found

  print("Found %d faces" % len(dets))
  d = dets[0]
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  landmarks = predictor.findLandmarks(gray_img, d)
  return np.asarray(landmarks, dtype=np.float16)

def visualize_landmarks(img, landmarks, radius=2):
  new_img = np.copy(img)
  print(new_img.shape)
  h, w, _ = new_img.shape
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
# print(person.shape)
# print("SHAPE")
person = cv2.imread("person.jpeg")
print(person.shape)
# person = cv2.resize(person, (128,128), interpolation = cv2.INTER_AREA)
person = torch.from_numpy(person)

ground_truth = predict_landmarks(person)
ground_truth = ground_truth - np.array([person.shape[0]/2, person.shape[0]/2])
# visualize_landmarks(person,ground_truth)

indices_model = np.loadtxt("Landmarks68_model2017-1_face12_nomouth.anl")

alpha = torch.autograd.Variable(torch.randn(30,), requires_grad=True)
delta = torch.autograd.Variable(torch.randn(20,), requires_grad=True)
omega = torch.autograd.Variable(torch.Tensor([0,0,0]),requires_grad=True)
t = torch.autograd.Variable(torch.Tensor([0,0,-250]), requires_grad=True)
optim = torch.optim.Adam([alpha, delta, omega, t], lr=0.05)

old_loss = np.inf
new_loss = 10000000

epsilon = 0.0001
i = 0
while (abs(old_loss - new_loss) > epsilon):
    optim.zero_grad()
    old_loss = new_loss
    G = torch.Tensor(id_mean) + torch.Tensor(id_pcabasis30) @ (alpha * torch.sqrt(torch.Tensor(id_pcavariance30))) + torch.Tensor(expr_mean) + torch.Tensor(expr_pcabasis20) @ (delta * torch.sqrt(torch.Tensor(expr_pcavariance20)))

    G_lm = G[indices_model,:]

    # print(alpha)
    # print(delta)
    # print(omega)
    # print(t)
    # G_new = (omega @ G.T).T + t
    # save_obj("test_tmp_before.obj",G.detach().numpy(),color,triangles)
    G_new = (general_rotation(omega) @ G_lm.T).T + t
    # print(general_rotation(omega))
    # save_obj("test_tmp_after.obj",G_new.detach().numpy(),color,triangles)
    predicted = convertTo2D(G_new)
    # print(len(predicted[:,0]))
    # TODO: MAKE IMAGE OF 2D POINTS
    # G_image = zbuffer_approach(G_2D)
    # predicted = G_2D[indices_model,:]
    if i == 0 or i == 10000:
        plt.scatter(predicted[:,0].detach().numpy(), predicted[:,1].detach().numpy(),label="pred")
        plt.scatter(ground_truth[:,0], ground_truth[:,1],label="gt")
        plt.legend()
        plt.show()
    # visualize_landmarks(person, predicted, radius=1)

    print("prediction #",i)

    loss_lan = torch.mean(torch.linalg.norm(predicted - torch.from_numpy(ground_truth))**2)

    lambda_alpha = 0.5
    lambda_delta = 0.5
    loss_reg = lambda_alpha * torch.sum(alpha**2) + lambda_delta * torch.sum(delta**2)

    loss_fit = loss_lan + loss_reg
    # print(loss_lan)
    # print(loss_reg)
    print(loss_fit.item())
    

    loss_fit.backward()

    optim.step()

    new_loss = loss_fit.item()
    i = i+1

print("alpha", alpha)
print("delta", delta)
print("omega", omega)
print("t", t)

G_def = torch.Tensor(id_mean) + torch.Tensor(id_pcabasis30) @ (alpha * torch.sqrt(torch.Tensor(id_pcavariance30))) + torch.Tensor(expr_mean) + torch.Tensor(expr_pcabasis20) @ (delta * torch.sqrt(torch.Tensor(expr_pcavariance20)))

G_new_def = (general_rotation(omega) @ G.T).T + t
# G_new_def = G + t

save_obj("pytorch.obj",G_new_def.detach().numpy(),color,triangles)



