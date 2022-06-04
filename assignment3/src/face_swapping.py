from pickletools import optimize
from tkinter import Variable
import h5py
import numpy as np
import dlib
import openface
import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image

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

def convertTo2D(G_new):
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
    f = torch.tensor(10)
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

def render(uvz, color, triangles, H=480, W=640):
    """ Renders an image of size WxH given u, v, z vertex coordinates, vertex color and triangle topology.
    
    uvz - matrix of shape Nx3, where N is an amount of vertices
    color - matrix of shape Nx3, where N is an amount of vertices, 3 channels represent R,G,B color scaled from 0 to 1
    triangles - matrix of shape Mx3, where M is an amount of triangles, each column represents a vertex index
    """

    assert len(uvz.shape) == 2
    assert len(color.shape) == 2
    assert len(triangles.shape) == 2
    assert uvz.shape[1] == color.shape[1] == triangles.shape[1] == 3
    assert np.min(triangles) == 0
    assert np.max(triangles) < uvz.shape[0]

    def bbox(v0, v1, v2):
        u_min = int(max(0, np.floor(min(v0[0], v1[0], v2[0]))))
        u_max = int(min(W - 1, np.ceil(max(v0[0], v1[0], v2[0]))))

        v_min = int(max(0, np.floor(min(v0[1], v1[1], v2[1]))))
        v_max = int(min(H - 1, np.ceil(max(v0[1], v1[1], v2[1]))))

        return u_min, u_max, v_min, v_max

    def cross_product(a, b):
        return a[0] * b[1] - b[0] * a[1]

    p = np.zeros([3])

    z_buffer = -np.ones([H, W]) * 100500

    image = np.zeros([H, W, 3])

    for triangle in triangles:
        id0, id1, id2 = triangle
        v0 = uvz[id0]
        v1 = uvz[id1]
        v2 = uvz[id2]
        v02 = v2 - v0
        v01 = v1 - v0

        u_min, u_max, v_min, v_max = bbox(v0, v1, v2)

        # double triangle signed area
        tri_a = cross_product(v1 - v0, v2 - v0)
        for v in range(v_min, v_max + 1):
            p[1] = v
            for u in range(u_min, u_max + 1):
                p[0] = u

                v0p = p - v0

                b1 = cross_product(v0p, v02) / tri_a
                b2 = cross_product(v01, v0p) / tri_a
                if (b1 < 0) or (b2 < 0) or (b1 + b2 > 1):
                    continue
                
                b0 = 1 - b1 - b2
                p[2] = b0 * v0[2] + b1 * v1[2] + b2 * v2[2]

                if p[2] > z_buffer[v, u]:
                    z_buffer[v, u] = p[2]
                    image[v, u] = b0 * color[id0] + b1 * color[id1] + b2 * color[id2]
    
    return image


def texturing_for_faceswap(G, color, triangles, shape, mask):
    G = G.astype(int)
    x_min = np.min(G[:,0])
    G[:,1] = -G[:,1]    # flip the y-axis
    y_min = np.min(G[:,1])

    # if image will be used be for faceswap, place face on position of mask
    zero_indices_x, zero_indices_y, _ = np.where(mask==0)

    x_correction = np.min(zero_indices_x)
    y_correction = np.min(zero_indices_y)

    # translate the image to have all positive indices
    G[:,0] += -x_min + y_correction
    G[:,1] += -y_min + x_correction

    # if image will be used for faceswap, image should be of the same size
    height, width, _ = shape
    # print("in texturing for face swap")
    # print("width ", width.shape)
    # print("height ", height.shape)
    save_obj("probeersel.obj", G, color, triangles)
    image = render(G, color, triangles.astype(int), H=height, W=width)
    return image


def texturing(G, color, triangles):
    G = G.astype(int)
    x_min = np.min(G[:,0])
    G[:,1] = -G[:,1]    # flip the y-axis
    y_min = np.min(G[:,1])

    # translate the image to have all positive indices
    G[:,0] += -x_min 
    G[:,1] += -y_min 

    # if image will be used for faceswap, image should be of the same size
    width = int(np.ceil(np.array(np.max(G[:,0]) - np.min(G[:,0]))))
    height = int(np.ceil(np.array(np.max(G[:,1]) - np.min(G[:,1]))))

    image = render(G, color, triangles.astype(int), H=height, W=width)
    return image

# landmarks = predict_landmarks(img_G)
# visualize_landmarks(img_G, landmarks, 4)

### LATENT PARAMETERS ESTIMATION


def find_ground_truth_landmarks(person):
    ground_truth = predict_landmarks(person)
    # visualize_landmarks(person, ground_truth, 7)

    x_translate = np.min(ground_truth[:,0]) + (np.max(ground_truth[:,0]) - np.min(ground_truth[:,0]))/2
    y_translate = np.min(ground_truth[:,1]) + (np.max(ground_truth[:,1]) - np.min(ground_truth[:,1]))/2
    ground_truth = ground_truth - np.array([x_translate, y_translate])
    return ground_truth




def optimize_geometry(ground_truth_landmarks, file_name):
    indices_model = np.loadtxt("Landmarks68_model2017-1_face12_nomouth.anl")

    alpha = torch.autograd.Variable(torch.randn(30,), requires_grad=True)
    delta = torch.autograd.Variable(torch.randn(20,), requires_grad=True)
    omega = torch.autograd.Variable(torch.Tensor([0,0,0]),requires_grad=True)
    t = torch.autograd.Variable(torch.Tensor([0,0,-200]), requires_grad=True)
    optim = torch.optim.Adam([alpha, delta, omega, t], lr=0.05)

    old_loss = np.inf
    new_loss = 10000000

    epsilon = 0.00001
    i = 0
    while (abs(old_loss - new_loss) > epsilon):
    # while (i < 100):
        optim.zero_grad()
        old_loss = new_loss
        G = torch.Tensor(id_mean) + torch.Tensor(id_pcabasis30) @ (alpha * torch.sqrt(torch.Tensor(id_pcavariance30))) + torch.Tensor(expr_mean) + torch.Tensor(expr_pcabasis20) @ (delta * torch.sqrt(torch.Tensor(expr_pcavariance20)))

        G_lm = G[indices_model,:]
        G_new = (general_rotation(omega) @ G_lm.T).T + t
        predicted = convertTo2D(G_new)

        # if i % 1000 == 0:
        #     tmp_predicted = predicted.detach().numpy()
        #     # gt = ground_truths[0].detach().numpy()
        #     plt.scatter(tmp_predicted[:,0],tmp_predicted[:,1],label="predicted")
        #     plt.scatter(ground_truth_landmarks[:,0], ground_truth_landmarks[:,1], label="ground truth")
        #     plt.legend()
        #     plt.show()

        loss_lan = torch.mean(torch.linalg.norm(predicted - torch.from_numpy(ground_truth_landmarks))**2)

        lambda_alpha = 1
        lambda_delta = 1
        loss_reg = lambda_alpha * torch.sum(alpha**2) + lambda_delta * torch.sum(delta**2)

        loss_fit = loss_lan + loss_reg
        print(loss_fit.item())

        loss_fit.backward()

        alpha.register_hook(lambda grad: torch.clamp(grad, -1, 1))
        delta.register_hook(lambda grad: torch.clamp(grad, -1, 1))

        optim.step()

        new_loss = loss_fit.item()
        i = i+1
        # break

        # if (abs(old_loss - new_loss) < epsilon):
        #     tmp_predicted = predicted.detach().numpy()
        #     # gt = ground_truths[0].detach().numpy()
        #     tmp_predicted[:,1] = -tmp_predicted[:,1]
        #     plt.scatter(tmp_predicted[:,0],tmp_predicted[:,1],label="predicted")
        #     ground_truth_landmarks[:,1] = -ground_truth_landmarks[:,1]
        #     plt.scatter(ground_truth_landmarks[:,0], ground_truth_landmarks[:,1], label="ground truth")
        #     plt.legend()
        #     plt.show()

    G_def = torch.Tensor(id_mean) + torch.Tensor(id_pcabasis30) @ (alpha * torch.sqrt(torch.Tensor(id_pcavariance30))) + torch.Tensor(expr_mean) + torch.Tensor(expr_pcabasis20) @ (delta * torch.sqrt(torch.Tensor(expr_pcavariance20)))
    G_new_def = (general_rotation(omega) @ G_def.T).T + t
    save_obj(f"{file_name}.obj",G_new_def.detach().numpy(),color,triangles)

    return G_new_def, alpha, delta, omega, t

def optimized_geometry_multiple(gt_landmarks_list, names_list):
    indices_model = np.loadtxt("Landmarks68_model2017-1_face12_nomouth.anl")
    alpha = [torch.autograd.Variable(torch.randn(30,), requires_grad=True)]
    deltas = []
    omegas = []
    ts = []
    for _ in gt_landmarks_list:
        deltas.append(torch.autograd.Variable(torch.randn(20,), requires_grad=True))
        omegas.append(torch.autograd.Variable(torch.Tensor([0,0,0]),requires_grad=True))
        ts.append(torch.autograd.Variable(torch.Tensor([0,0,-200]), requires_grad=True))
    optim = torch.optim.Adam(alpha + deltas + omegas + ts, lr=0.05)

    old_loss = np.inf
    new_loss = 10000000

    epsilon = 0.001
    j = 0
    while (abs(old_loss - new_loss) > epsilon):
    # while (j < 25):
        print("prediction #",j)

        optim.zero_grad()
        old_loss = new_loss

        losses = torch.zeros(len(gt_landmarks_list))
        for i in range(len(gt_landmarks_list)):
            G = torch.Tensor(id_mean) + torch.Tensor(id_pcabasis30) @ (alpha[0] * torch.sqrt(torch.Tensor(id_pcavariance30))) + torch.Tensor(expr_mean) + torch.Tensor(expr_pcabasis20) @ (deltas[i] * torch.sqrt(torch.Tensor(expr_pcavariance20)))

            G_lm = G[indices_model,:]

            G_new = (general_rotation(omegas[i]) @ G_lm.T).T + ts[i]
            predicted = convertTo2D(G_new)

            loss_lan = torch.mean(torch.linalg.norm(predicted - torch.from_numpy(gt_landmarks_list[i]))**2)

            lambda_alpha = 1
            lambda_delta = 1
            loss_reg = lambda_alpha * torch.sum(alpha[0]**2) + lambda_delta * torch.sum(deltas[i]**2)

            loss_fit = loss_lan + loss_reg

            losses[i] = loss_fit

        loss_combi = torch.mean(torch.Tensor(losses))
        print(loss_combi.item())
        loss_combi.backward()

        optim.step()

        new_loss = loss_combi.item()
        j = j+1

    Gs = []
    for i in range(len(gt_landmarks_list)):
        G_def = torch.Tensor(id_mean) + torch.Tensor(id_pcabasis30) @ (alpha[0] * torch.sqrt(torch.Tensor(id_pcavariance30))) + torch.Tensor(expr_mean) + torch.Tensor(expr_pcabasis20) @ (deltas[i] * torch.sqrt(torch.Tensor(expr_pcavariance20)))
        G_new_def = (general_rotation(omegas[i]) @ G_def.T).T + ts[i]
        save_obj(names_list[i], G_new_def.detach().numpy(), color, triangles)
        Gs.append(G_new_def)

    return Gs, alpha, deltas, omegas, ts

def optimized_geometry_video(video_path):
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

    frames = []
    for no in range(count):
        if no % 10 == 0:
            frame = cv2.imread("frame%d.jpg" % no)
            frame = frame[100:550, 50:500]
            frame = cv2.resize(frame, (256,256), interpolation = cv2.INTER_AREA)

            # plt.imshow(frame)
            # plt.show()
            frame = torch.from_numpy(frame)
            frames.append(frame)
    print("len frames: ", len(frames))

    ground_truths = []
    for frame in frames:
        ground_truths.append(find_ground_truth_landmarks(frame))

    indices_model = np.loadtxt("Landmarks68_model2017-1_face12_nomouth.anl")
    alpha = [torch.autograd.Variable(torch.randn(30,), requires_grad=True)]
    deltas = []
    omegas = []
    ts = []
    for no in range(len(frames)):
        deltas.append(torch.autograd.Variable(torch.randn(20,), requires_grad=True))
        omegas.append(torch.autograd.Variable(torch.Tensor([0,0,0]),requires_grad=True))
        ts.append(torch.autograd.Variable(torch.Tensor([0,0,-200]), requires_grad=True))
    optim = torch.optim.Adam(alpha + deltas + omegas + ts, lr=0.05)

    old_loss = np.inf
    new_loss = 10000000

    epsilon = 0.01
    j = 0
    while (abs(old_loss - new_loss) > epsilon):
    # while (j < 25):
        print("prediction #",j)

        optim.zero_grad()
        old_loss = new_loss

        losses = torch.zeros(len(frames))
        for i in range(len(frames)):
            G = torch.Tensor(id_mean) + torch.Tensor(id_pcabasis30) @ (alpha[0] * torch.sqrt(torch.Tensor(id_pcavariance30))) + torch.Tensor(expr_mean) + torch.Tensor(expr_pcabasis20) @ (deltas[i] * torch.sqrt(torch.Tensor(expr_pcavariance20)))

            G_lm = G[indices_model,:]

            G_new = (general_rotation(omegas[i]) @ G_lm.T).T + ts[i]
            predicted = convertTo2D(G_new)

            loss_lan = torch.mean(torch.linalg.norm(predicted - torch.from_numpy(ground_truths[i]))**2)

            lambda_alpha = 1
            lambda_delta = 1
            loss_reg = lambda_alpha * torch.sum(alpha[0]**2) + lambda_delta * torch.sum(deltas[i]**2)

            loss_fit = loss_lan + loss_reg

            losses[i] = loss_fit

        loss_combi = torch.mean(torch.Tensor(losses))
        print(loss_combi.item())
        loss_combi.backward()

        optim.step()

        new_loss = loss_combi.item()
        j = j+1

    Gs = []
    for i in range(len(frames)):
        G_def = torch.Tensor(id_mean) + torch.Tensor(id_pcabasis30) @ (alpha[0] * torch.sqrt(torch.Tensor(id_pcavariance30))) + torch.Tensor(expr_mean) + torch.Tensor(expr_pcabasis20) @ (deltas[i] * torch.sqrt(torch.Tensor(expr_pcavariance20)))
        G_new_def = (general_rotation(omegas[i]) @ G_def.T).T + ts[i]
        Gs.append(G_new_def)

    return frames, Gs, alpha, deltas, omegas, ts

def face_swapping_video(bg_video_path, fg_image_path):
    bg_frames, bg_Gs, bg_alpha, bg_deltas, bg_omegas, bg_ts = optimized_geometry_video(bg_video_path)
    print("optimized_geometry_video done")

    fg_img = cv2.imread(fg_image_path)
    fg_img = cv2.resize(fg_img, (256,256), interpolation = cv2.INTER_AREA)
    fg_img  = torch.from_numpy(fg_img)
    fg_ground_truth = find_ground_truth_landmarks(fg_img)

    fg_G, fg_alpha, fg_delta, fg_omega, fg_t = optimize_geometry(fg_ground_truth, "fg_image")
    print("optimize_geometry done")

    all_images = []
    counting = 0
    print("Nu nog")
    print(len(bg_frames))
    print("loops")
    for bg_frame, bg_omega, bg_t in zip(bg_frames, bg_omegas, bg_ts):
        print(counting)
        fg_adopted = (general_rotation(bg_omega) @ fg_G.T).T + bg_t
        print("fg_adopted shape", fg_adopted.shape)
        bg_landmarks = predict_landmarks(bg_frame)
        # render an image of the new face, using the position of the face in the original image
        mask = np.full(bg_frame.shape, 255, dtype=np.uint8)
        print("mask shape ", mask.shape)
        forehead = bg_landmarks[17:27]
        forehead[:,1] += -35
        roi_corners = np.array([np.concatenate((bg_landmarks[:17], # chin
                                                forehead[::-1], # eyebrows
                                                ), axis=0)], dtype=np.int32)
        channel_count = bg_frame.shape[2]
        ignore_mask_color = (0,)*channel_count
        cv2.fillPoly(mask, roi_corners, ignore_mask_color)

        fg_adopted_img = texturing_for_faceswap(fg_adopted.detach().numpy(), color, triangles, bg_frame.shape, mask)
        print("fg_adopted_img shape", fg_adopted_img.shape)
        # plt.imshow(fg_adopted_img)
        # plt.show()
        # first version
        # swapped = np.where(mask, (bg_frame.detach().numpy()/255)[..., ::-1], fg_adopted_img)
        # plt.imshow(swapped)
        # plt.axis("off")
        # plt.show()

        # second version swapped
        new_mask = np.tile(np.any(fg_adopted_img > 0, axis=2, keepdims=True), (1, 1, 3))
        print("new_mask shape ", new_mask.shape)
        new_bg = np.copy(bg_frame)
        print("new_bg shape", new_bg.shape)
        new_bg[new_mask] = 0
        new_output = np.where(new_mask, fg_adopted_img, (new_bg/255)[..., ::-1])

        plt.imshow(new_output)
        plt.axis("off")
        plt.savefig("faceswap%d.jpg" % counting)
        # plt.show()

        # im = Image.fromarray(new_output)
        # im.save("faceswap%d.jpg" % counting)
        # cv2.imwrite("faceswap%d.jpg" % counting, new_output)  # save frame as JPEG file

        all_images.append(new_output)
        counting += 1

    out = cv2.VideoWriter('TEST.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (bg_frame.shape[0],bg_frame.shape[1]))

    for i in range(len(all_images)):
        out.write(all_images[i])
    out.release()

    print("KLAAR?")

def face_swapping(fg_path, bg_path):
    fg = cv2.imread(fg_path)
    fg = cv2.resize(fg, (256,256), interpolation = cv2.INTER_AREA)
    bg = cv2.imread(bg_path)
    bg = cv2.resize(bg, (256,256), interpolation = cv2.INTER_AREA)

    print(fg.shape)
    print(bg.shape)

    fg  = torch.from_numpy(fg)
    bg  = torch.from_numpy(bg)

    fg_ground_truth = find_ground_truth_landmarks(fg)
    bg_ground_truth = find_ground_truth_landmarks(bg)

    fg_G, fg_alpha, fg_delta, fg_omega, fg_t = optimize_geometry(fg_ground_truth, "fg")
    bg_G, bg_alpha, bg_delta, bg_omega, bg_t = optimize_geometry(bg_ground_truth, "bg")

    fg_adopted = (general_rotation(bg_omega) @ fg_G.T).T + bg_t

    bg_landmarks = predict_landmarks(bg)
    print("bg landmarks", bg_landmarks)
    # render an image of the new face, using the position of the face in the original image
    mask = np.full(bg.shape, 255, dtype=np.uint8)
    forehead = bg_landmarks[17:27]
    forehead[:,1] += -35
    roi_corners = np.array([np.concatenate((bg_landmarks[:17], # chin
                                            forehead[::-1], # eyebrows
                                            ), axis=0)], dtype=np.int32)
    channel_count = bg.shape[2]
    ignore_mask_color = (0,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    fg_adopted_img = texturing_for_faceswap(fg_adopted.detach().numpy(), color, triangles, bg.shape, mask)

    # first version
    # swapped = np.where(mask, (bg.detach().numpy()/255)[..., ::-1], fg_adopted_img)
    # plt.imshow(swapped)
    # plt.axis("off")
    # plt.show()

    # second version swapped
    new_mask = np.tile(np.any(fg_adopted_img > 0, axis=2, keepdims=True), (1, 1, 3))
    new_bg = np.copy(bg)
    new_bg[new_mask] = 0
    new_output = np.where(new_mask, fg_adopted_img, (new_bg/255)[..., ::-1])
    plt.imshow(new_output)
    plt.axis("off")
    plt.show()

def train_model(img_path, name):
    image = cv2.imread(img_path)
    image = torch.from_numpy(image)

    gt_landmarks = find_ground_truth_landmarks(image)
    G, alpha, delta, omega, t = optimize_geometry(gt_landmarks, name)

def train_model_multiple(img_path_list, names_list):
    gt_landmarks_list = []
    for img_path in img_path_list:
        image = cv2.imread(img_path)
        image = torch.from_numpy(image)
        gt_landmarks = find_ground_truth_landmarks(image)
        gt_landmarks_list.append(gt_landmarks)

    Gs, alpha, deltas, omegas, ts = optimized_geometry_multiple(gt_landmarks_list, names_list)
    print(Gs)
    print(alpha)
    print(deltas)
    print(omegas)

train_model_multiple(["img1.jpg", "img2.jpg", "img3.jpg"], ["multiple1.obj", "multiple2.obj", "multiple3.obj"])
# train_model("person.jpeg", "person_model")
# train_model("person2.jpeg", "person2_model")
# face_swapping("person.jpeg", "person2.jpeg")
# face_swapping("../images/cropped_fg.png", "../images/cropped_bg.png")
# face_swapping_video("../emotions.mp4", "person2.jpeg")