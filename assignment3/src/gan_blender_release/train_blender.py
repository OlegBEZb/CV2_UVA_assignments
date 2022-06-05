import os
import time
import math

import torch
import torch.nn.parallel
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
import imageio

import cv2

import utils
import img_utils

import dlib
import openface

# Configurations
######################################################################
# Fill in your experiment names and the other required components
experiment_name = 'Blender'
data_root = '/content/gdrive/MyDrive/CV_2/data_set/data'
val_data_root = '/content/gdrive/MyDrive/CV_2/data_set/val_data'
# data_root = '../../data_set/data_set/data'
pre_trained_models_path = '/content/gdrive/MyDrive/CV_2/data_set/Pretrained_model'
# pre_trained_models_path = '../../data_set/Pretrained_model'
train_list = ''
test_list = ''
batch_size = 8
nthreads = 2
max_epochs = 20 
displayIter = 10
saveIter = 1
img_resolution = 256

lr_gen = 1e-4
lr_dis = 1e-4

momentum = 0.9
weightDecay = 1e-4
step_size = 30
gamma = 0.1

pix_weight = 0.1
rec_weight = 1.0
gan_weight = 0.001
######################################################################
# Independent code. Don't change after this line. All values are automatically
# handled based on the configuration part.

if batch_size < nthreads:
    nthreads = batch_size
check_point_loc = 'Exp_%s/checkpoints/' % experiment_name.replace(' ', '_')
visuals_loc = 'Exp_%s/visuals/' % experiment_name.replace(' ', '_')
os.makedirs(check_point_loc, exist_ok=True)
os.makedirs(visuals_loc, exist_ok=True)
checkpoint_pattern = check_point_loc + 'checkpoint_%s_%d.pth'
logTrain = check_point_loc + 'LogTrain.txt'

torch.backends.cudnn.benchmark = True

cudaDevice = ''

if len(cudaDevice) < 1:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('[*] GPU Device selected as default execution device.')
    else:
        device = torch.device('cpu')
        print('[X] WARN: No GPU Devices found on the system! Using the CPU. '
              'Execution maybe slow!')
else:
    device = torch.device('cuda:%s' % cudaDevice)
    print('[*] GPU Device %s selected as default execution device.' %
          cudaDevice)

done = u'\u2713'
print('[I] STATUS: Initiate Network and transfer to device...', end='')
# Define your generators and Discriminators here
from discriminators_pix2pix import MultiscaleDiscriminator
from res_unet import MultiScaleResUNet
from utils import loadModels

discriminator = MultiscaleDiscriminator().to(device)
generator = MultiScaleResUNet(in_nc=7, out_nc=3).to(device)
print('\ngenerator, discriminator devices are GPU', next(generator.parameters()).is_cuda, next(discriminator.parameters()).is_cuda)
print(done)

print('[I] STATUS: Load Networks...', end='')
# Load your pretrained models here. Pytorch requires you to define the model
# before loading the weights, since the weight files does not contain the model
# definition. Make sure you transfer them to the proper training device. Hint:
# use the .to(device) function, where device is automatically detected
# above.

discriminator, discrim_optim_state, checkpoint_iters_d = loadModels(discriminator,
                                                                    os.path.join(pre_trained_models_path,
                                                                                 'checkpoint_D.pth'),
                                                                    device=device)
generator, gen_optim_state, checkpoint_iters_g = loadModels(generator,
                                                            os.path.join(pre_trained_models_path, 'checkpoint_G.pth'),
                                                            device=device)
print('\ngenerator, discriminator devices are GPU', next(generator.parameters()).is_cuda, next(discriminator.parameters()).is_cuda)

print(done)

print('[I] STATUS: Initiate optimizer...', end='')
# Define your optimizers and the schedulers and connect the networks from
# before
optimizer_G = torch.optim.SGD(generator.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.1)

optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.1)
if gen_optim_state is not None:
    optimizer_G.load_state_dict(gen_optim_state)
    optimizer_G_state = None
if discrim_optim_state is not None:
    optimizer_D.load_state_dict(discrim_optim_state)
    optimizer_D_state = None
print(done)

print('[I] STATUS: Initiate Criterions and transfer to device...', end='')
# Define your criterions here and transfer to the training device. They need to
# be on the same device type.
from gan_loss import GANLoss

criterion_gan = GANLoss(use_lsgan=True).to(device)

criterion_pixelwise = torch.nn.L1Loss().to(device)

from vgg_loss import VGGLoss

criterion_id = VGGLoss().to(device)

criterion_attr = VGGLoss().to(device)
print(done)

print('[I] STATUS: Initiate Dataloaders...')
# Initialize your datasets here

# from ..datasets.img_landmarks_transforms import ToTensor, Compose
# from torchvision import transforms
# numpy_transforms = None
# tensor_transforms = (ToTensor(), transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])),  # applied during loading
# img_transforms = Compose(numpy_transforms + tensor_transforms)

from SwappedDataset import SwappedDatasetLoader

train_loader = SwappedDatasetLoader(data_root, transform=None, use_first_n=100)  # global variables is all you need
val_loader = SwappedDatasetLoader(val_data_root, transform=None, use_first_n=100)  # global variables is all you need

from torch.utils.data import DataLoader
train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True, num_workers=nthreads,
                          pin_memory=True, drop_last=True)
val_loader = DataLoader(val_loader, batch_size=batch_size, shuffle=True, num_workers=nthreads,
                        pin_memory=True, drop_last=True)
print(done)

print('[I] STATUS: Initiate Logs...', end='')
trainLogger = open(logTrain, 'w')
print(done)


def transfer_mask(img1, img2, mask):
    # face reenacted to the background face without face
    return img1 * mask + img2 * (1 - mask)


def blend_imgs_bgr(source_img, target_img, mask):
    # Implement poisson blending here. You can use the built-in seamlessClone
    # function in opencv which is an implementation of Poisson Blending.
    a = np.where(mask != 0)
    if len(a[0]) == 0 or len(a[1]) == 0:
        return target_img
    if (np.max(a[0]) - np.min(a[0])) <= 10 or (np.max(a[1]) - np.min(a[1])) <= 10:
        return target_img

    center = (np.min(a[1]) + np.max(a[1])) // 2, (np.min(a[0]) + np.max(a[0])) // 2
    output = cv2.seamlessClone(source_img, target_img, mask * 255, center, cv2.NORMAL_CLONE)

    return output


def blend_imgs(source_tensor: torch.Tensor, target_tensor: torch.Tensor, mask_tensor: torch.Tensor, blend='alpha', alpha=0.5):
    assert blend in ['alpha', 'poisson'], "'alpha' and 'poisson' blending methods are implemented at the moment"
    out_tensors = []
    for b in range(source_tensor.shape[0]):
        source_img = img_utils.tensor2bgr(source_tensor[b])
        target_img = img_utils.tensor2bgr(target_tensor[b])
        if blend == 'alpha':
            out_bgr = cv2.addWeighted(src1=source_img, alpha=alpha, src2=target_img, beta=1-alpha, gamma=0)
        else:
            mask = mask_tensor[b].squeeze().permute(1, 2, 0).cpu().numpy()
            mask = np.round(mask * 255).astype('uint8')
            out_bgr = blend_imgs_bgr(source_img, target_img, mask)
        out_tensors.append(img_utils.bgr2tensor(out_bgr))

    return torch.cat(out_tensors, dim=0)


def Train(G: torch.nn.Module, D: torch.nn.Module, epoch_count, iter_count, **blend_kwargs):
    G.train(True)
    D.train(True)
    epoch_count += 1
    batches_train = math.floor(len(train_loader) / train_loader.batch_size)
    pbar = tqdm(enumerate(train_loader), total=batches_train, leave=False)

    total_loss_pix = 0
    total_loss_id = 0
    total_loss_attr = 0
    total_loss_rec = 0
    total_loss_G_Gan = 0
    total_loss_D_Gan = 0

    Epoch_time = time.time()

    for i, data in pbar:
        iter_count += 1
        # images, _ = data

        # Implement your training loop here. images will be the datastructure
        # being returned from your dataloader.
        # 1) Load and transfer data to device
        source, target, swap, mask = data['source'].squeeze(), data['target'].squeeze(), data['swap'].squeeze(), data['mask'].squeeze()
        # print('source before device', source.get_device())
        source, target, swap, mask = source.to(device), target.to(device), swap.to(device), mask.to(device)
        # print('source shape, type, device after device', source.shape, source.type(), source.get_device())
        img_transfer = transfer_mask(source, target, mask)
        # print('img_transfer.shape', img_transfer.shape, img_transfer.type())


        img_blend = blend_imgs(swap, target, mask, **blend_kwargs)
        # print('img_blend shape, device', img_blend.shape, img_blend.get_device())
        img_blend = img_blend.to(device)
        # print('img_blend device', img_blend.get_device())

        # print('before concat', img_transfer.shape, img_transfer.type(), target.shape, target.type(), mask[:, :1, :, :].shape, mask[:, :1, :, :].type())
        img_transfer_input = torch.cat((img_transfer, target, mask[:, :1, :, :]), dim=1)
        # print('img_transfer_input.shape', img_transfer_input.shape, img_transfer_input.get_device())

        img_transfer_input_pyd = img_utils.create_pyramid(img_transfer_input.to(device), 1)  # len(source[0])
        # print('len(img_transfer_input_pyd)', len(img_transfer_input_pyd), img_transfer_input_pyd[0].shape, img_transfer_input_pyd[0].get_device())

        # 2) Feed the data to the networks.
        # Blend images
        img_blend_pred = G([p.to(device) for p in img_transfer_input_pyd])
        # print('img_blend_pred shape, device', img_blend_pred.shape, img_blend_pred.get_device())


        # Fake Detection and Loss
        img_blend_pred_pyd = img_utils.create_pyramid(img_blend_pred, len(source[0]))
        pred_fake_pool = D([x.detach() for x in img_blend_pred_pyd])
        loss_D_fake = criterion_gan(pred_fake_pool, False)

        # Real Detection and Loss
        pred_real = D(target)
        loss_D_real = criterion_gan(pred_real, True)

        # 4) Calculate the losses.
        loss_D_total = (loss_D_fake + loss_D_real) * 0.5

        # GAN loss (Fake Passability Loss)
        pred_fake = D(img_blend_pred_pyd)
        loss_G_GAN = criterion_gan(pred_fake, True)

        # Reconstruction
        loss_pixelwise = criterion_pixelwise(img_blend_pred, img_blend)
        loss_id = criterion_id(img_blend_pred, img_blend)
        loss_attr = criterion_attr(img_blend_pred, img_blend)
        loss_rec = pix_weight * loss_pixelwise + 0.5 * loss_id + 0.5 * loss_attr


        loss_G_total = rec_weight * loss_rec + gan_weight * loss_G_GAN    # Oleg's loss
        # loss_G_total = - torch.clone(loss_D_total)
        # img_blend_pred = G([p.to(device) for p in img_transfer_input_pyd])
        # img_blend_pred_pyd = img_utils.create_pyramid(img_blend_pred, len(source[0]))
        # pred_fake_pool = D([x.detach() for x in img_blend_pred_pyd])
        # pred_fake_pool2 = pred_fake_pool.copy()
        # loss_G_total = criterion_gan(pred_fake_pool2, True)               # too large

        # total_loss_pix += loss_pixelwise
        # total_loss_id += loss_id
        # total_loss_attr += loss_attr
        # total_loss_rec += loss_rec
        # total_loss_G_Gan += loss_G_GAN
        # total_loss_D_Gan += loss_D_total

        # 5) Perform backward calculation.
        # 6) Perform the optimizer step.
        # Update generator weights
        optimizer_G.zero_grad()
        loss_G_total.backward()
        optimizer_G.step()

        # Update discriminator weights
        # optimizer_D.zero_grad()
        # loss_D_total.backward()
        # optimizer_D.step()

        if iter_count % displayIter == 0:
            # Write to the log file.
            trainLogger.write(f'Epoch: {epoch_count} / {max_epochs}\n'
                              f'pixelwise={loss_pixelwise:0.4f}, id={loss_id:0.4f}, attr={loss_attr:0.4f}, rec={loss_rec:0.4f}, '
                              f'g_gan={loss_G_GAN:0.4f}, d_gan={loss_D_total:0.4f}')

        # Print out the losses here. Tqdm uses that to automatically print it
        # in front of the progress bar.
        print(f'Epoch: {epoch_count} / {max_epochs}\n'
              f'pixelwise={loss_pixelwise:0.4f}, id={loss_id:0.4f}, attr={loss_attr:0.4f}, rec={loss_rec:0.4f}, '
              f'g_gan={loss_G_GAN:0.4f}, d_gan={loss_D_total:0.4f}')
        pbar.set_description()

    # Save output of the network at the end of each epoch. The Generator
    swap_type="naive"

    t_source, t_swap, t_target, t_pred, t_blend = Test(G, type=swap_type, **blend_kwargs)
    for b in range(t_pred.shape[0]):
        total_grid_load = [t_source[b], t_swap[b], t_target[b],
                           t_pred[b], t_blend[b]]
        grid = img_utils.make_grid(total_grid_load,
                                   cols=len(total_grid_load))
        grid = img_utils.tensor2rgb(grid.detach())[..., ::-1]
        imageio.imwrite(visuals_loc + '/%s_Epoch_%d_output_%d.png' %
                        (swap_type, epoch_count, b), grid)

    utils.saveModels(G, optimizer_G, iter_count,
                     checkpoint_pattern % ('G', epoch_count))
    utils.saveModels(D, optimizer_D, iter_count,
                     checkpoint_pattern % ('D', epoch_count))
    tqdm.write('[!] Model Saved!')

    # return np.nanmean(total_loss_pix.detach().numpy().cpu()), \
    #        np.nanmean(total_loss_id.detach().numpy().cpu()), np.nanmean(total_loss_attr.detach().numpy().cpu()), \
    #        np.nanmean(total_loss_rec.detach().numpy().cpu()), np.nanmean(total_loss_G_Gan.detach().numpy().cpu()), \
    #        np.nanmean(total_loss_D_Gan.detach().numpy().cpu()), iter_count


def Test(G, type='normal', **blend_kwargs):
    with torch.no_grad():
        G.eval()


        t = enumerate(val_loader)
        i, images = next(t)

        # Feed the network with images from test set
        source, target, swap, mask = images['source'].squeeze(), images['target'].squeeze(), images['swap'].squeeze(), images['mask'].squeeze()

        if type == "normal":
            t = enumerate(val_loader)
            i, images = next(t)

            # Feed the network with images from test set
            source, target, swap, mask = images['source'].squeeze(), images['target'].squeeze(), images['swap'].squeeze(), images['mask'].squeeze()

        if type == "naive":
            # find mask
            # fg_lm = predict_landmarks(source).astype(np.int32)

            # mask = np.zeros(source.shape, dtype=np.uint8)
            # roi_corners = np.array([np.concatenate((fg_lm[:17], # chin
            #                                         fg_lm[17:27][::-1], # eyebrows
            #                                         ), axis=0)], dtype=np.int32)
            # channel_count = source.shape[2]
            # ignore_mask_color = (255,)*channel_count
            # cv2.fillPoly(mask, roi_corners, ignore_mask_color)

            # # find swap
            # swap = np.where(mask, target, 0)
            combinations = [("0000","0","9999"), ("0001","1","9998"), ("0002","10","9997")]
            source = []
            target = []
            swap = []
            mask = []
            for combi in combinations:
                # source_img = np.swapaxes(cv2.imread(f"/content/gdrive/MyDrive/CV_2/images_Emily/{combi[0]}_bg_{combi[2]}.png"),0,-1)
                # print(source_img.shape)
                # source.append(np.swapaxes(cv2.imread(f"/content/gdrive/MyDrive/CV_2/images_Emily/{combi[0]}_bg_{combi[2]}.png"),0,-1))
                # target.append(np.swapaxes(cv2.imread(f"/content/gdrive/MyDrive/CV_2/images_Emily/{combi[0]}_fg_{combi[1]}.png"),0,-1))
                # swap.append(np.swapaxes(cv2.resize(cv2.imread(f"/content/gdrive/MyDrive/CV_2/images_Emily/{combi[0]}_sw_{combi[2]}_{combi[1]}.png"),(224,224),interpolation = cv2.INTER_AREA),0,-1))
                # mask.append(np.swapaxes(cv2.resize(cv2.imread(f"/content/gdrive/MyDrive/CV_2/images_Emily/{combi[0]}_mask_{combi[2]}_{combi[1]}.png"),(224,224),interpolation = cv2.INTER_AREA),0,-1))

                source_img = torch.permute(torch.from_numpy(cv2.imread(f"/content/gdrive/MyDrive/CV_2/images_Emily/{combi[0]}_bg_{combi[2]}.png")),(2,1,0))
                print(source_img.shape)
                source.append(torch.permute(torch.from_numpy(cv2.imread(f"/content/gdrive/MyDrive/CV_2/images_Emily/{combi[0]}_bg_{combi[2]}.png")),(2,1,0)))
                target.append(torch.permute(torch.from_numpy(cv2.imread(f"/content/gdrive/MyDrive/CV_2/images_Emily/{combi[0]}_fg_{combi[1]}.png")),(2,1,0)))
                swap.append(torch.permute(torch.from_numpy(cv2.resize(cv2.imread(f"/content/gdrive/MyDrive/CV_2/images_Emily/{combi[0]}_sw_{combi[2]}_{combi[1]}.png"),(224,224),interpolation = cv2.INTER_AREA)),(2,1,0)))
                mask.append(torch.permute(torch.from_numpy(cv2.resize(cv2.imread(f"/content/gdrive/MyDrive/CV_2/images_Emily/{combi[0]}_mask_{combi[2]}_{combi[1]}.png"),(224,224),interpolation = cv2.INTER_AREA)),(2,1,0)))

            source, target, swap, mask = torch.tensor(source), torch.tensor(target), torch.tensor(swap), torch.tensor(mask)


        elif type == "dl":
            from face_swap import FaceSwap
            fs = FaceSwap()
            swap = fs.swap(target, source)
            mask = np.tile(np.any(swap > 0, axis=2, keepdims=True), (1, 1, 3))

        elif type == "opt":
            mask, swap, _ = face_swapping(target, source)

        source, target, swap, mask = source.to(device), target.to(device), swap.to(device), mask.to(device)
        img_transfer = transfer_mask(source, target, mask)

        img_blend = blend_imgs(swap, target, mask, **blend_kwargs)
        img_transfer_input = torch.cat((img_transfer, target, mask[:, :1, :, :]), dim=1).to(device)

        # Blend images
        pred = G(img_transfer_input)
        # You want to return 4 components:
        # 1) The source face.
        # 2) The 3D reconsturction. ?
        # 3) The target face.
        # 4) The prediction from the generator.
        # 5) The GT Blend that the network is targeting.
        return source, swap, target, pred, img_blend




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


# bfm = h5py.File("/content/gdrive/MyDrive/CV_2/model2017-1_face12_nomouth.h5", 'r')

# # Select a specific weight from BFM
# id_mean = np.asarray(bfm['shape/model/mean'], dtype=np.float32) # 3N
# # Sometimes you will need to reshape it to a proper shape
# id_mean = np.reshape(id_mean, (-1,3))

# id_pcabasis = np.asarray(bfm['shape/model/pcaBasis'], dtype=np.float32) # 3N x 199
# id_pcabasis = np.reshape(id_pcabasis, (-1,3,199))
# id_pcabasis30 = id_pcabasis[:,:,:30]

# id_pcavariance = np.asarray(bfm['shape/model/pcaVariance'], dtype=np.float32) #199
# id_pcavariance30 = id_pcavariance[:30]

# expr_mean = np.asarray(bfm['expression/model/mean'], dtype=np.float32) # 3N
# expr_mean = np.reshape(expr_mean, (-1,3))

# expr_pcabasis = np.asarray(bfm['expression/model/pcaBasis'], dtype=np.float32) # 3N x 100
# expr_pcabasis = np.reshape(expr_pcabasis, (-1,3,100))
# expr_pcabasis20 = expr_pcabasis[:,:,:20]

# expr_pcavariance = np.asarray(bfm['expression/model/pcaVariance'], dtype=np.float32) # 100
# expr_pcavariance20 = expr_pcavariance[:20]

# alpha = np.random.uniform(-1,1,30)
# delta = np.random.uniform(-1,1,20)

# color = np.asarray(bfm['color/model/mean'], dtype=np.float32)
# color = np.reshape(color, (-1,3))

# triangles = np.asarray(bfm['shape/representer/cells'], dtype=np.float32).T # 3xK

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

# Face Detector
detector = dlib.get_frontal_face_detector()
# Landmark Detector
predictor = openface.AlignDlib("shape_predictor_68_face_landmarks.dat")

def predict_landmarks(img):
  img = img.detach().numpy()
  dets = detector(img/255, 1)
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


def face_swapping(fg, bg):
    # fg = cv2.imread(fg_path)
    fg = cv2.resize(fg, (256,256), interpolation = cv2.INTER_AREA)
    # bg = cv2.imread(bg_path)
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
    return new_mask, fg_adopted_img, new_output

def main():
    # Print out the experiment configurations. You can also save these to a file if
    # you want them to be persistent.
    print('[*] Beginning Training:')
    print('\tMax Epoch: ', max_epochs)
    print('\tLogging iter: ', displayIter)
    print('\tSaving frequency (per epoch): ', saveIter)
    print('\tModels Dumped at: ', check_point_loc)
    print('\tVisuals Dumped at: ', visuals_loc)
    print('\tExperiment Name: ', experiment_name)

    for i in range(max_epochs):
        # Call the Train function here
        # Step through the schedulers if using them.
        # You can also print out the losses of the network here to keep track of
        # epoch wise loss.
        Train(G=generator, D=discriminator, epoch_count=i,
                           iter_count=iter_count, blend='alpha')  # love passing global variables

        # # Schedulers step (in PyTorch 1.1.0+ it must follow after the epoch training and validation steps)
        # if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        #     scheduler_G.step(total_loss)
        #     scheduler_D.step(total_loss)
        # else:
        #     scheduler_G.step()
        #     scheduler_D.step()
        #
        # # Save models checkpoints
        # is_best = total_loss < best_loss
        # best_loss = min(best_loss, total_loss)
        # utils.save_checkpoint(exp_dir, 'Gb', {
        #     'resolution': res,
        #     'epoch': epoch + 1,
        #     'state_dict': Gb.module.state_dict() if gpus and len(gpus) > 1 else Gb.state_dict(),
        #     'optimizer': optimizer_G.state_dict(),
        #     'best_loss': best_loss,
        # }, is_best)
        # utils.save_checkpoint(exp_dir, 'D', {
        #     'resolution': res,
        #     'epoch': epoch + 1,
        #     'state_dict': D.module.state_dict() if gpus and len(gpus) > 1 else D.state_dict(),
        #     'optimizer': optimizer_D.state_dict(),
        #     'best_loss': best_loss,
        # }, is_best)

    trainLogger.close()


if __name__ == '__main__':
    iter_count = 0
    epoch_count = 0
    main()


