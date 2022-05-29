import os
import time
import torch
import torch.nn.parallel
import numpy as np
from tqdm import tqdm
import imageio

import cv2

import utils
import img_utils

# Configurations
######################################################################
# Fill in your experiment names and the other required components
experiment_name = 'Blender'
data_root = '/content/gdrive/MyDrive/CV_2/data_set/data'
train_list = ''
test_list = ''
batch_size = 8
nthreads = 1
max_epochs = 20
displayIter = 20
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

# pre_trained_models_path = '/content/gdrive/MyDrive/CV_2/data_set/Pretrained_model'
pre_trained_models_path = '../../data_set/Pretrained_model'

discriminator = MultiscaleDiscriminator().to(device)
generator = MultiScaleResUNet(in_nc=7, out_nc=3).to(device)
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

criterion_pixelwise = torch.nn.L1Loss()

from vgg_loss import VGGLoss

criterion_id = VGGLoss()

criterion_attr = VGGLoss()
print(done)

print('[I] STATUS: Initiate Dataloaders...')
# Initialize your datasets here

# from ..datasets.img_landmarks_transforms import ToTensor, Compose
# from torchvision import transforms
# numpy_transforms = None
# tensor_transforms = (ToTensor(), transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])),  # applied during loading
# img_transforms = Compose(numpy_transforms + tensor_transforms)

from SwappedDataset import SwappedDatasetLoader

data_root = '../../data_set/data_set/data'
train_loader = SwappedDatasetLoader(data_root, transform=None)  # global variables is all you need

from torch.utils.data import DataLoader
train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True, num_workers=nthreads,
                          pin_memory=True, drop_last=True)
# if val_dataset is not None:
#     val_dataset = VideoSeqDataset(transform=img_transforms)
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


def blend_imgs(source_tensor: torch.Tensor, target_tensor: torch.Tensor, mask_tensor: torch.Tensor):
    out_tensors = []
    for b in range(source_tensor.shape[0]):
        source_img = img_utils.tensor2bgr(source_tensor[b])
        target_img = img_utils.tensor2bgr(target_tensor[b])
        mask = mask_tensor[b].squeeze().permute(1, 2, 0).cpu().numpy()
        mask = np.round(mask * 255).astype('uint8')
        out_bgr = blend_imgs_bgr(source_img, target_img, mask)
        out_tensors.append(img_utils.bgr2tensor(out_bgr))

    return torch.cat(out_tensors, dim=0)


def Train(G: torch.nn.Module, D: torch.nn.Module, epoch_count, iter_count):
    G.train(True)
    D.train(True)
    epoch_count += 1
    batches_train = len(train_loader) / train_loader.batch_size
    pbar = tqdm(enumerate(train_loader), total=batches_train, leave=False)

    Epoch_time = time.time()

    for i, data in pbar:
        iter_count += 1
        # images, _ = data

        # Implement your training loop here. images will be the datastructure
        # being returned from your dataloader.
        # 1) Load and transfer data to device
        source, target, swap, mask = data['source'].squeeze(), data['target'].squeeze(), data['swap'].squeeze(), data['mask'].squeeze()
        print('source.shape', source.shape, source.type())
        img_transfer = transfer_mask(source, target, mask)
        print('img_transfer.shape', img_transfer.shape, img_transfer.type())
        img_blend = blend_imgs(source, target, mask)
        print('img_blend.shape', img_blend.shape)

        print('before concat', img_transfer.shape, img_transfer.type(), target.shape, target.type(), mask.shape, mask.type())
        img_transfer_input = torch.cat((img_transfer, target, mask), dim=2)
        print('img_transfer_input.shape', img_transfer_input.shape)

        img_transfer_input_pyd = img_utils.create_pyramid(img_transfer_input, len(source[0]))
        print('len(img_transfer_input_pyd)', len(img_transfer_input_pyd), img_transfer_input_pyd[0].shape)

        # 2) Feed the data to the networks.
        # Blend images
        img_blend_pred = G(img_transfer_input_pyd)
        print('img_blend_pred.shape', img_blend_pred.shape)


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

        loss_G_total = rec_weight * loss_rec + gan_weight * loss_G_GAN

        # 5) Perform backward calculation.
        # 6) Perform the optimizer step.
        # Update generator weights
        optimizer_G.zero_grad()
        loss_G_total.backward()
        optimizer_G.step()

        # Update discriminator weights
        optimizer_D.zero_grad()
        loss_D_total.backward()
        optimizer_D.step()

        if iter_count % displayIter == 0:
            # Write to the log file.
            # trainLogger.write(f'Epoch: {epoch_count} / {max_epochs}; LR: {scheduler_G.get_lr()[0]:.0e};\n'
            #                   f'pixelwise={loss_pixelwise}, id={loss_id}, attr={loss_attr}, rec={loss_rec}, '
            #                   f'g_gan={loss_G_GAN}, d_gan={loss_D_total}')
            print(f'Epoch: {epoch_count} / {max_epochs}; LR: {scheduler_G.get_lr()[0]:.0e};\n'
                  f'pixelwise={loss_pixelwise}, id={loss_id}, attr={loss_attr}, rec={loss_rec}, '
                  f'g_gan={loss_G_GAN}, d_gan={loss_D_total}')

        # Print out the losses here. Tqdm uses that to automatically print it
        # in front of the progress bar.
        pbar.set_description()

    # Save output of the network at the end of each epoch. The Generator

    t_source, t_swap, t_target, t_pred, t_blend = Test(G)
    for b in range(t_pred.shape[0]):
        total_grid_load = [t_source[b], t_swap[b], t_target[b],
                           t_pred[b], t_blend[b]]
        grid = img_utils.make_grid(total_grid_load,
                                   cols=len(total_grid_load))
        grid = img_utils.tensor2rgb(grid.detach())
        imageio.imwrite(visuals_loc + '/Epoch_%d_output_%d.png' %
                        (epoch_count, b), grid)

    utils.saveModels(G, optimizer_G, iter_count,
                     checkpoint_pattern % ('G', epoch_count))
    utils.saveModels(D, optimizer_D, iter_count,
                     checkpoint_pattern % ('D', epoch_count))
    tqdm.write('[!] Model Saved!')

    return np.nanmean(total_loss_pix), \
           np.nanmean(total_loss_id), np.nanmean(total_loss_attr), \
           np.nanmean(total_loss_rec), np.nanmean(total_loss_G_Gan), \
           np.nanmean(total_loss_D_Gan), iter_count


def Test(G):
    with torch.no_grad():
        G.eval()
        t = enumerate(testLoader)
        i, (images) = next(t)

        # Feed the network with images from test set

        # Blend images
        pred = G(img_transfer_input)
        # You want to return 4 components:
        # 1) The source face.
        # 2) The 3D reconsturction.
        # 3) The target face.
        # 4) The prediction from the generator.
        # 5) The GT Blend that the network is targettting.


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
        total_loss = Train(G=generator, D=discriminator, epoch_count=epoch_count,
                           iter_count=iter_count)  # love passing global variables

        # total_loss = proces_epoch(train_loader, train=True)
        # if val_loader is not None:
        #     with torch.no_grad():
        #         total_loss = proces_epoch(val_loader, train=False)
        #
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
