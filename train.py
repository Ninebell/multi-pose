from PIL import Image
import h5py
from PIL import ImageDraw

from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from torch_module.utils import train_model
from torch_module.losses import pixel_logistic_focal_loss
from HRNetTorch.HRNet import HRNet
import os

import json

import numpy as np
import torch
from matplotlib import pyplot as plt


# from utils import data_generator, save_heatmap, save_limb

from CenterNetPose import CenterNetPose
from data_maker import data_generator_from_hdf, inference_joints, pair_joint_to_person, image_list_blend, MPiiDataset
import tqdm

root_path = 'E:\\dataset\\mpii\\model'


def mask(target):
    zeros = torch.zeros(target.shape).cuda()
    return torch.where(target != zeros, 1 - zeros, zeros)


def l2_loss(target, predict):
    return torch.pow(target - predict, 2)


def joint_loss(target, predict):
    joint_mask = mask(target)
    return (joint_mask*l2_loss(target, predict)).sum()


def limb_loss(target_x, target_y, predict_x, predict_y):
    limb_x_mask = mask(target_x)
    limb_y_mask = mask(target_y)
    limb_mask = limb_x_mask + limb_y_mask - limb_x_mask*limb_y_mask

    limb_x_loss = l2_loss(target_x, predict_x)
    limb_y_loss = l2_loss(target_y, predict_y)
    return (limb_mask*(limb_x_loss+limb_y_loss)).sum()


def hr_net_loss(target, predict):
    joint_target = target[0]
    limb_x_target = target[1][:, 0:28:2]
    limb_y_target = target[1][:, 1:28:2]

    joint_predict = predict[:,:15]
    limb_x_predict = predict[:,15:43:2]
    limb_y_predict = predict[:,16:43:2]

    return joint_loss(joint_target, joint_predict)+limb_loss(limb_x_target,limb_y_target, limb_x_predict, limb_y_predict)


def center_net_loss(target, predict):
    total_loss = 0
    for i in range(len(predict)):
        joint = joint_loss(target[0], predict[i][0])
        limb = limb_loss(target[1][:,0:28:2], target[1][:,1:28:2], predict[i][1][:,0:28:2], predict[i][1][:,1:28:2])
        total_loss = total_loss + joint + limb
        # total_loss = total_loss + hr_net_loss(target, predict[i])
    return total_loss


# def custom_loss(target, predict):
#     depth_loss_joint = 0
#     depth_loss_limb = 0
#     for i in range(len(predict)):
#         depth_loss_joint += pixel_logistic_focal_loss(target[0], predict[i][0])
#         depth_loss_limb += torch.sqrt(torch.mean(torch.pow(target[1]-predict[i][1], 2)))
#
#     return (depth_loss_joint/len(predict) + depth_loss_limb/len(predict))/2
#

def check_point(model, train_info):
    if train_info.validate_loss < train_info.min_validate_loss:
        train_info.min_validate_loss = train_info.validate_loss
        Path = root_path + '\\{0}'.format(train_info.min_validate_loss)
        os.makedirs(Path, exist_ok=True)
        torch.save(model.state_dict(), Path + '\\model.dict')


def test_model(net, model_path, save_path, data_loader_conf):
    net.cuda()
    net.load_state_dict(torch.load(os.path.join(save_path, model_path)))

    for idx, (x, c) in tqdm.tqdm(enumerate(data_loader_conf['loader'](data_loader_conf['conf']))):
        test = net(x)

        img = x[0].detach().cpu().numpy()
        img = np.transpose(img, (1,2,0))
        img = np.array(img*255, dtype=np.uint8)
        print(img.shape)
        print(test.shape)
        heat = np.squeeze(test[0, :15].detach().cpu().numpy())
        print(np.max(heat))
        limb = np.squeeze(test[0, 15:].detach().cpu().numpy())
        print(heat.shape, limb.shape)
        crop_size = c[0].type(torch.IntTensor)
        heat_base = heat[0,:,:]
        for i in range(1,15):
            heat_base = np.maximum(heat_base, heat[i,:,:])
        max_pair, joints_list = inference_joints(heat, limb)
        joints_list = pair_joint_to_person(max_pair, joints_list,
                                           '{},{},{},{}'.format(crop_size[0], crop_size[1], crop_size[2], crop_size[3]))
        base = np.zeros((crop_size[3] - crop_size[1], crop_size[2] - crop_size[0]), dtype=np.uint8)
        base = Image.fromarray(base)
        base_draw = ImageDraw.ImageDraw(base)

        for joints in joints_list:
            for joint in joints:
                joint = joints[joint]
                if joint is None:
                    continue
                base_draw.ellipse((joint[0] - 10, joint[1] - 10, joint[0] + 10, joint[1] + 10), fill='white')
        base = base / np.max(base)
        plt.subplot(2, 4, 1)
        plt.imshow(img)
        plt.subplot(2, 4, 2)
        plt.imshow(image_list_blend(img, [base]))
        plt.subplot(2, 4, 3)
        plt.imshow(image_list_blend(img, heat[:-1]))
        plt.subplot(2, 4, 4)
        plt.imshow(image_list_blend(base, heat[:-1]))
        plt.show()


if __name__ == "__main__":
    net = CenterNetPose(256, [15, 28], ['sigmoid', 'tanh'], n_layer=4, n_stack=5)
    # net = HRNet(192, 11, 3, [15, 28], ['sigmoid', 'tanh'], 'relu')
    # net = HRNet(feature=256, depth=7, input_ch=6, output_ch=[15,28], out_act=['sigmoid', 'sigmoid'], act='selu').cuda()
    net.cuda()

    data_set_root = 'E:\\dataset\\mpii'
    save_path = '{}\\result'.format(data_set_root)
    os.makedirs(save_path, exist_ok=True)
    lr = 1e-4
    exp = 0.99
    optim = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=0.1)

    batch_size = 4
    train_hdf5 = h5py.File('{}\\info\\train_info.h5'.format(data_set_root), 'r')
    validate_hdf5 = h5py.File('{}\\info\\validate_info.h5'.format(data_set_root), 'r')
    test_hdf5 = h5py.File('{}\\info\\test_info.h5'.format(data_set_root), 'r')

    is_train = True
    scheduler = lr_scheduler.ExponentialLR(optim, gamma=exp)

    if is_train:
        train_model(50, net, center_net_loss, optim,
                {
                    'loader': data_generator_from_hdf,
                    'conf': {
                        'hdf': train_hdf5,
                        'batch': batch_size,
                        'shuffle': True,
                        'is_train': True,
                        'is_tensor': True,
                        'is_cuda': True}
                },
                {
                    'loader': data_generator_from_hdf,
                    'conf': {
                        'hdf': validate_hdf5,
                        'batch': batch_size,
                        'shuffle': False,
                        'is_train': True,
                        'is_tensor': True,
                        'is_cuda': True}
                },
                scheduler,
                save_path, 'center_256_4_5_adam_{}_exp_{}'.format(lr,exp), check_point,
            )
    else:
        test_model(net, 'model.dict', save_path,
                   {
                       'loader': data_generator_from_hdf,
                       'conf': {
                           'hdf': train_hdf5,
                           'batch': 1,
                           'shuffle': False,
                           'is_train': False,
                           'is_tensor': True,
                           'is_cuda': True}
                   })




