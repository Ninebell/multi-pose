from PIL import Image
import h5py

from torch.optim import SGD

from torch_module.utils import train_model
from torch_module.losses import pixel_logistic_focal_loss

import os

import json

import numpy as np
import torch
from matplotlib import pyplot as plt


# from utils import data_generator, save_heatmap, save_limb

from CenterNetPose import CenterNetPose
from data_maker import data_generator_from_hdf, inference_joints, pair_joint_to_person
import tqdm

root_path = 'D:\\dataset\\mpii\\model'


def custom_loss(target, predict):
    depth_loss_joint = 0
    depth_loss_limb = 0
    for i in range(len(predict)):
        depth_loss_joint += pixel_logistic_focal_loss(target[0], predict[i][0])
        depth_loss_limb += torch.sqrt(torch.mean(torch.pow(target[1]-predict[i][1], 2)))

    return (depth_loss_joint/len(predict) + depth_loss_limb/len(predict))/2


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

        print(test[1][1].shape)
        heat = np.squeeze(test[1][0].detach().cpu().numpy())
        plt.imshow(heat[0,:,:])
        plt.show()
        limb = np.squeeze(test[1][1].detach().cpu().numpy())
        print(heat.shape, limb.shape)
        max_pair, joints_list = inference_joints(heat, limb)
        print(max_pair)
        pair_joint_to_person(max_pair, joints_list, '{},{},{},{}'.format(c[0][0],c[0][1],c[0][2],c[0][3]))


if __name__ == "__main__":
    net = CenterNetPose(256, [15, 28], ['sigmoid', 'tanh'])
    net.cuda()
    data_set_root = 'D:\\dataset\\mpii'
    save_path = '{}\\result'.format(data_set_root)
    os.makedirs(save_path, exist_ok=True)
    optim = SGD(params=net.parameters(), lr=1e-2)

    batch_size = 8
    torch.nn.functional.cross_entropy()
    train_hdf5 = h5py.File('{}\\info\\train_info.h5'.format(data_set_root), 'r')
    validate_hdf5 = h5py.File('{}\\info\\validate_info.h5'.format(data_set_root), 'r')
    test_hdf5 = h5py.File('{}\\info\\test_info.h5'.format(data_set_root), 'r')

    is_train = True

    if is_train:
        train_model(50, net, custom_loss, optim,
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
                save_path, 'test', check_point,
                )
    else:
        test_model(net, 'best_model.dict', save_path,
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

