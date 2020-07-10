from time import sleep

import PIL.Image

import matplotlib.pyplot as plt

import json
import os
import argparse

from tqdm import tqdm
import numpy as np
import torch

import conf
import torch_model.center_net

from torch_model.losses import focal_loss
from utils import data_generator, save_heatmap, save_limb
from data_set.mpii import data_generator as mpii_generator
from data_set.mpii import validate_image

from data_set.mpii import get_joints_from_heat_map


def center_loss(output, target):
    o_heat = output[0]
    o_limb = output[1]
    t_heat = target[:,:17,:,:]
    t_limb = target[:,17:,:,:]

    # point_fl = focal_loss(o_heat, t_heat)
    # limb_fl = focal_loss(o_limb, t_limb)

    point_fl = torch.mean(torch.pow(torch.abs(o_heat-t_heat), 2))
    limb_fl = torch.mean(torch.pow(torch.abs(o_limb-t_limb) ,2))
    print(point_fl.item(), limb_fl.item())

    # sz = limb_loss(o_limb, t_size, t_heat)

    return point_fl + limb_fl


def train_model(net, optim, criterion, batch_size, is_cuda=True):
    iter_count = 0
    epoch_loss = 0
    repeat = net.n_stack

    for data in tqdm(data_generator(batch_size)):
        iter_count += 1
        x, heat, limb, org = data
        if is_cuda:
            tmp = [0 for _ in range(repeat * 2)]
            target = [0 for _ in range(repeat)]
            for i in range(repeat):
                tmp[i * 2] = heat.copy()
                tmp[i * 2 + 1] = limb.copy()

            for i in range(repeat):
                target[i] = np.concatenate([tmp[i * 2], tmp[i * 2 + 1]], axis=1)

            target = np.asarray([tar for tar in target])
            target = torch.from_numpy(target).type(torch.FloatTensor).cuda(non_blocking=True)
            x = torch.from_numpy(x).type(torch.FloatTensor).cuda(non_blocking=True)

        result = net(x)
        loss = 0
        for i in range(repeat):
            print('======================')
            inter_loss = criterion(result[i], target[i])
            print('======================')
            loss += inter_loss
        optim.zero_grad()
        loss.backward()
        optim.step()
        epoch_loss += loss.item()
    return epoch_loss, iter_count


def test_model(net, limit, path, is_cuda=True):

    os.makedirs(path, exist_ok=True)
    with torch.no_grad():
        for idx, value in enumerate(data_generator(1, shuffle=False, is_train=False)):
            if True:
                if limit !=0 and idx > limit:
                    break
                x, heat, limb, org = value
                temp = np.moveaxis(x[0], 0, 2)
                temp = np.array(temp*255, dtype=np.uint8)
                img = PIL.Image.fromarray(temp)
                img.save('{0}/input_{1}.png'.format(path, idx))
                if is_cuda:
                    x = torch.from_numpy(x).type(torch.FloatTensor).cuda()
                result = net(x)
                i=-1
                # for i in range(len(result)):
                    # heat = heat[0]
                    # heat = np.moveaxis(heat,0, 2)
                    # save_heatmap(heat, '{0}/{1}_heatmap_target{2}.png'.format(path, i, idx))

                heat_map_result = result[i][0][0, :, :, :]
                limb_map_result = result[i][1][0, :, :, :]
                validate_image(heat_map_result, limb_map_result, org[0])
                # save_heatmap(heat_map_result, '{0}/{1}_heatmap_{2}.png'.format(path, i, idx))
                # save_limb(limb_map_result, '{0}/{1}_limb{2}.png'.format(path, i, idx))



def print_train_info(epoch, batch_size):
    print()
    print("{0:^40s}".format('Train Information'))
    print("{0:^40s}".format("{0:22s}: {1:10,d}".format('epoch', epoch)))
    print("{0:^40s}".format("{0:22s}: {1:10,d}".format('batch size', batch_size)))
    print("{0:^40s}".format("{0:22s}: {1:10,d}".format('input data', conf.get_train_data_num())))


def _main(epoches, batch_size, repeat, n_layer, save_root_path, pretrain):
    min_loss = None

    net = torch_model.center_net.CenterNet(128, [17, 16], out_activation=[torch.sigmoid, torch.sigmoid], n_layer=n_layer, n_stack=repeat)
    net.info()
    print_train_info(epoches, batch_size)

    criterion = center_loss
    # criterion = torch.nn.L1Loss()

    is_cuda = torch.cuda.is_available()

    if is_cuda:
        net = net.cuda()

    lr = 1e-3

    if pretrain is not None:
        net.load_state_dict(torch.load(pretrain))

    optim = torch.optim.Adam(net.parameters(), lr)

    sch = torch.optim.lr_scheduler.StepLR(optim, 50)

    for epoch in range(1, epoches):
        # epoch_loss, iter_count = train_model(net, optim, criterion, batch_size)
        # epoch_loss /= iter_count
        # # sch.step()
        # print('\n', epoch, epoch_loss, '\n')
        save_path = '{0}\\{1}_{2:4d}\\'.format(save_root_path, epoch, int(1*100))
        # os.makedirs(save_path, exist_ok=True)
        # torch.save(net.state_dict(), '{0}\\model.dict'.format(save_path))
        # if min_loss is None or min_loss > epoch_loss:
        #     min_loss = epoch_loss
        test_model(net, 0, save_path)


def get_arguments():
    data_set_path = "E:\\dataset\\{0}\\result".format(conf.data_set_name)
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat', '-r', nargs='+', help='hourglass count', default=[3], dest='repeat', type=int)
    parser.add_argument('--nstack', '-n', nargs='+', help='hourglass layer count', default=[4], dest='n_stack', type=int)
    parser.add_argument('--save', '-s', nargs='+', help='save path', default=[data_set_path], dest='save_path')
    parser.add_argument('--epoch', '-e', nargs='+', help='  epoch count', default=[200], dest='epoch', type=int)
    parser.add_argument('--batch', '-b', nargs='+', help='batch size', default=[10], dest='batch_size', type=int)
    parser.add_argument('--pretrain', '-p', nargs='+', help='pretrain model', default=['model.dict'], dest='pretrain')

    repeat = parser.parse_args().repeat
    n_stack = parser.parse_args().n_stack
    save_path = parser.parse_args().save_path
    epoch = parser.parse_args().epoch
    batch_size = parser.parse_args().batch_size
    pretrain = parser.parse_args().pretrain

    return epoch[0], batch_size[0], repeat[0], n_stack[0], save_path[0], pretrain[0]


if __name__ == "__main__":
    epoch, batch_size, repeat, n_stack, save_path, pretrain = get_arguments()
    _main(epoch, batch_size, repeat, n_stack, save_path, pretrain)

