from time import sleep
import os

from tqdm import tqdm
import numpy as np
import torch

import conf
import torch_model.center_net
from torch_model.losses import focal_loss
from utils import data_generator, save_heatmap, save_limb


def center_loss(output, target):
    o_heat = output[0]
    o_size = output[1]
    t_heat = torch.from_numpy(target[0]).type(torch.FloatTensor).cuda()
    t_size = torch.from_numpy(target[1]).type(torch.FloatTensor).cuda()

    o_heat_2 = output[2]
    o_size_2 = output[3]
    t_heat_2 = torch.from_numpy(target[0]).type(torch.FloatTensor).cuda()
    t_size_2 = torch.from_numpy(target[1]).type(torch.FloatTensor).cuda()

    point_fl = focal_loss(o_heat, t_heat)
    limb_fl = focal_loss(o_size, t_size)

    point_fl_2 = focal_loss(o_heat_2, t_heat_2)
    limb_fl_2 = focal_loss(o_size_2, t_size_2)
    # sz = size_loss(o_size, t_size, t_heat)
    return point_fl + limb_fl + point_fl_2 + limb_fl_2


if __name__ == "__main__":
    epoches = 1000
    min_loss = 10000

    data_set_path = "E:\\dataset\\{0}\\result".format(conf.data_set_name)

    # net = torch_model.center_net.CenterNet(256, [17, 16], [torch.sigmoid, torch.sigmoid])
    net = torch_model.center_net.CenterNet2(256, 33)

    criterion = torch.nn.MSELoss()

    is_cuda = torch.cuda.is_available()

    if is_cuda:
        # net.cuda_adopt()
        net = net.cuda()

    lr = 1e-3

    losses = []
    ioues = []

    optim = torch.optim.Adam(net.parameters(), lr)

    # net.load_state_dict(torch.load( '{1}/{0}/model.dict'.format(17, data_set_path)))

    pytorch_total_params = sum(p.numel() for p in net.parameters())

    print(pytorch_total_params)

    sch = torch.optim.lr_scheduler.StepLR(optim, 150)

    epoch = 0
    # for epoch in range(1, epoches):
    while True:
        epoch = epoch+1
        iou_count = 0
        epoch_loss = 0
        for data in tqdm(data_generator(8, shuffle=True, is_train=True)):
            x, heat, limb = data
            optim.zero_grad()
            if is_cuda:
                target = [0 for i in range(4)]

                target[0] = heat.copy()
                target[1] = limb.copy()
                target[2] = heat.copy()
                target[3] = limb.copy()

                # inter = target[0]
                inter = np.concatenate([target[0], target[1]],axis=1)
                final = np.concatenate([target[2], target[3]],axis=1)
                target = np.asarray([inter,final])
                # target = np.asarray(target)
                target = torch.from_numpy(target).type(torch.FloatTensor).cuda()
                x = torch.from_numpy(x).type(torch.FloatTensor).cuda()

            result = net(x)
            inter_loss = criterion(result[0], target[0])
            loss = criterion(result[1], target[1])
            loss = inter_loss + loss
            loss.backward()
            epoch_loss += loss.item()
            optim.step()
            # sch.step()

        print(epoch, epoch_loss)
        # losses.append(loss.item())
        sleep(0.1)

        os.makedirs('{1}/{0}'.format(epoch, data_set_path),exist_ok=True)
        # if min_loss > epoch_loss:
        min_loss = epoch_loss
        torch.save(net.state_dict(), '{1}/{0}/model.dict'.format(epoch, data_set_path))

        with torch.no_grad():
            for idx, value in enumerate(data_generator(batch_size=1, shuffle=False, is_train=False)):
                if True:
                    if idx > 50:
                        break
                    x, heat, limb = value
                    heat_map = np.moveaxis(heat[0,:,:,:],0,2)
                    save_heatmap(heat_map, '{2}/{0}/heatmap_gt_{1}.png'.format(epoch, idx, data_set_path))
                    save_limb(np.moveaxis(limb[0,:,:,:],0,2), '{2}/{0}/limb_gt_{1}.png'.format(epoch, idx, data_set_path))

                    if is_cuda:
                        x = torch.from_numpy(x).type(torch.FloatTensor).cuda()
                    result = net(x)
                    heat_map_result = np.moveaxis(result[1][0,0:17,:,:].cpu().numpy(),0, 2)
                    limb_map_result = np.moveaxis(result[1][0,17:,:,:].cpu().numpy(),0, 2)
                    save_heatmap(heat_map_result, '{2}/{0}/heatmap_{1}.png'.format(epoch, idx, data_set_path))
                    save_limb(limb_map_result, '{2}/{0}/limb{1}.png'.format(epoch, idx, data_set_path))

