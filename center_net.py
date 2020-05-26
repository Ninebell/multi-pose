import torch
import torch.nn as nn
import torch.nn.functional as F
import os
# from CenterNet.data_util import data_generator, check_iou, calc_inter
from time import sleep
from tqdm import tqdm
from utils import data_generator, save_limb, save_heatmap
import matplotlib.pyplot as plt
from conf import *
import numpy as np
from PIL import ImageDraw, Image
# from models import save_limb, save_heatmap


class AttentionBlock(nn.Module):
    def __init__(self, feature, ratio):
        super(AttentionBlock, self).__init__()
        self.__build__(feature, ratio)

    def __build__(self,feature, ratio):
        self.shared_mlp1 = nn.Linear(feature, feature//ratio)
        self.shared_mlp2 = nn.Linear(feature//ratio, feature)

        self.avg_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.attention_conv = nn.Conv2d(feature*2, feature, 7, stride=1, padding=3)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = torch.nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        init = x
        ch_avg = self.avg_pool(x)
        ch_max = self.max_pool(x)
        ch_avg = torch.flatten(ch_avg, start_dim=1)
        ch_max = torch.flatten(ch_max, start_dim=1)

        ch_avg = self.shared_mlp1(ch_avg)
        ch_avg = self.shared_mlp2(ch_avg)

        ch_max = self.shared_mlp1(ch_max)
        ch_max = self.shared_mlp2(ch_max)

        channel_attention = torch.sigmoid(ch_max + ch_avg)

        channel_attention = channel_attention.view((channel_attention.shape[0],channel_attention.shape[1], 1, 1))
        x = channel_attention*init

        sp_avg = self.avg_pool(x)
        sp_max = self.max_pool(x)
        sp_conv = torch.cat((sp_avg, sp_max), 1)
        sp_conv = self.attention_conv(sp_conv)
        sp_conv = torch.sigmoid(sp_conv)

        x = x * sp_conv
        return x


class ResidualBlock(nn.Module):
    def __init__(self, input_feature, output_feature, attention):
        super(ResidualBlock, self).__init__()
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.attention=attention
        self.__build__()

    def __build__(self):
        self.conv1 = nn.Conv2d(self.input_feature, self.output_feature, 1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(self.output_feature, self.output_feature, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.output_feature, self.output_feature, 1, stride=1, padding=0)
        if self.attention:
            self.attention = AttentionBlock(self.output_feature, 8)

        # self.channel_attention =

        if self.input_feature != self.output_feature:
            self.conv4 = nn.Conv2d(self.input_feature, self.output_feature,3, stride=1, padding=1)

        self.batch1 = nn.BatchNorm2d(self.output_feature)
        self.batch2 = nn.BatchNorm2d(self.output_feature)
        self.batch3 = nn.BatchNorm2d(self.output_feature)

    def forward(self, x):
        init = x
        # print(x.size()[-2:])
        x = torch.selu(self.batch1(self.conv1(x)))
        x = torch.selu(self.batch2(self.conv2(x)))
        x = torch.selu(self.batch3(self.conv3(x)))
        if self.attention:
            x = self.attention.forward(x)

        if self.input_feature != self.output_feature:
            init = torch.selu(self.conv4(init))
        return x + init


class Hourglass(nn.Module):
    def __init__(self, input_feature, output_feature):
        super(Hourglass, self).__init__()
        self.input_feature = input_feature
        self.output_feature = output_feature

        self.__build__()

    def __build__(self):
        i_f = self.input_feature
        o_f = self.output_feature

        self.down1 = ResidualBlock(i_f, o_f, False)
        self.down2 = ResidualBlock(o_f, o_f, False)
        self.down3 = ResidualBlock(o_f, o_f, False)
        self.down4 = ResidualBlock(o_f, o_f, False)
        self.down5 = ResidualBlock(o_f, o_f, False)

        self.skip1 = ResidualBlock(o_f, o_f, True)
        self.skip2 = ResidualBlock(o_f, o_f, True)
        self.skip3 = ResidualBlock(o_f, o_f, True)
        self.skip4 = ResidualBlock(o_f, o_f, True)

        self.middle1 = ResidualBlock(o_f, o_f, False)
        self.middle2 = ResidualBlock(o_f, o_f, False)
        self.middle3 = ResidualBlock(o_f, o_f, False)

        self.up1 = ResidualBlock(i_f, o_f, False)
        self.up2 = ResidualBlock(o_f, o_f, False)
        self.up3 = ResidualBlock(o_f, o_f, False)
        self.up4 = ResidualBlock(o_f, o_f, False)
        self.up5 = ResidualBlock(o_f, o_f, False)

    def forward(self, x):
        down1 = self.down1(x)
        skip1 = self.skip1(down1)
        down1 = F.max_pool2d(down1, (2,2))

        down2 = self.down2(down1)
        skip2 = self.skip2(down2)
        down2 = F.max_pool2d(down2, (2,2))

        down3 = self.down3(down2)
        skip3 = self.skip3(down3)
        down3 = F.max_pool2d(down3, (2,2))

        down4 = self.down4(down3)
        skip4 = self.skip4(down4)
        down4 = F.max_pool2d(down4, (2,2))

        down5 = self.down5(down4)

        middle1 = self.middle1(down5)
        middle2 = self.middle2(middle1)
        middle3 = self.middle3(middle2)

        up1 = F.interpolate(middle3, scale_factor=2)
        up1 = skip4 + up1
        up1 = self.up1(up1)

        up2 = F.interpolate(up1, scale_factor=2)
        up2 = skip3 + up2
        up2 = self.up2(up2)

        up3 = F.interpolate(up2, scale_factor=2)
        up3 = skip2 + up3
        up3 = self.up3(up3)

        up4 = F.interpolate(up3, scale_factor=2)
        up4 = skip1 + up4
        up4 = self.up4(up4)

        up5 = self.up5(up4)

        return up5


class CenterNet(nn.Module):

    def __init__(self, feature, output):
        super(CenterNet, self).__init__()
        self.feature = feature
        self.output = output

        self.__build__()

    def __build__(self):
        feature = self.feature
        self.conv7 = nn.Conv2d(3, feature, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(feature, feature, 3, stride=1, padding=1)
        self.hour1 = Hourglass(feature, feature)

        self.hour2 = Hourglass(feature, feature)
        self.res1 = ResidualBlock(feature, feature, False)
        self.res2 = ResidualBlock(feature, feature, False)
        self.res3 = ResidualBlock(feature, feature, False)
        # self.intermediate = nn.Conv2d(feature, self.output, 1, stride=1, padding=0)
        # self.intermediate_res = ResidualBlock(self.output, feature)
        self.heat_last_f = nn.Conv2d(feature, feature, 3, stride=1, padding=1)
        self.heat_last = nn.Conv2d(feature, 17, 1, stride=1, padding=0)

        self.size_last_f = nn.Conv2d(feature, feature, 3, stride=1, padding=1)
        self.size_last = nn.Conv2d(feature, 16, 1, stride=1, padding=0)
        self.batch1 = nn.BatchNorm2d(feature)
        self.batch2 = nn.BatchNorm2d(feature)
        self.batch3 = nn.BatchNorm2d(feature)
        self.batch4 = nn.BatchNorm2d(feature)

    def forward(self, x):
        x = torch.selu(self.batch1(self.conv7(x)))
        x = F.max_pool2d(torch.selu(self.batch2(self.conv3(x))), (2,2))
        # init = x
        x = self.hour1(x)
        x = self.res1(x)
        x = self.hour2(x)
        res = self.res2(x)
        res = self.res3(res)

        heat = torch.selu(self.batch3(self.heat_last_f(res)))
        heat = torch.sigmoid(self.heat_last(heat))

        size = torch.selu(self.batch4(self.size_last_f(res)))
        size = torch.sigmoid(self.size_last(size))

        return heat, size


def focal_loss(output, target):

    ones = torch.ones((64,64)).cuda()
    zeros = torch.zeros((64,64)).cuda()

    ones_board = torch.where(target == 1, output, ones)
    zeros_board = torch.where(target != 1, output, zeros)

    alpha = 2
    beta = 4

    N = torch.where(target == 1, target, zeros)
    N = torch.sum(N)

    epsilon = 1e-10

    ones_board = torch.pow(1-ones_board, alpha) * torch.log(ones_board+epsilon)
    zeros_board = torch.pow(1-target, beta) * torch.pow(zeros_board, alpha) * torch.log(1-zeros_board+epsilon)

    return -(ones_board+zeros_board).sum()/N


def center_loss(output, target):
    o_heat = output[0]
    o_size = output[1]
    t_heat = torch.from_numpy(target[0]).type(torch.FloatTensor).cuda()
    t_size = torch.from_numpy(target[1]).type(torch.FloatTensor).cuda()

    point_fl = focal_loss(o_heat, t_heat)
    limb_fl = focal_loss(o_size, t_size)

    # sz = size_loss(o_size, t_size, t_heat)
    return point_fl + limb_fl


def draw_roi(img, heat, size):
    heat = heat[0]
    img = np.asarray(img*255,dtype=np.uint8)
    img = Image.fromarray(img)
    img_draw = ImageDraw.Draw(img)

    center = []
    for r in range(1,63):
        for c in range(1,63):
            if heat[r,c] == np.max(heat[r-1:r+2, c-1:c+2]) and heat[r,c] > 0.5:
                center.append((c,r))

    for point in center:
        w = size[0,point[1],point[0]] / 592 * 256
        h = size[1,point[1],point[0]] / 480 * 256
        point = point[0]*4, point[1]*4
        img_draw.rectangle((point[0]-w//2, point[1]-h//2, point[0]+w//2, point[1]+h//2), outline='red', width=1)

    return img


if __name__ == "__main__":
    epoches = 1000
    min_loss = 10000

    data_set_path = "D:\\{0}\\result".format('mpii')

    net = CenterNet(256, 3)
    print(net)

    criterion = center_loss

    is_cuda = torch.cuda.is_available()

    if is_cuda:
        net = net.cuda()

    lr = 1e-4

    losses = []
    ioues = []

    optim = torch.optim.Adam(net.parameters(), lr)

    net.load_state_dict(torch.load( '{1}/{0}/model.dict'.format(38, data_set_path)))

    pytorch_total_params = sum(p.numel() for p in net.parameters())

    print(pytorch_total_params)

    epoch = 38
    # for epoch in range(1, epoches):
    while True:
        epoch = epoch+1
        iou_count = 0
        epoch_loss = 0
        for data in tqdm(data_generator(8, shuffle=True, is_train=True)):
            x, heat, limb = data
            if is_cuda:
                x = torch.from_numpy(x).type(torch.FloatTensor).cuda()

            optim.zero_grad()
            result = net(x)
            loss = criterion(result, [heat, limb])
            loss.backward()
            epoch_loss += loss.item()

            optim.step()

        print(epoch, epoch_loss)
        losses.append(loss.item())
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
                    # result[0]
                    heat_map_result = np.moveaxis(result[0][0].cpu().numpy(),0, 2)
                    limb_map_result = np.moveaxis(result[1][0].cpu().numpy(),0, 2)
                    save_heatmap(heat_map_result, '{2}/{0}/heatmap_{1}.png'.format(epoch, idx, data_set_path))
                    save_limb(limb_map_result, '{2}/{0}/limb{1}.png'.format(epoch, idx, data_set_path))

                    # save_heatmap(result[2][0], '{2}/{0}/heatmap_{1}.png'.format(epoch, idx, data_set_path))
                    # save_limb(result[3][0], '{2}/{0}/limb_{1}.png'.format(epoch, idx, data_set_path))
                # except:
                #     continue
    count = 0
