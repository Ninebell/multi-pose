from torch_module.layers import *
import json
import torch.nn as nn
import numpy as np
import os
import random
from PIL import Image

from conf import *

from torch_module.utils import BaseDataLoader, train_model


class CenterNet(nn.Module):

    def __init__(self, pre_train_path, state_dict):
        super(CenterNet, self).__init__()
        self.load_model(pre_train_path, state_dict)
        self.__build__()

    def __init__(self, feature, out_channel, out_activation, n_layer=5, n_stack=2):
        super(CenterNet, self).__init__()
        self.feature = feature
        self.n_layer = n_layer
        self.n_stack = n_stack
        self.out_ch = out_channel
        self.out_act_str = out_activation
        self.out_activation = [activation_layer[name] for name in self.out_act_str]
        self.__build__()

    def save_meta_data(self, save_path):
        meta = {'feature': self.feature,
                'out_channel': self.out_ch,
                'out_activation': self.out_act_str,
                'n_layer': self.n_layer,
                'n_stack': self.n_stack}
        state_dict = self.state_dict()

        meta = json.dumps(meta)
        fp = open(save_path+'\\meta.ini', 'w')
        fp.write(meta)
        fp.close()

        torch.save(state_dict, '{0}\\model.dict'.format(save_path))

    def load_model(self, meta, state_dict):
        self.feature = meta['feature']
        self.out_ch = meta['out_channel']
        self.out_act_str = meta['out_activation']
        self.out_activation = [activation_layer[name] for name in self.out_act_str]
        self.n_layer = meta['n_layer']
        self.n_stack = meta['n_stack']
        self.load_state_dict(torch.load(state_dict))

    def __build__(self):
        feature = self.feature
        n_s = self.n_stack
        self.conv7 = Conv2D(3, feature, 7, 2, 3, torch.relu)

        self.block1 = BottleNeckBlock(feature, feature)
        self.block2 = BottleNeckBlock(feature, feature)
        self.block3 = BottleNeckBlock(feature, feature)
        self.block4 = BottleNeckBlock(feature, feature)

        self.hours = nn.ModuleList([Hourglass(feature, self.n_layer) for _ in range(n_s)])

        self.bottles_1 = nn.ModuleList([BottleNeckBlock(feature, feature) for _ in range(n_s-1)])
        self.bottles_2 = nn.ModuleList([BottleNeckBlock(feature, feature) for _ in range(n_s-1)])

        self.inter_h_o = nn.ModuleList([
            nn.Sequential(
                Conv2D(feature, self.out_ch[0], 1, 1, 0, self.out_activation[0])
            ) for _ in range(n_s - 1)])
        self.inter_l_o = nn.ModuleList([
            nn.Sequential(
                Conv2D(feature, self.out_ch[1], 1, 1, 0, self.out_activation[1])
            ) for _ in range(n_s - 1)])

        self.inter_h_a = nn.ModuleList([Conv2D(self.out_ch[0], feature, 1, 1, 0, torch.relu, True) for _ in range(n_s - 1)])
        self.inter_l_a = nn.ModuleList([Conv2D(self.out_ch[1], feature, 1, 1, 0, torch.relu, True) for _ in range(n_s - 1)])

        self.out_h_f = Conv2D(feature, feature, 3, stride=1, padding=1, activation=torch.relu)
        self.out_l_f = Conv2D(feature, feature, 3, stride=1, padding=1, activation=torch.relu)

        self.out_h = Conv2D(feature, self.out_ch[0], 1, stride=1, padding=0, activation=self.out_activation[0])
        self.out_l = Conv2D(feature, self.out_ch[1], 1, stride=1, padding=0, activation=self.out_activation[1])

        self.batch3 = nn.BatchNorm2d(feature)

        self.max_pool = nn.MaxPool2d(2, stride=2, padding=0)

    def forward(self, x):
        out_put = []
        x = self.conv7(x)
        x = self.max_pool(self.block1(x))
        x = self.block2(x)
        x = self.block3(x)

        for i in range(self.n_stack-1):
            init = x
            x = self.hours[i](x)
            x = self.bottles_1[i](x)

            inter_h = self.out_activation[0](self.inter_h_o[i](x))
            inter_l = self.out_activation[1](self.inter_l_o[i](x))

            out_put.append([inter_h, inter_l])
            x = self.bottles_2[i](x)

            i_h = self.inter_h_a[i](inter_h)
            i_l = self.inter_l_a[i](inter_l)
            inter = (i_h + i_l) / 2
            x = x + init + inter

        last_hour = self.hours[-1](x)

        out_block = torch.relu(self.batch3(self.block4(last_hour)))
        h_f = self.out_h_f(out_block)
        l_f = self.out_l_f(out_block)
        out_h = self.out_h(h_f)
        out_l = self.out_l(l_f)

        out_put.append([out_h, out_l])

        return out_put

    def info(self):
        total_params = sum(p.numel() for p in self.parameters())
        print("{0:^40s}".format('CenterNet Information'))
        print("{0:^40s}".format("{0:22s}: {1:10d}".format('hourglass repeat', self.n_stack)))
        print("{0:^40s}".format("{0:22s}: {1:10d}".format('hourglass layer count', self.n_layer)))
        print("{0:^40s}".format("{0:22s}: {1:10,d}".format('total parameter', total_params)))


heat_map_path = "/heat/"
limb_path = "/limb/"


class CenterNetDataLoader(BaseDataLoader):
    def __init__(self, batch_size, path, shuffle, repeat, is_cuda):
        super(CenterNetDataLoader).__init__()
        self.batch_size = batch_size
        self.path = path
        self.shuffle = shuffle
        self.repeat = repeat
        self.is_cuda = is_cuda

    def data_load(self):
        def concat_map(path, count):
            file_path = path + '\\{}\\'.format(0) + file_name
            base = np.asarray(Image.open(file_path)) / 255.
            base = np.reshape(base, (64, 64, 1))

            for h in range(1, count):
                file_path = path + '\\{}\\'.format(h) + file_name
                cc = np.asarray(Image.open(file_path)) / 255.
                cc = np.reshape(cc, (64, 64, 1))
                base = np.concatenate([base, cc], axis=-1)
            return base

        base_path = self.path
        dirs = os.listdir(base_path + heat_map_path + '0')
        if self.shuffle:
            random.shuffle(dirs)
        batch_iter = len(dirs) // self.batch_size
        heat_map_len = len(os.listdir(base_path + heat_map_path))
        limb_len = len(os.listdir(base_path + limb_path))
        mean = np.array([0.485, 0.456, 0.406])
        mean = np.array([np.ones((256, 256)) * m for m in mean])
        std = np.array([0.229, 0.224, 0.225])
        std = np.array([np.ones((256, 256)) * s for s in std])
        mean = np.moveaxis(mean, 0, -1)
        std = np.moveaxis(std, 0, -1)
        for idx in range(batch_iter):
            x = []
            org_img = []
            heat_maps = []
            limbs = []

            for b_i in range(self.batch_size):
                file_name = dirs[idx * self.batch_size + b_i]
                img = Image.open(base_path + '\\image\\' + file_name)
                if img.mode == 'L':
                    continue
                # print(img.mode, file_name)
                img = img.resize((256, 256))
                org_img.append(img)
                img = np.asarray(img) / 255.
                img = (img - mean) / std
                img = np.moveaxis(img, 2, 0)
                x.append(img)

                base_heat = concat_map(base_path + heat_map_path, heat_map_len)
                base_limb = concat_map(base_path + limb_path, limb_len)

                base_heat = np.moveaxis(base_heat, 2, 0)
                base_limb = np.moveaxis(base_limb, 2, 0)
                heat_maps.append(base_heat)
                limbs.append(base_limb)

            x = torch.from_numpy(np.asarray(x)).type(torch.FloatTensor).cuda()
            heat_maps = torch.from_numpy(np.asarray(heat_maps)).type(torch.FloatTensor).cuda()
            limbs = torch.from_numpy(np.asarray(limbs)).type(torch.FloatTensor).cuda()
            yield x, [(heat_maps, limbs) for _ in range(self.repeat)]


def center_net_loss(target, predict):
    total_loss = 0
    for layer in range(len(predict)):
        point_loss = torch.mean(torch.pow(torch.abs(target[layer][0]-predict[layer][0]), 2))
        limb_loss = torch.mean(torch.pow(torch.abs(target[layer][1]-predict[layer][1]), 2))
        total_loss += (point_loss + limb_loss) / 2

    return total_loss / len(predict)


if __name__ == "__main__":
    net = CenterNet(128, [17, 16], out_activation=['sigmoid', 'sigmoid'], n_layer=4, n_stack=3)

    inp = torch.rand((1,3,256,256)).cuda()
    net.cuda()
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)

    cnd = CenterNetDataLoader(4, root_path+train_path, True, 3, True)

    train_model(100, net, center_net_loss, optim, cnd, None, root_path+'/tensor_result')

    result = net(inp)

    for ret in result:
        print(ret[0].shape)
        print(ret[1].shape)


