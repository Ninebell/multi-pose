from torch_model.layers import BottleNeckBlock, Hourglass
import numpy as np
import torch.functional as F
from torch_model.losses import focal_loss
import torch
from torchvision import datasets, transforms
import torch.nn as nn


class CenterNet(nn.Module):

    def __init__(self, feature, out_channel, out_activation, n_stack=2):
        super(CenterNet, self).__init__()
        self.feature = feature
        self.n_stack = n_stack
        self.num_out = len(out_channel)
        self.out_ch = out_channel
        self.out_activation = out_activation
        self.__build__()

    def __build__(self):
        feature = self.feature
        self.conv7 = nn.Conv2d(3, feature, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(feature, feature, 3, stride=1, padding=1)

        self.hours = [Hourglass(feature, feature) for _ in range(self.n_stack)]

        self.bottles_1 = [BottleNeckBlock(feature, feature) for _ in range(self.n_stack)]
        self.bottles_2 = [BottleNeckBlock(feature, feature) for _ in range(self.n_stack)]

        self.out_front = [nn.Conv2d(feature, feature, 3, stride=1, padding=1) for _ in range(self.num_out)]
        self.out = [nn.Conv2d(feature, self.out_ch[i], 3, stride=1, padding=1) for i in range(self.num_out)]
        self.out_batch = [nn.BatchNorm2d(feature) for _ in range(self.num_out)]

        self.batch1 = nn.BatchNorm2d(feature)
        self.batch2 = nn.BatchNorm2d(feature)

        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        output = []
        x = torch.selu(self.batch1(self.conv7(x)))
        x = self.max_pool(torch.selu(self.batch2(self.conv3(x))))

        for i in range(self.n_stack):
            x = self.hours[i](x)
            x = self.bottles_1[i](x)
            x = self.bottles_2[i](x)

        for i in range(self.num_out):
            out = torch.selu(self.out_batch[i](self.out_front[i](x)))
            out = self.out_activation[i](self.out[i](out))
            output.append(out)

        return output

    def inference(self, x):
        self(x)

    def cuda_adopt(self):
        for i in range(self.n_stack):
            self.hours[i] = self.hours[i].cuda()
            self.bottles_1[i] = self.bottles_1[i].cuda()
            self.bottles_2[i] = self.bottles_2[i].cuda()

        for i in range(self.num_out):
            self.out_batch[i] = self.out_batch[i].cuda()
            self.out_front[i] = self.out_front[i].cuda()
            self.out[i] = self.out[i].cuda()

        self = self.cuda()


def center_loss(output, target):
    o_heat = output[0]
    o_limb = output[1]
    t_heat = target[0]
    t_limb = target[1]
    return focal_loss(o_limb, t_limb) + focal_loss(o_heat, t_heat)


class CenterNet2(nn.Module):

    def __init__(self, feature, output):
        super(CenterNet2, self).__init__()
        self.feature = feature
        self.output = output

        self.__build__()

    def __build__(self):
        feature = self.feature
        self.conv7 = nn.Conv2d(3, feature, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(feature, feature, 3, stride=1, padding=1)
        self.hour1 = Hourglass(feature, feature)

        self.hour2 = Hourglass(feature, feature)
        self.res1 = BottleNeckBlock(feature, feature, True)
        self.res2 = BottleNeckBlock(feature, feature, True)
        self.res3 = BottleNeckBlock(feature, feature, True)
        self.heat_last_f = nn.Conv2d(feature, feature, 3, stride=1, padding=1)
        self.heat_last = nn.Conv2d(feature, self.output[0], 1, stride=1, padding=0)

        self.size_last_f = nn.Conv2d(feature, feature, 3, stride=1, padding=1)
        self.size_last = nn.Conv2d(feature, self.output[1], 1, stride=1, padding=0)
        self.batch1 = nn.BatchNorm2d(feature)
        self.batch2 = nn.BatchNorm2d(feature)
        self.batch3 = nn.BatchNorm2d(feature)
        self.batch4 = nn.BatchNorm2d(feature)

        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = torch.selu(self.batch1(self.conv7(x)))
        x = self.max_pool(torch.selu(self.batch2(self.conv3(x))))
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


if __name__ == "__main__":
    trn_dataset = datasets.MNIST('../mnist_data/',
                                 download=True,
                                 train=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),  # image to Tensor
                                     transforms.Normalize((0.1307,), (0.3081,))  # image, label
                                 ]))
    output_channel = [16,15]
    output_activation = [torch.sigmoid, torch.sigmoid]
    net = Test()
    if torch.cuda.is_available():
        net.cuda_adopt()
    net_total_params = sum(p.numel() for p in net.parameters())
    rd = torch.randint(0, 255, [4, 3, 64, 64]).type(torch.FloatTensor).cuda()

    labels = torch.tensor([1, 2, 3, 5])
    one_hot = np.array(labels)

    one_hot = torch.from_numpy(one_hot).type(torch.LongTensor).cuda()

    lr = 1e-4

    optim = torch.optim.Adam(net.parameters(), lr)

    criterion = torch.nn.CrossEntropyLoss()

    i = 0

    batch_size = 64
    trn_loader = torch.utils.data.DataLoader(trn_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)
    while True:
        for i, data in enumerate(trn_loader):
            x, label = data
            optim.zero_grad()
            x = x.cuda()
            label = label.cuda()

    # criterion = center_loss

            out = net(x)

            losses = criterion(out, label)
            losses.backward()
            optim.step()

            with torch.no_grad():
                result = out.cpu().numpy()
                t = [np.argmax(result[i]) for i in range(result.shape[0])]
                print('==================================={0}===================================='.format(i))
                print(np.array(t))
                label = label.cpu().numpy()
            # t = [np.argmax(label[i]) for i in range(4)]
                print(label)
                print(np.sum(np.equal(np.array(t), label)))
