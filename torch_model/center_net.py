from torch_model.layers import BottleNeckBlock, Hourglass
import torch
import torch.nn as nn
import torch.functional as F


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
            self.out[i] = self.out_batch[i].cuda()
            self.out_front[i] = self.out_batch[i].cuda()

        self = self.cuda()


if __name__ == "__main__":
    output_channel = [16,15]
    output_activation = [torch.sigmoid, torch.sigmoid]
    net = CenterNet(256, output_channel, output_activation)
    if torch.cuda.is_available():
        net.cuda_adopt()
    net_total_params = sum(p.numel() for p in net.parameters())
    rd = torch.randint(0, 255, [4, 3, 256, 256]).type(torch.FloatTensor).cuda()
    out = net(rd)
    with torch.no_grad():
        print(out[0].shape, out[1].shape)
