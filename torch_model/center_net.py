from torch_model.layers import BottleNeckBlock, Hourglass
import torch
import torch.nn as nn


class CenterNet(nn.Module):

    def __init__(self, feature, out_channel, out_activation, n_layer=5, n_stack=2):
        super(CenterNet, self).__init__()
        self.feature = feature
        self.n_layer = n_layer
        self.n_stack = n_stack
        self.out_ch = out_channel
        self.out_activation = out_activation
        self.__build__()

    def __build__(self):
        feature = self.feature
        n_s = self.n_stack
        n_o = len(self.out_ch)
        self.conv7 = nn.Conv2d(3, feature, 7, stride=2, padding=3)

        self.block1 = BottleNeckBlock(feature, feature)
        self.block2 = BottleNeckBlock(feature, feature)
        self.block3 = BottleNeckBlock(feature, feature)
        self.block4 = BottleNeckBlock(feature, feature)

        self.hours = nn.ModuleList([Hourglass(feature, feature, self.n_layer) for _ in range(n_s)])

        self.bottles_1 = nn.ModuleList([BottleNeckBlock(feature, feature) for _ in range(n_s-1)])
        self.bottles_2 = nn.ModuleList([BottleNeckBlock(feature, feature) for _ in range(n_s-1)])

        self.inter_h_o = nn.ModuleList([nn.Conv2d(feature, self.out_ch[0], 1, stride=1, padding=0) for _ in range(n_s - 1)])
        self.inter_l_o = nn.ModuleList([nn.Conv2d(feature, self.out_ch[1], 1, stride=1, padding=0) for _ in range(n_s - 1)])

        self.inter_h_a = nn.ModuleList([nn.Conv2d(self.out_ch[0], feature, 1, stride=1, padding=0) for _ in range(n_s - 1)])
        self.inter_l_a = nn.ModuleList([nn.Conv2d(self.out_ch[1], feature, 1, stride=1, padding=0) for _ in range(n_s - 1)])

        self.out_h_f = nn.Conv2d(feature, feature, 3, stride=1, padding=1)
        self.out_l_f = nn.Conv2d(feature, feature, 3, stride=1, padding=1)
        self.out_h = nn.Conv2d(feature, self.out_ch[0], 1, stride=1, padding=0)
        self.out_l = nn.Conv2d(feature, self.out_ch[1], 1, stride=1, padding=0)
        self.out_h_b = nn.BatchNorm2d(feature)
        self.out_l_b = nn.BatchNorm2d(feature)

        self.batch1 = nn.BatchNorm2d(feature)
        self.batch2 = nn.BatchNorm2d(feature)

        self.batch3 = nn.BatchNorm2d(feature)
        self.batch4 = nn.BatchNorm2d(feature)

        self.max_pool = nn.MaxPool2d(2, stride=2, padding=0)

    def forward(self, x):
        out_put = []
        x = torch.relu(self.batch1(self.conv7(x)))
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
            inter = i_h + i_l
            x = x + init + inter

        last_hour = self.hours[-1](x)

        out_block = self.block4(last_hour)
        h_f = torch.relu(self.batch3(self.out_h_f(out_block)))
        l_f = torch.relu(self.batch4(self.out_l_f(out_block)))
        out_h = self.out_activation[0](self.out_h(h_f))
        out_l = self.out_activation[1](self.out_l(l_f))

        out_put.append([out_h, out_l])

        return out_put

    def info(self):
        total_params = sum(p.numel() for p in self.parameters())
        print("{0:^40s}".format('CenterNet Information'))
        print("{0:^40s}".format("{0:22s}: {1:10d}".format('hourglass repeat', self.n_stack)))
        print("{0:^40s}".format("{0:22s}: {1:10d}".format('hourglass layer count', self.n_layer)))
        print("{0:^40s}".format("{0:22s}: {1:10,d}".format('total parameter', total_params)))


if __name__ == "__main__":
    net = CenterNet(256, [17,16], [torch.relu, torch.relu], n_layer=2, n_stack=4)
    lr = 1e-4

    optim = torch.optim.Adam(net.parameters(), lr)

    pytorch_total_params = sum(p.numel() for p in net.parameters())

    print(pytorch_total_params)

    input = torch.zeros((1,3,256,256))
    criterion = torch.nn.CrossEntropyLoss()
    output = net(input)
    print(len(output), output[-1][1].shape)

