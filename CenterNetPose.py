from torch_module.layers import BottleNeckBlock, Hourglass, activation_layer, Conv2D
import torch
import json
import torch.nn as nn


class CenterNetPose(nn.Module):

    def __init__(self, feature, out_channel, out_activation, n_layer=5, n_stack=2):
        super(CenterNetPose, self).__init__()
        self.feature = feature
        self.n_layer = n_layer
        self.n_stack = n_stack
        self.out_ch = out_channel
        self.out_act_str = out_activation
        self.out_activation = []
        for act in self.out_act_str:
            if type(act) == str:
                self.out_activation.append(activation_layer[act])
            else:
                self.out_activation.append(act)

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
        self.conv7 = Conv2D(3, feature, 7, 2, 3, torch.relu, batch=True)

        self.block1 = BottleNeckBlock(feature, feature)
        self.block2 = BottleNeckBlock(feature, feature)
        self.block3 = BottleNeckBlock(feature, feature)
        self.block4 = BottleNeckBlock(feature, feature)

        self.hours = nn.ModuleList([Hourglass(feature, self.n_layer) for _ in range(n_s)])

        self.bottles_1 = nn.ModuleList([BottleNeckBlock(feature, feature) for _ in range(n_s-1)])
        self.bottles_2 = nn.ModuleList([BottleNeckBlock(feature, feature) for _ in range(n_s-1)])

        self.inter_h_o = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature, self.out_ch[0], 1, stride=1, padding=0),
            ) for _ in range(n_s - 1)])
        self.inter_l_o = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature, self.out_ch[1], 1, stride=1, padding=0),
            ) for _ in range(n_s - 1)])

        self.inter_h_a = nn.ModuleList([Conv2D(self.out_ch[0], feature, 1, 1, 0, torch.relu, batch=True) for _ in range(n_s - 1)])
        self.inter_l_a = nn.ModuleList([Conv2D(self.out_ch[1], feature, 1, 1, 0, torch.relu, batch=True) for _ in range(n_s - 1)])

        self.out_h_f = Conv2D(feature, feature, 3, stride=1, padding=1, activation=torch.relu, batch=True)
        self.out_l_f = Conv2D(feature, feature, 3, stride=1, padding=1, activation=torch.relu, batch=True)

        self.out_h = nn.Conv2d(feature, self.out_ch[0], 1, stride=1, padding=0)
        self.out_l = nn.Conv2d(feature, self.out_ch[1], 1, stride=1, padding=0)

        self.batch3 = nn.BatchNorm2d(feature)

        self.max_pool = nn.MaxPool2d(2, stride=2, padding=0)

    def forward(self, x):
        out_put = []
        x = self.conv7(x)
        x = self.max_pool(self.block1(x))
        x = self.block2(x)
        x = self.block3(x)
        # org = x
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
            # x = x + org + inter

        last_hour = self.hours[-1](x)

        out_block = torch.selu(self.batch3(self.block4(last_hour)))
        h_f = self.out_h_f(out_block)
        l_f = self.out_l_f(out_block)
        out_h = self.out_activation[0](self.out_h(h_f))
        out_l = self.out_activation[1](self.out_l(l_f))

        out_put.append([out_h, out_l])

        return out_put


if __name__ == "__main__":
    net = CenterNetPose(128, [17, 16], [torch.selu, torch.selu], n_layer=4, n_stack=2)
    lr = 1e-4

    optim = torch.optim.Adam(net.parameters(), lr)
    pytorch_total_params = sum(p.numel() for p in net.parameters())

    print(pytorch_total_params)

    input = torch.zeros((3, 3, 256, 256))
    criterion = torch.nn.CrossEntropyLoss()
    output = net(input)
    print(len(output))
    print(output[0][1].shape)
