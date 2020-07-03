import torch.nn as nn
import torch.nn.functional as F
import torch


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class AttentionBlock(nn.Module):
    def __init__(self, feature, ratio):
        super(AttentionBlock, self).__init__()
        self.__build__(feature, ratio)

    def __build__(self,feature, ratio):
        self.w0 = nn.Linear(feature, feature//ratio)
        self.w1 = nn.Linear(feature//ratio, feature)
        self.g_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.g_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.compress = ChannelPool()
        self.sp_w0 = nn.Conv2d(2, 1, 7, stride=1, padding=3)

    def __channel_forward__(self, x):
        def __chanel_attention__(ch_input):
            temp = torch.flatten(ch_input, start_dim=1)
            temp = torch.relu(self.w0(temp))
            temp = self.w1(temp)
            return temp

        ch_avg = self.g_avg_pool(x)
        ch_avg = __chanel_attention__(ch_avg)
        ch_max = self.g_max_pool(x)
        ch_max = __chanel_attention__(ch_max)
        ch_attention = torch.sigmoid(ch_max + ch_avg)
        return ch_attention

    def __spatial_forward__(self, x):
        x_compress = self.compress(x)
        x_out = self.sp_w0(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale

        return sp_attention

    def forward(self, x):
        init = x
        ch_attention = self.__channel_forward__(x)

        # reshape n, f => n, f, 1, 1
        ch_attention = ch_attention.view((ch_attention.shape[0], ch_attention.shape[1], 1, 1))

        x = ch_attention*init
        sp_attention = self.__spatial_forward__(x)

        x = x * sp_attention
        return x


class BottleNeckBlock(nn.Module):
    def __init__(self, input_feature, output_feature, attention=False, ratio=16, activation=torch.relu):
        super(BottleNeckBlock, self).__init__()
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.attention = attention
        self.ratio = ratio
        self.activation = activation
        self.__build__()

    def __build__(self):
        self.c1 = nn.Conv2d(self.input_feature, self.output_feature, 1, stride=1, padding=0)
        self.c2 = nn.Conv2d(self.output_feature, self.output_feature, 3, stride=1, padding=1)
        self.c3 = nn.Conv2d(self.output_feature, self.output_feature, 1, stride=1, padding=0)

        if self.input_feature != self.output_feature:
            self.c4 = nn.Conv2d(self.input_feature, self.output_feature, 3, stride=1, padding=1)

        if self.attention:
            self.attention = AttentionBlock(self.output_feature, self.ratio)

        self.batch1 = nn.BatchNorm2d(self.output_feature)
        self.batch2 = nn.BatchNorm2d(self.output_feature)
        self.batch3 = nn.BatchNorm2d(self.output_feature)

    def forward(self, x):
        init = x
        x = self.batch1(x)
        x = self.activation(x)
        x = self.c1(x)

        x = self.batch2(x)
        x = self.activation(x)
        x = self.c2(x)

        x = self.batch3(x)
        x = self.activation(x)
        x = self.c3(x)

        if self.attention:
            x = self.attention.forward(x)

        return x + init


class Hourglass(nn.Module):
    def __init__(self, input_feature, output_feature, layers, attention=True):
        super(Hourglass, self).__init__()
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.layers = layers
        self.attention = attention

        self.__build__()

    def __build__(self):
        i_f = self.input_feature
        o_f = self.output_feature

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.skips = nn.ModuleList()

        for i in range(self.layers):
            self.downs.append(BottleNeckBlock(o_f, o_f, self.attention) if i != 0
                              else BottleNeckBlock(i_f, o_f, self.attention))
            self.ups.append(BottleNeckBlock(o_f, o_f, self.attention))
            self.skips.append(BottleNeckBlock(o_f, o_f, self.attention))

        self.final_skip = BottleNeckBlock(o_f, o_f, self.attention)

        self.max_pool = nn.MaxPool2d(2, stride=2, padding=0)

    def forward(self, x):
        skips = []
        down = x
        for i in range(self.layers):
            down = self.downs[i](down)
            skip = self.skips[i](down)
            skips.append(skip)
            if i != self.layers-1:
                down = self.max_pool(down)

        skips[self.layers-1] = self.final_skip(skips[self.layers-1])

        for i in range(self.layers):
            if i == 0:
                up = self.ups[i](skips[self.layers-i-1])
            else:
                up = F.interpolate(up, scale_factor=2)
                up = up + skips[self.layers-i-1]
                up = self.ups[i](up)

        return up
