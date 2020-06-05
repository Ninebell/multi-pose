import torch.nn as nn
import torch.nn.functional as F
import torch


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


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
        # self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        self.sp_w0 = nn.Conv2d(2, 1, 7, stride=1, padding=6//2)

        # self.sp_w0 = nn.Conv2d(feature*2, feature, 3, stride=1, padding=1)
        # self.sp_w1 = nn.Conv2d(feature, feature, 3, stride=1, padding=1)
        # self.avg_pool = nn.AvgPool2d(3, stride=1, padding=1)
        # self.max_pool = nn.MaxPool2d(3, stride=1, padding=1)

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
        scale = F.sigmoid(x_out)  # broadcasting
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
    def __init__(self, input_feature, output_feature, attention=False, ratio=8, activation=torch.selu):
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
        x = self.activation(self.batch1(self.c1(x)))
        x = self.activation(self.batch2(self.c2(x)))
        x = self.activation(self.batch3(self.c3(x)))
        if self.attention:
            x = self.attention.forward(x)

        if self.input_feature != self.output_feature:
            init = self.activation(self.c4(init))
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

        self.down1 = BottleNeckBlock(i_f, o_f, True)
        self.down2 = BottleNeckBlock(o_f, o_f, True)
        self.down3 = BottleNeckBlock(o_f, o_f, True)
        self.down4 = BottleNeckBlock(o_f, o_f, True)
        self.down5 = BottleNeckBlock(o_f, o_f, True)

        self.skip1 = BottleNeckBlock(o_f, o_f, True)
        self.skip2 = BottleNeckBlock(o_f, o_f, True)
        self.skip3 = BottleNeckBlock(o_f, o_f, True)
        self.skip4 = BottleNeckBlock(o_f, o_f, True)

        self.middle1 = BottleNeckBlock(o_f, o_f, True)
        self.middle2 = BottleNeckBlock(o_f, o_f, True)
        self.middle3 = BottleNeckBlock(o_f, o_f, True)

        self.up1 = BottleNeckBlock(i_f, o_f, True)
        self.up2 = BottleNeckBlock(o_f, o_f, True)
        self.up3 = BottleNeckBlock(o_f, o_f, True)
        self.up4 = BottleNeckBlock(o_f, o_f, True)
        self.up5 = BottleNeckBlock(o_f, o_f, True)

        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        down1 = self.down1(x)
        skip1 = self.skip1(down1)
        down1 = self.max_pool(down1)

        down2 = self.down2(down1)
        skip2 = self.skip2(down2)
        down2 = self.max_pool(down2)

        down3 = self.down3(down2)
        skip3 = self.skip3(down3)
        down3 = self.max_pool(down3)

        down4 = self.down4(down3)
        skip4 = self.skip4(down4)
        down4 = self.max_pool(down4)

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
