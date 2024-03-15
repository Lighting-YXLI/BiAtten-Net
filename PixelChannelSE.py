import torch.nn as nn
import torch
from torchstat import stat
#from groupdcnplus import *

class shuffle_channels(nn.Module):
    def __init__(self, inc, groups=2):
        super(shuffle_channels, self).__init__()
        self.groups = groups
        self.linear = nn.Linear(int(groups),int(groups))
        self.conv = nn.Conv2d(groups,groups,kernel_size=1)
    def forward(self,x):
        batch_size, channels, height, width = x.size()
        assert channels % self.groups == 0
        channels_per_group = channels // self.groups
        # split into groups
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
       # transpose 1, 2 axis
        x = self.linear(x.permute(0,2,3,4,1).contiguous()).permute(0,1,4,2,3).contiguous()
        #x = self.conv(x).view(batch_size, channels_per_group,self.groups, height, width)
        #x = x.transpose(1, 2).contiguous()
        # reshape into orignal
        x = x.view(batch_size, channels, height, width)
        return x

# 全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.group=inchannel//ratio
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        #self.fc = nn.Sequential(
        #    nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
        #    nn.ReLU(),
        #    nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
        #    nn.Sigmoid()
        #)
        self.upscale_factor = 2
        self.subchannel_before = nn.Sequential(
            nn.Conv2d(inchannel,inchannel*(self.upscale_factor), kernel_size=1,groups=2),
            nn.Tanh()
        )
        self.subchannel_after = nn.Sequential(
            nn.Conv2d(inchannel*(self.upscale_factor), inchannel, kernel_size=1)
        )
        self.subchannel = nn.Sequential(
            nn.Conv2d(inchannel, inchannel // ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(inchannel // ratio, inchannel , kernel_size=1,groups=inchannel // ratio),

        )
        self.shuffle = shuffle_channels(groups=(self.upscale_factor),inc=inchannel*(self.upscale_factor))
        #self.gcs = nn.Conv2d(inchannel,1,kernel_size=(3,3),padding=(1,1))
        self.pixelse = nn.Sequential(
            #nn.Conv2d(inchannel, 64, (5, 5), (1, 1), (2, 2)), #c=1 -> c=64
            #nn.Tanh(),
            nn.Conv2d(inchannel, 32, (3, 3), (1, 1), (1, 1)),
            nn.Tanh(),
            nn.Conv2d(32, 1 * (self.upscale_factor ** 2), (3, 3), (1, 1), (1, 1)),
            #DeformConv2d(in_dim=32, out_dim=1 * (self.upscale_factor ** 2), kernel_size1=3, kernel_size2=7, groups=2, offset_groups=1, with_mask=True),
            nn.PixelShuffle(self.upscale_factor),
            nn.Tanh(),
            nn.Conv2d(1,1,kernel_size=(3,3),padding=(1,1),stride=(self.upscale_factor,self.upscale_factor)),
            #nn.Conv2d(1,1,kernel_size=(self.upscale_factor,self.upscale_factor),padding=(0,0),stride=(self.upscale_factor,self.upscale_factor)),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).view(b, c,1,1)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        #y = shuffle_channels(self.subchannel(y),groups=self.group)
        y = self.subchannel_after(self.shuffle(self.subchannel_before(y)))
        sigmoid = nn.Sigmoid()
        y = sigmoid(y)
        # Fscale操作：将得到的权重乘以原来的特征图x
        #z = self.gcs(x).view(b,1,h,w)
        z = self.pixelse(x).view(b,1,h,w)
        return  x * z.expand_as(x) * y.expand_as(x)

if __name__ == "__main__":
    x = torch.randn(32,32,224,224)
    pixel_channel_se = SE_Block(inchannel=32)
    out = pixel_channel_se(x)
    stat(pixel_channel_se, (32, 224, 224))