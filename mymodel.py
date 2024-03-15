import torch.nn as nn
import torch
import torch.nn.functional as F
from groupdcnplus import DeformConv2d
from PixelChannelSE import SE_Block


class myBlock(nn.Module):
    def __init__(self, channel, upchannel=False):
        super(myBlock, self).__init__()
        self.q1 = nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=(1, 1))
        self.q2 = nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=(1, 1))
        self.k1 = nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=(1, 1))
        self.br1 = DeformConv2d(channel, channel, kernel_size1=3, kernel_size2=7, groups=8)
        self.k2 = nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=(1, 1))
        self.br2 = DeformConv2d(channel, channel, kernel_size1=3, kernel_size2=7, groups=8)
        self.v1 = nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=(1, 1))
        self.v2 = nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=(1, 1))
        self.SE = SE_Block(inchannel=channel)
        self.upchannel = upchannel
        self.after_block1 = nn.Sequential(
            nn.Conv2d(channel, 2 * channel, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(2 * channel), nn.ReLU(inplace=True))
        self.after_block2 = nn.Sequential(
            nn.Conv2d(channel, 2 * channel, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(2 * channel), nn.ReLU(inplace=True))
        self.after_block3 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=(1, 1)),
                                          nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.after_block4 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=(1, 1)),
                                          nn.BatchNorm2d(channel), nn.ReLU(inplace=True))
        self.softmax = nn.Softmax2d()

    def forward(self, x1, x2):
        q1 = self.q1(x1)
        q2 = self.q2(x2)
        k1 = self.k1(x1)
        k1 = self.br1(k1)
        k2 = self.k2(x2)
        k2 = self.br2(k2)
        v1 = self.v1(x1)
        v2 = self.v2(x2)
        att1 = torch.matmul(q1, k2.permute(0, 1, 3, 2))
        att1 = att1 / torch.std(att1)
        out1 = torch.matmul(v1, self.softmax(att1))
        att2 = torch.matmul(q2, k1.permute(0, 1, 3, 2))
        att2 = att2 / torch.std(att2)
        out2 = torch.matmul(v2, self.softmax(att2))
        out1 = self.SE(out1) + x1
        out2 = self.SE(out2) + x2
        if self.upchannel:
            out1 = self.after_block1(out1)
            out2 = self.after_block2(out2)
        else:
            out1 = self.after_block3(out1)
            out2 = self.after_block4(out2)

        return out1, out2


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.stem1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.stem2 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True))

        # self.stem_pool1 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        # elf.stem_pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.block1 = myBlock(channel=16, upchannel=True)
        # self.maxpool11 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        # self.maxpool12 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.block2 = myBlock(channel=32, upchannel=True)
        # self.maxpool21 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # self.maxpool22 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        #self.block3 = myBlock(channel=64, upchannel=False)
        # self.block4 = myBlock(channel=128)
        self.pool1 = nn.AdaptiveAvgPool2d((4, 4))
        self.pool2 = nn.AdaptiveAvgPool2d((4, 4))
        self.fc11 = nn.Linear(1024, 512)
        self.fc12 = nn.Linear(1024, 512)
        self.fc21 = nn.Linear(512, 128)
        self.fc22 = nn.Linear(512, 128)
        self.drop11 = nn.Dropout(0.35)
        self.drop12 = nn.Dropout(0.35)
        self.drop21 = nn.Dropout(0.5)
        self.drop22 = nn.Dropout(0.5)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(256, 256)
        self.out = nn.Linear(256, 1)

    def forward(self, x1, x2):
        x1 = self.stem1(x1)
        x2 = self.stem2(x2)
        # x1 = self.stem_pool1(F.relu(x1))
        # x2 = self.stem_pool2(F.relu(x2))
        x1, x2 = self.block1(x1, x2)
        x1, x2 = self.block2(x1, x2)
        #x1, x2 = self.block3(x1, x2)
        # x1,x2 = self.block4(x1,x2)
        x1 = self.pool1(x1)
        x2 = self.pool2(x2)
        x1 = self.drop11(F.relu(self.fc11(self.flat(x1))))
        x2 = self.drop21(F.relu(self.fc12(self.flat(x2))))
        x1 = self.drop21(F.relu(self.fc21(x1)))
        x2 = self.drop22(F.relu(self.fc22(x2)))
        x = torch.cat((x1, x2), 1)
        x = torch.squeeze(self.out(self.fc(x)), dim=1)

        return x


if __name__ == "__main__":
    x1 = torch.randn(2, 3, 32, 32)
    x2 = torch.randn(2, 3, 32, 32)
    model = Net()
    out = model(x1, x2)
    print(out.size(), out)

