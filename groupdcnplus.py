import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d
#from torchsummary import summary
#from torchstat import stat

class DeformConv2d(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size1,
        kernel_size2,
        stride1=1,
        stride2=1,
        dilation1=1,
        dilation2=1,
        groups=1,
        bias=True,
        offset_groups=1,
        with_mask=False
    ):
        super(DeformConv2d,self).__init__()
        assert in_dim % groups == 0
        self.in_dim=in_dim
        self.mid_dim=in_dim*groups
        self.out_dim=out_dim
        self.stride1 = stride1
        self.padding1 = kernel_size1//2
        self.dilation1 = dilation1
        self.stride2 = stride2
        self.padding2 = kernel_size2//2
        self.dilation2 = dilation2
        self.groups = groups
        self.output_pj=nn.Linear(self.mid_dim,in_dim)
        self.weight1 = nn.Parameter(torch.empty(self.in_dim//groups, in_dim // groups, kernel_size1, kernel_size1))
        self.weight2 = nn.Parameter(torch.empty(self.in_dim // groups, in_dim // groups, kernel_size2, kernel_size2))
        self.pw = nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.in_dim//groups))
        else:
            self.bias = None

        self.with_mask = with_mask
        if with_mask:
            # batch_size, (2+1) * offset_groups * kernel_height * kernel_width, out_height, out_width
            self.param_generator1 = nn.Conv2d(in_dim // groups, 3 * offset_groups * kernel_size1 * kernel_size1, kernel_size=3, padding=1, stride=1)
            self.param_generator2 = nn.Conv2d(in_dim // groups, 3 * offset_groups * kernel_size2 * kernel_size2, kernel_size=3, padding=1, stride=1)
            #self.param_generator1 = nn.Linear(in_dim // groups, (3 * offset_groups * kernel_size1 * kernel_size1))
            #self.param_generator2 = nn.Linear(in_dim // groups, (3 * offset_groups * kernel_size2 * kernel_size2))


        else:
            self.param_generator1 = nn.Conv2d(in_dim//groups, 2 * offset_groups * kernel_size1 * kernel_size1, kernel_size=3, padding=1, stride=1)
            self.param_generator2 = nn.Conv2d(in_dim // groups, 2 * offset_groups * kernel_size2 * kernel_size2,kernel_size=3, padding=1, stride=1)
    def forward(self, x):
        #print(x.shape)
        x=list(torch.split(x,self.in_dim//self.groups,1))
        #print(x)
        #print('-----------')

        for idx in range(self.groups):

          if self.with_mask:
             if idx % 2 == 0:

                  oh, ow, mask = self.param_generator1(x[idx]).chunk(3, dim=1)
                  #oh, ow, mask = self.param_generator1(x[idx].permute(0,2,3,1)).permute(0,3,1,2).chunk(3, dim=1)
             else:
                  oh, ow, mask = self.param_generator2(x[idx]).chunk(3, dim=1)
                  #oh, ow, mask = self.param_generator2(x[idx].permute(0,2,3,1)).permute(0,3,1,2).chunk(3, dim=1)

             offset = torch.cat([oh, ow], dim=1)
             mask = mask.sigmoid()
          else:
              if idx % 2 == 0:
                  offset = self.param_generator1(x[idx])
              else:
                  offset = self.param_generator2(x[idx])
              mask = None
          if idx % 2 == 0:
              x[idx] = deform_conv2d(x[idx], offset=offset, weight=self.weight1, bias=self.bias, stride=(self.stride1,self.stride1), padding=(self.padding1,self.padding1), dilation=(self.dilation1,self.dilation1), mask=mask)
          else:
              x[idx] = deform_conv2d(x[idx], offset=offset, weight=self.weight2, bias=self.bias, stride=(self.stride2,self.stride2), padding=(self.padding2,self.padding2), dilation=(self.dilation2,self.dilation2), mask=mask)
          #print(x[idx].shape)
        out = torch.cat(x,dim=1)
        #print(out.shape)
        #print(out.permute(0,2,3,1).shape)
        #out = self.output_pj(out.permute(0,2,3,1)).permute(0,3,1,2)
        #print(out.shape)
        out= self.pw(out)
        #print(out.shape)
        return out




if __name__ == "__main__":

    #deformable_conv2d = DeformableConv2d(in_dim=3, out_dim=6, kernel_size=3,padding=1, offset_groups=3, with_mask=True)
    #print(deformable_conv2d(torch.randn(1, 3, 16, 16)).shape)
    #stat(deformable_conv2d, (3, 16, 16))
    print('------------------------------------------------')
    #deformable_conv2d1 = DeformConv2d(in_dim=9, out_dim=9, kernel_size=3,padding=1, groups=3, offset_groups=3, with_mask=True)
    deformable_conv2d1 = DeformConv2d(in_dim=9, out_dim=9, kernel_size1=3, kernel_size2=5, groups=3, offset_groups=3,
                                          with_mask=True)
    print(deformable_conv2d1)
    #deformable_conv2d1(torch.randn(1, 9, 16, 16))
    #stat(deformable_conv2d1, (9, 16, 16))
    #h = w = 3
    # batch_size, num_channels, out_height, out_width
    #x = torch.arange(h * w * 3, dtype=torch.float32).reshape(1, 3, h, w)
    #print(deformable_conv2d(x))
    #print(deformable_conv2d(x).shape)
    #print(deformable_conv2d(x))
    #print(deformable_conv2d(x).shape)
