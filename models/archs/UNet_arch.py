import functools
import paddle.nn as nn
import paddle.nn.functional as F
import models.archs.arch_util as arch_util


class HDRUNet(nn.Layer):

    def __init__(self, in_nc=3, out_nc=3, nf=64, act_type='relu'):
        super(HDRUNet, self).__init__()

        self.conv_first = nn.Conv2D(in_nc, nf, 3, 1, 1)
        
        self.SFT_layer1 = arch_util.SFTLayer()
        self.HR_conv1 = nn.Conv2D(nf, nf, 3, 1, 1, bias_attr=True)

        self.down_conv1 = nn.Conv2D(nf, nf, 3, 2, 1)
        self.down_conv2 = nn.Conv2D(nf, nf, 3, 2, 1)
        
        basic_block = functools.partial(arch_util.ResBlock_with_SFT, nf=nf)
        self.recon_trunk1 = arch_util.make_layer(basic_block, 2)
        self.recon_trunk2 = arch_util.make_layer(basic_block, 8)
        self.recon_trunk3 = arch_util.make_layer(basic_block, 2)

        self.up_conv1 = nn.Sequential(nn.Conv2D(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))
        self.up_conv2 = nn.Sequential(nn.Conv2D(nf, nf*4, 3, 1, 1), nn.PixelShuffle(2))

        self.SFT_layer2 = arch_util.SFTLayer()
        self.HR_conv2 = nn.Conv2D(nf, nf, 3, 1, 1, bias_attr=True)
        self.conv_last = nn.Conv2D(nf, out_nc, 3, 1, 1, bias_attr=True)

        cond_in_nc=3
        cond_nf=64
        self.cond_first = nn.Sequential(nn.Conv2D(cond_in_nc, cond_nf, 3, 1, 1), nn.LeakyReLU(0.1), 
                                        nn.Conv2D(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1), 
                                        nn.Conv2D(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1))
        self.CondNet1 = nn.Sequential(nn.Conv2D(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1), nn.Conv2D(cond_nf, 32, 1))
        self.CondNet2 = nn.Sequential(nn.Conv2D(cond_nf, cond_nf, 3, 2, 1), nn.LeakyReLU(0.1), nn.Conv2D(cond_nf, 32, 1))
        self.CondNet3 = nn.Sequential(nn.Conv2D(cond_nf, cond_nf, 3, 2, 1), nn.LeakyReLU(0.1), nn.Conv2D(cond_nf, 32, 3, 2, 1))

        self.mask_est = nn.Sequential(nn.Conv2D(in_nc, nf, 3, 1, 1), 
                                      nn.ReLU(), 
                                      nn.Conv2D(nf, nf, 3, 1, 1),
                                      nn.ReLU(), 
                                      nn.Conv2D(nf, nf, 1),
                                      nn.ReLU(), 
                                      nn.Conv2D(nf, out_nc, 1),
                                     )

        # activation function
        if act_type == 'relu':
            self.act = nn.ReLU()
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        # x[0]: img; x[1]: cond
        mask = self.mask_est(x[0])

        cond = self.cond_first(x[1])   
        cond1 = self.CondNet1(cond)
        cond2 = self.CondNet2(cond)
        cond3 = self.CondNet3(cond)

        fea0 = self.act(self.conv_first(x[0]))

        fea0 = self.SFT_layer1((fea0, cond1))
        fea0 = self.act(self.HR_conv1(fea0))

        fea1 = self.act(self.down_conv1(fea0))
        fea1, _ = self.recon_trunk1((fea1, cond2))

        fea2 = self.act(self.down_conv2(fea1))
        out, _ = self.recon_trunk2((fea2, cond3))
        out = out + fea2

        out = self.act(self.up_conv1(out)) + fea1
        out, _ = self.recon_trunk3((out, cond2))

        out = self.act(self.up_conv2(out)) + fea0
        out = self.SFT_layer2((out, cond1))
        out = self.act(self.HR_conv2(out))

        out = self.conv_last(out)
        out = mask * x[0] + out
        return out