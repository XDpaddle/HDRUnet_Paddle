import functools
import paddle.nn as nn
import models.archs.arch_util as arch_util

class SRResNet(nn.Layer):
    ''' modified SRResNet for ITM'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, act_type='relu'):
        super(SRResNet, self).__init__()

        self.conv_first = nn.Conv2D(in_nc, nf, 3, 2, 1, bias_attr=True)
        basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.recon_trunk = arch_util.make_layer(basic_block, nb)

        self.upconv = nn.Conv2D(nf, nf*4, 3, 1, 1, bias_attr=True)
        self.upsampler = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2D(nf, nf, 3, 1, 1, bias_attr=True)
        self.conv_last = nn.Conv2D(nf, out_nc, 3, 1, 1, bias_attr=True)

        # activation function
        if act_type == 'relu':
            self.act = nn.ReLU()
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1)
        
        # initialization
        arch_util.initialize_weights([self.conv_first, self.upconv, self.HRconv, self.conv_last],
                                     0.1)

    def forward(self, x):
        fea = self.act(self.conv_first(x))
        out = self.recon_trunk(fea)
        out = self.act(self.upsampler(self.upconv(out)))
        out = self.conv_last(self.act(self.HRconv(out)))
        return out