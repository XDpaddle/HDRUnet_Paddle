import math
from numpy import mod
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import models.archs.initalize as init

#from .dynamic_conv import Dynamic_conv2d

#def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#    return Dynamic_conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=True, dilation=dilation)

# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

#def conv1x1(in_planes, out_planes, stride=1):
#    return Dynamic_conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.sublayers():
            if isinstance(m, nn.Conv2D):
                # init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                init.kaiming_uniform_(m.weight, a=0, mode='fan_in')
                # m.weight.data *= scale  # for residual block
                m.weight.set_value(scale*m.weight)
                if m.bias is not None:
                    init.constant_(m.bias,value=0.)
            elif isinstance(m, nn.Linear):
                # init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                init.kaiming_uniform_(m.weight, a=0, mode='fan_in')
                # m.weight.data *= scale  # for residual block
                m.weight.set_value(scale*m.weight)
                if m.bias is not None:
                    init.constant_(m.bias,value=0.)
            elif isinstance(m, nn.BatchNorm2D):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
                                  
class SFTLayer(nn.Layer):
    def __init__(self, in_nc=32, out_nc=64, nf=32):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2D(in_nc, nf, 1)
        self.SFT_scale_conv1 = nn.Conv2D(nf, out_nc, 1)
        self.SFT_shift_conv0 = nn.Conv2D(in_nc, nf, 1)
        self.SFT_shift_conv1 = nn.Conv2D(nf, out_nc, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        return x[0] * (scale + 1) + shift


class ResBlock_with_SFT(nn.Layer):
    def __init__(self, nf=64):
        super(ResBlock_with_SFT, self).__init__()
        self.conv1 = nn.Conv2D(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2D(nf, nf, 3, 1, 1, bias=True)

        self.sft1 = SFTLayer(in_nc=32, out_nc=64, nf=32)
        self.conv1 = nn.Conv2D(nf, nf, 3, 1, 1)
        self.sft2 = SFTLayer(in_nc=32, out_nc=64, nf=32)
        self.conv2 = nn.Conv2D(nf, nf, 3, 1, 1)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        fea = self.sft1(x)
        fea = F.relu(self.conv1(fea), inplace=True)
        fea = self.sft2((fea, x[1]))
        fea = self.conv2(fea)
        return (x[0] + fea, x[1])
    
    
def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Layer):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2D(nf, nf, 3, 1, 1, bias_attr=True)
        self.conv2 = nn.Conv2D(nf, nf, 3, 1, 1, bias_attr=True)
        init.kaiming_normal_(self.conv1.weight, a=0, mode='fan_in')
        init.constant_(self.conv1.bias, value=0.)
        init.kaiming_normal_(self.conv2.weight, a=0, mode='fan_in')
        init.constant_(self.conv2.bias, value=0.)
        # initialization
        #initialize_weights([self.conv1, self.conv2], scale=0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out

class ResidualBlockNoBN(nn.Layer):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2D(num_feat, num_feat, 3, 1, 1, bias_attr=True)
        self.conv2 = nn.Conv2D(num_feat, num_feat, 3, 1, 1, bias_attr=True)
        self.relu = nn.ReLU()

        if not pytorch_init:
            initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.shape[-2:] == flow.shape[1:3]
    B, C, H, W = x.shape
    # mesh grid
    grid_y, grid_x = paddle.meshgrid(paddle.arange(0, H), paddle.arange(0, W))
    grid = paddle.stack((grid_x, grid_y), 2).astype('float32') # W(x), H(y), 2
    # grid.requires_grad = False
    grid.stop_gradient = True
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = paddle.stack((vgrid_x, vgrid_y), axis=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output



# rcan arch
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2D(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias_attr=bias)

class MeanShift(nn.Conv2D):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = paddle.to_tensor(rgb_std)
        # self.weight.data = paddle.eye(3).view(3, 3, 1, 1)
        # self.weight.data.div_(std.view(3, 1, 1, 1))
        # self.bias.data = sign * rgb_range * paddle.Tensor(rgb_mean)
        # self.bias.data.div_(std)
        self.weight.set_value(paddle.divide(paddle.eye(3).reshape([3, 3, 1, 1]), std.reshape([3, 1, 1, 1])))
        self.bias.set_value(paddle.divide(sign * rgb_range * paddle.to_tensor(rgb_mean), std))
        # self.requires_grad = False
        self.stop_gradient = True

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2D(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias_attr=bias)
        ]
        if bn: m.append(nn.BatchNorm2D(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Layer):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2D(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2D(n_feat))
                if act: m.append(act)
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2D(n_feat))
            if act: m.append(act)
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class EResidualBlock(nn.Layer):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(EResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(),
            nn.Conv2D(out_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(),
            nn.Conv2D(out_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2D(n_feat))
                if act: m.append(act)
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2D(n_feat))
            if act: m.append(act)
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class UpsampleBlock(nn.Layer):
    def __init__(self,
                 n_channels, scale, multi_scale,
                 group=1):
        super(UpsampleBlock, self).__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, group=group)
        else:
            self.up = _UpsampleBlock(n_channels, scale=scale, group=group)

        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)


class _UpsampleBlock(nn.Layer):
    def __init__(self,
                 n_channels, scale,
                 group=1):
        super(_UpsampleBlock, self).__init__()

        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [nn.Conv2D(n_channels, 4 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [nn.Conv2D(n_channels, 9 * n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        out = self.body(x)
        return out


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.shape[-2:] == flow.shape[1:3]
    B, C, H, W = x.shape
    # mesh grid
    grid_y, grid_x = paddle.meshgrid(paddle.arange(0, H), paddle.arange(0, W))
    grid = paddle.stack((grid_x, grid_y), 2).astype('float32')  # W(x), H(y), 2
    # grid.requires_grad = False
    grid.stop_gradient = True
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = paddle.stack((vgrid_x, vgrid_y), axis=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output


