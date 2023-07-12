import functools
import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class Condition(nn.Layer):
    def __init__(self, in_nc=3, nf=32):
        super(Condition, self).__init__()
        stride = 2
        pad = 0
        self.pad = nn.ZeroPad2D(1)
        self.conv1 = nn.Conv2D(in_nc, nf, 7, stride, pad, bias_attr=True)
        self.conv2 = nn.Conv2D(nf, nf, 3, stride, pad, bias_attr=True)
        self.conv3 = nn.Conv2D(nf, nf, 3, stride, pad, bias_attr=True)
        self.act = nn.ReLU()

    def forward(self, x):
        conv1_out = self.act(self.conv1(self.pad(x)))
        conv2_out = self.act(self.conv2(self.pad(conv1_out)))
        conv3_out = self.act(self.conv3(self.pad(conv2_out)))
        out = paddle.mean(conv3_out, axis=[2, 3], keepdim=False)

        return out


# 3layers with control
class CSRNet(nn.Layer):
    def __init__(self, in_nc=3, out_nc=3, base_nf=64, cond_nf=32):
        super(CSRNet, self).__init__()

        self.base_nf = base_nf
        self.out_nc = out_nc

        self.cond_net = Condition(in_nc=in_nc, nf=cond_nf)
      
        self.cond_scale1 = nn.Linear(cond_nf, base_nf, bias_attr=True)
        self.cond_scale2 = nn.Linear(cond_nf, base_nf,  bias_attr=True)
        self.cond_scale3 = nn.Linear(cond_nf, 3, bias_attr=True)

        self.cond_shift1 = nn.Linear(cond_nf, base_nf, bias_attr=True)
        self.cond_shift2 = nn.Linear(cond_nf, base_nf, bias_attr=True)
        self.cond_shift3 = nn.Linear(cond_nf, 3, bias_attr=True)

        self.conv1 = nn.Conv2D(in_nc, base_nf, 1, 1, bias_attr=True) 
        self.conv2 = nn.Conv2D(base_nf, base_nf, 1, 1, bias_attr=True)
        self.conv3 = nn.Conv2D(base_nf, out_nc, 1, 1, bias_attr=True)

        self.act = nn.ReLU()


    def forward(self, x):
        cond = self.cond_net(x)

        scale1 = self.cond_scale1(cond)
        shift1 = self.cond_shift1(cond)
        scale1 = paddle.reshape(scale1, [-1, self.base_nf, 1, 1])
        shift1 = paddle.reshape(shift1, [-1, self.base_nf, 1, 1])

        scale2 = self.cond_scale2(cond)
        shift2 = self.cond_shift2(cond)
        scale2 = paddle.reshape(scale2, [-1, self.base_nf, 1, 1])
        shift2 = paddle.reshape(shift2, [-1, self.base_nf, 1, 1])

        scale3 = self.cond_scale3(cond)
        shift3 = self.cond_shift3(cond)
        scale3 = paddle.reshape(scale3, [-1, self.out_nc, 1, 1])
        shift3 = paddle.reshape(shift3, [-1, self.out_nc, 1, 1])

        out = self.conv1(x)
        out = out * scale1 + shift1 + out
        out = self.act(out)
        

        out = self.conv2(out)
        out = out * scale2 + shift2 + out
        out = self.act(out)

        out = self.conv3(out)
        out = out * scale3 + shift3 + out
        return out