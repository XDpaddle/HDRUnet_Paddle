import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class Up(nn.Layer):

    def __init__(self, nc, bias):
        super(Up, self).__init__()
        self.up = nn.Conv2DTranspose(in_channels=nc, out_channels=nc, kernel_size=2, stride=2, bias_attr=bias)

    def forward(self, x1, x):
        x2 = self.up(x1)

        diffY = x.shape[2] - x2.shape[2]
        diffX = x.shape[3] - x2.shape[3]
        x3 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return x3


## Spatial Attention
class Basic(nn.Layer):
    def __init__(self, in_planes, out_planes, kernel_size, padding=0, bias=False):
        super(Basic, self).__init__()
        self.out_channels = out_planes
        groups = 1
        self.conv = nn.Conv2D(in_planes, out_planes, kernel_size=kernel_size, padding=padding, groups=groups, bias_attr=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class ChannelPool(nn.Layer):
    def __init__(self):
        super(ChannelPool, self).__init__()

    def forward(self, x):
        return paddle.concat((paddle.max(x, 1).unsqueeze(1), paddle.mean(x, 1).unsqueeze(1)), axis=1)


class SAB(nn.Layer):
    def __init__(self):
        super(SAB, self).__init__()
        kernel_size = 5
        self.compress = ChannelPool()
        self.spatial = Basic(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return x * scale


## Channel Attention Layer
class CAB(nn.Layer):
    def __init__(self, nc, reduction=8, bias=False):
        super(CAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv_du = nn.Sequential(
                nn.Conv2D(nc, nc // reduction, kernel_size=1, padding=0, bias_attr=bias),
                nn.ReLU(),
                nn.Conv2D(nc // reduction, nc, kernel_size=1, padding=0, bias_attr=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RAB(nn.Layer):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super(RAB, self).__init__()
        kernel_size = 3
        stride = 1
        padding = 1
        layers = []
        layers.append(nn.Conv2D(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias_attr=bias))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2D(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias_attr=bias))
        self.res = nn.Sequential(*layers)
        self.sab = SAB()

    def forward(self, x):
        x1 = x + self.res(x)
        x2 = x1 + self.res(x1)
        x3 = x2 + self.res(x2)

        x3_1 = x1 + x3
        x4 = x3_1 + self.res(x3_1)
        x4_1 = x + x4

        x5 = self.sab(x4_1)
        x5_1 = x + x5

        return x5_1


class HDRAB(nn.Layer):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super(HDRAB, self).__init__()
        kernel_size = 3
        reduction = 8

        self.cab = CAB(in_channels, reduction, bias)

        self.conv1 = nn.Conv2D(in_channels, out_channels, kernel_size=kernel_size, padding=1, dilation=1, bias_attr=bias)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2D(in_channels, out_channels, kernel_size=kernel_size, padding=2, dilation=2, bias_attr=bias)

        self.conv3 = nn.Conv2D(in_channels, out_channels, kernel_size=kernel_size, padding=3, dilation=3, bias_attr=bias)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2D(in_channels, out_channels, kernel_size=kernel_size, padding=4, dilation=4, bias_attr=bias)

        self.conv3_1 = nn.Conv2D(in_channels, out_channels, kernel_size=kernel_size, padding=3, dilation=3, bias_attr=bias)
        self.relu3_1 = nn.ReLU()

        self.conv2_1 = nn.Conv2D(in_channels, out_channels, kernel_size=kernel_size, padding=2, dilation=2, bias_attr=bias)

        self.conv1_1 = nn.Conv2D(in_channels, out_channels, kernel_size=kernel_size, padding=1, dilation=1, bias_attr=bias)
        self.relu1_1 = nn.ReLU()

        self.conv_tail = nn.Conv2D(in_channels, out_channels, kernel_size=kernel_size, padding=1, dilation=1, bias_attr=bias)

    def forward(self, y):
        y1 = self.conv1(y)
        y1_1 = self.relu1(y1)
        y2 = self.conv2(y1_1)
        y2_1 = y2 + y

        y3 = self.conv3(y2_1)
        y3_1 = self.relu3(y3)
        y4 = self.conv4(y3_1)
        y4_1 = y4 + y2_1

        y5 = self.conv3_1(y4_1)
        y5_1 = self.relu3_1(y5)
        y6 = self.conv2_1(y5_1+y3)
        y6_1 = y6 + y4_1

        y7 = self.conv1_1(y6_1+y2_1)
        y7_1 = self.relu1_1(y7)
        y8 = self.conv_tail(y7_1+y1)
        y8_1 = y8 + y6_1

        y9 = self.cab(y8_1)
        y9_1 = y + y9

        return y9_1


class DRANet(nn.Layer):
    def __init__(self, in_nc=3, out_nc=3, nc=128, bias=True):
        super(DRANet, self).__init__()
        kernel_size = 3

        self.conv_head = nn.Conv2D(in_nc, nc, kernel_size=kernel_size, padding=1, bias_attr=bias)

        self.rab = RAB(nc, nc, bias)

        self.hdrab = HDRAB(nc, nc, bias)

        self.conv_tail = nn.Conv2D(nc, out_nc, kernel_size=kernel_size, padding=1, bias_attr=bias)

        self.dual_tail = nn.Conv2D(2*out_nc, out_nc, kernel_size=kernel_size, padding=1, bias_attr=bias)

        self.down = nn.Conv2D(nc, nc, kernel_size=2, stride=2, bias_attr=bias)

        self.up = Up(nc, bias)

    def forward(self, x):
        x1 = self.conv_head(x)
        x2 = self.rab(x1)
        x2_1 = self.down(x2)
        x3 = self.rab(x2_1)
        x3_1 = self.down(x3)
        x4 = self.rab(x3_1)
        x4_1 = self.up(x4, x3)
        x5 = self.rab(x4_1 + x3)
        x5_1 = self.up(x5, x2)
        x6 = self.rab(x5_1 + x2)
        x7 = self.conv_tail(x6 + x1)
        X = x - x7

        y1 = self.conv_head(x)
        y2 = self.hdrab(y1)
        y3 = self.hdrab(y2)
        y4 = self.hdrab(y3)
        y5 = self.hdrab(y4 + y3)
        y6 = self.hdrab(y5 + y2)
        y7 = self.conv_tail(y6 + y1)
        Y = x -y7

        z1 = paddle.concat([X, Y], axis=1)
        z = self.dual_tail(z1)
        Z = x - z

        return Z