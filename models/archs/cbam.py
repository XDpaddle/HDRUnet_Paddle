import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class ChannelAttentionModule(nn.Layer):
    def __init__(self, in_channels, r):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)
        self.mlp = nn.Sequential(
            nn.Conv2D(in_channels, in_channels // r, 1, bias_attr=False),
            nn.ReLU(),
            nn.Conv2D(in_channels // r, in_channels, 1, bias_attr=False),
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        y1 = self.avg_pool(x)
        y1 = self.mlp(y1)

        y2 = self.max_pool(x)
        y2 = self.mlp(y2)

        y = self.act(y1 + y2)

        return x * y


class SpatialAttentionModule(nn.Layer):
    def __init__(self, kernel_size):
        super().__init__()

        padding = (kernel_size - 1) // 2

        self.layer = nn.Sequential(
            nn.Conv2D(2, 1, kernel_size, padding=padding),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_mask = paddle.mean(x, axis=1, keepdim=True)
        #print(paddle.max(x, axis=1, keepdim=True))
        max_mask = paddle.max(x, axis=1, keepdim=True)
        mask = paddle.concat([avg_mask, max_mask], axis=1)

        mask = self.layer(mask)
        return x * mask


class CBAM(nn.Layer):
    def __init__(self, in_channels, r=16, kernel_size=7):
        super().__init__()

        self.conv = nn.Conv2D(in_channels, in_channels, 3, stride=1, padding=1)
        self.cam = ChannelAttentionModule(in_channels, r)
        self.sam = SpatialAttentionModule(kernel_size)

    def forward(self, x):
        y = self.conv(x)
        y = self.cam(y)
        y = self.sam(y)

        return x + y
