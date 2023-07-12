import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.cvlibs import param_init as init
from models.archs.kb_utils import KBAFunction
from models.archs.kb_utils import LayerNorm2d, SimpleGate


class KBBlock_s(nn.Layer):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, nset=32, k=3, gc=4, lightweight=False):
        super(KBBlock_s, self).__init__()
        self.k, self.c = k, c
        self.nset = nset
        dw_ch = int(c * DW_Expand)
        ffn_ch = int(FFN_Expand * c)

        self.g = c // gc
        #self.w = nn.Parameter(paddle.zeros(1, nset, c * c // self.g * self.k ** 2))
        x=paddle.zeros(shape=[1, nset, c * c // self.g * self.k ** 2])
        self.w = paddle.create_parameter(shape=x.shape,
                        dtype=str(x.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Assign(x))
        #self.b = nn.Parameter(paddle.zeros(1, nset, c))
        x=paddle.zeros(shape=[1, nset, c])
        self.b = paddle.create_parameter(shape=x.shape,
                        dtype=str(x.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Assign(x))
        self.init_p(self.w, self.b)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                      groups=1, bias_attr=True),
        )

        if not lightweight:
            self.conv11 = nn.Sequential(
                nn.Conv2D(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                          bias_attr=True),
                nn.Conv2D(in_channels=c, out_channels=c, kernel_size=5, padding=2, stride=1, groups=c // 4,
                          bias_attr=True),
            )
        else:
            self.conv11 = nn.Sequential(
                nn.Conv2D(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                          bias_attr=True),
                nn.Conv2D(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c,
                          bias_attr=True),
            )

        self.conv1 = nn.Conv2D(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                               bias_attr=True)
        self.conv21 = nn.Conv2D(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c,
                                bias_attr=True)

        interc = min(c, 32)
        self.conv2 = nn.Sequential(
            nn.Conv2D(in_channels=c, out_channels=interc, kernel_size=3, padding=1, stride=1, groups=interc,
                      bias_attr=True),
            SimpleGate(),
            nn.Conv2D(interc // 2, self.nset, 1, padding=0, stride=1),
        )

        self.conv211 = nn.Conv2D(in_channels=c, out_channels=self.nset, kernel_size=1)

        self.conv3 = nn.Conv2D(in_channels=dw_ch // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias_attr=True)

        self.conv4 = nn.Conv2D(in_channels=c, out_channels=ffn_ch, kernel_size=1, padding=0, stride=1, groups=1,
                               bias_attr=True)
        self.conv5 = nn.Conv2D(in_channels=ffn_ch // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias_attr=True)

        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()

        #self.ga1 = nn.Parameter(paddle.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)
        x=paddle.zeros(shape=[1, c, 1, 1]) + 1e-2
        self.ga1 = paddle.create_parameter(shape=x.shape,
                        dtype=str(x.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Assign(x))
        #self.attgamma = nn.Parameter(paddle.zeros((1, self.nset, 1, 1)) + 1e-2, requires_grad=True)
        x=paddle.zeros(shape=[1, self.nset, 1, 1]) + 1e-2
        self.attgamma = paddle.create_parameter(shape=x.shape,
                        dtype=str(x.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Assign(x))
        self.sg = SimpleGate()

        #self.beta = nn.Parameter(paddle.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)
        x=paddle.zeros(shape=[1, c, 1, 1]) + 1e-2
        self.beta = paddle.create_parameter(shape=x.shape,
                        dtype=str(x.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Assign(x))
        #self.gamma = nn.Parameter(paddle.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)
        x=paddle.zeros(shape=[1, c, 1, 1]) + 1e-2
        self.gamma = paddle.create_parameter(shape=x.shape,
                        dtype=str(x.numpy().dtype),
                        default_initializer=paddle.nn.initializer.Assign(x))

    def init_p(self, weight, bias=None):
        init.kaiming_uniform(weight)
        if bias is not None:
            init.uniform_init(bias)

    def KBA(self, x, att, selfk, selfg, selfb, selfw):
        return KBAFunction.apply(x, att, selfk, selfg, selfb, selfw)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        sca = self.sca(x)
        x1 = self.conv11(x)

        # KBA module
        att = self.conv2(x) * self.attgamma + self.conv211(x)
        uf = self.conv21(self.conv1(x))
        x = self.KBA(uf, att, self.k, self.g, self.b, self.w) * self.ga1 + uf
        x = x * x1 * sca

        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        # FFN
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        return y + x * self.gamma


class KBNet_s(nn.Layer):
    def __init__(self, img_channel=3, width=64, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8],
                 dec_blk_nums=[2, 2, 2, 2], basicblock='KBBlock_s', lightweight=False, ffn_scale=2):
        super().__init__()
        basicblock = eval(basicblock)

        self.intro = nn.Conv2D(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1, bias_attr=True)

        self.encoders = nn.LayerList()
        self.middle_blks = nn.LayerList()
        self.decoders = nn.LayerList()

        self.ending = nn.Conv2D(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1, bias_attr=True)

        self.ups = nn.LayerList()
        self.downs = nn.LayerList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[basicblock(chan, FFN_Expand=ffn_scale, lightweight=lightweight) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2D(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[basicblock(chan, FFN_Expand=ffn_scale, lightweight=lightweight) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2D(chan, chan * 2, 1, bias_attr=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[basicblock(chan, FFN_Expand=ffn_scale, lightweight=lightweight) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.shape # size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
