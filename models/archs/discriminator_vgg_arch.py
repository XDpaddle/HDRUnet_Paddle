import paddle
import paddle.nn as nn
from x2paddle import models

class Discriminator_VGG_128(nn.Layer):
    def __init__(self, in_nc, nf):
        super(Discriminator_VGG_128, self).__init__()
        # [64, 128, 128]
        self.conv0_0 = nn.Conv2D(in_nc, nf, 3, 1, 1, bias_attr=True)
        self.conv0_1 = nn.Conv2D(nf, nf, 4, 2, 1, bias_attr=False)
        self.bn0_1 = nn.BatchNorm2D(nf)
        # [64, 64, 64]
        self.conv1_0 = nn.Conv2D(nf, nf * 2, 3, 1, 1, bias_attr=False)
        self.bn1_0 = nn.BatchNorm2D(nf * 2)
        self.conv1_1 = nn.Conv2D(nf * 2, nf * 2, 4, 2, 1, bias_attr=False)
        self.bn1_1 = nn.BatchNorm2D(nf * 2)
        # [128, 32, 32]
        self.conv2_0 = nn.Conv2D(nf * 2, nf * 4, 3, 1, 1, bias_attr=False)
        self.bn2_0 = nn.BatchNorm2D(nf * 4)
        self.conv2_1 = nn.Conv2D(nf * 4, nf * 4, 4, 2, 1, bias_attr=False)
        self.bn2_1 = nn.BatchNorm2D(nf * 4)
        # [256, 16, 16]
        self.conv3_0 = nn.Conv2D(nf * 4, nf * 8, 3, 1, 1, bias_attr=False)
        self.bn3_0 = nn.BatchNorm2D(nf * 8)
        self.conv3_1 = nn.Conv2D(nf * 8, nf * 8, 4, 2, 1, bias_attr=False)
        self.bn3_1 = nn.BatchNorm2D(nf * 8)
        # [512, 8, 8]
        self.conv4_0 = nn.Conv2D(nf * 8, nf * 8, 3, 1, 1, bias_attr=False)
        self.bn4_0 = nn.BatchNorm2D(nf * 8)
        self.conv4_1 = nn.Conv2D(nf * 8, nf * 8, 4, 2, 1, bias_attr=False)
        self.bn4_1 = nn.BatchNorm2D(nf * 8)

        self.linear1 = nn.Linear(512 * 15 * 15, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))

        fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))

        #fea = fea.view(fea.size(0), -1)
        fea = paddle.reshape(fea, [fea.shape[0], -1])
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out


class VGGFeatureExtractor(nn.Layer):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True,
                 device=paddle.set_device("gpu")):
        super(VGGFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        if use_bn:
            model = models.vgg19_bn_pth(pretrained=True)
        else:
            model = models.vgg19_pth(pretrained=True)
        if self.use_input_norm:
            mean = paddle.to_tensor([0.485, 0.456, 0.406])
            mean = paddle.reshape(mean, [1, 3, 1, 1])
            # [0.485 - 1, 0.456 - 1, 0.406 - 1] if input in range [-1, 1]
            std = paddle.to_tensor([0.229, 0.224, 0.225])
            std = paddle.reshape(std, [1, 3, 1, 1])
            # [0.229 * 2, 0.224 * 2, 0.225 * 2] if input in range [-1, 1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.stop_gradient = True

    def forward(self, x):
        # Assume input range is [0, 1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output
