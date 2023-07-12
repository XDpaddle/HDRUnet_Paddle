import paddle
import paddle.nn as nn
from paddleseg.cvlibs import param_init

eps = 1e-16

def reluBlock():
    return nn.ReLU()

def convBlock_input(out_channels):
    return nn.Conv2D(in_channels=6, out_channels=out_channels, 
                     kernel_size=3, stride=1, padding=1, bias_attr=True)

def convBlock_dr(out_channels):
    return nn.Conv2D(in_channels=2 * out_channels, 
                     kernel_size=1, out_channels=out_channels, 
                     stride=1, padding=0, bias_attr=True)

def convBlock_scalec(in_channels, scale):
    return nn.Conv2D(in_channels=in_channels, 
                     kernel_size=3, out_channels=scale * scale * in_channels,
                     stride=1, padding=1, bias_attr=True)

def convBlock_c(channels):
    return nn.Conv2D(in_channels=channels,out_channels=channels,
                    kernel_size=3, stride=1, padding=1, bias_attr=True)

def convBlock_c_end(in_channels):
    return nn.Conv2D(in_channels=in_channels,out_channels=3,
                    kernel_size=3, stride=1, padding=1, bias_attr=True)

def pixel_shuffle(scale):
    return nn.PixelShuffle(upscale_factor=scale)

def bicubic(scale):
    return nn.Upsample(scale_factor=scale, mode='bicubic', align_corners=False)

# Define ResBlocks
class BaseBlock(nn.Layer):
    
    def __init__(self, channels):
        super(BaseBlock, self).__init__()
        self.conv1 = convBlock_c(channels=channels)
        self.conv2 = convBlock_c(channels=channels)
        self.relu = reluBlock()
        
    
    def forward(self, x):
        
        out = self.relu(x)
        out = self.conv1(out)
        
        out = self.relu(out)
        out = self.conv2(out)
        
        return out

class ResBlock(nn.Layer):
    
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.baseblock = BaseBlock(channels=channels)
        
    def forward(self, x):
        identity = x
        
        out = self.baseblock(x)
        out += identity
        
        return out

    
class ResModBlock(nn.Layer):
    
    def __init__(self, channels):
        super(ResModBlock, self).__init__()
        self.baseblock = BaseBlock(channels=channels)
        self.conv_smf_1 = convBlock_c(channels=channels)
        self.conv_smf_2 = convBlock_c(channels=channels)
        self.relu = reluBlock()
        
    def forward(self, x, SMF):
        identity = x
        
        out = self.baseblock(x)
        out_shared = self.conv_smf_1(SMF)
        out_shared = self.relu(out_shared)
        out_shared = self.conv_smf_2(out_shared)
        
        # multiply elementwisely
        out *= out_shared
        out += identity
        
        return out
    
class SkipBlock(nn.Layer):
    
    def __init__(self, channels):
        super(SkipBlock, self).__init__()
        self.conv1 = convBlock_c(channels=channels)
        self.conv2 = convBlock_c(channels=channels)
        self.relu = reluBlock()
        self.dr = convBlock_dr(out_channels=channels)
        
    
    def forward(self, x, block_out_feature):
        
        in_features = paddle.concat([x, block_out_feature], axis=1)
        out = self.relu(in_features)
        out = self.dr(out)
        out = self.conv1(out)
        
        out = self.relu(out)
        out = self.conv2(out)
        
        
        return out
    
class ResSkipBlock(nn.Layer):
    
    def __init__(self, channels):
        super(ResSkipBlock, self).__init__()
        self.skipblock = SkipBlock(channels=channels)
           
    def forward(self, x, RMB_out_feature):
        identity = x
        
        out = self.skipblock(x, RMB_out_feature)
        out += identity
        
        return out

class ResSkipModBlock(nn.Layer):
    
    def __init__(self, channels):
        super(ResSkipModBlock, self).__init__()
        self.skipblock = SkipBlock(channels=channels)
        self.conv_smf_1 = convBlock_c(channels=channels)
        self.conv_smf_2 = convBlock_c(channels=channels)
        self.relu = reluBlock() 
        
    def forward(self, x, RB_out_feature, SMF):
        identity = x
        
        out = self.skipblock(x, RB_out_feature)
        
        out_shared = self.conv_smf_1(SMF)
        out_shared = self.relu(out_shared)
        out_shared = self.conv_smf_2(out_shared)
        
        out *= out_shared
        out += identity
        
        return out
    
class SR_ITM_base_net(nn.Layer):
    r""" Base SR_ITM Net for pretraining"""
    
    def __init__(self, channels, scale):
        super(SR_ITM_base_net, self).__init__()
        self.conv_base_1 = convBlock_input(out_channels=channels)
        self.conv_detail_1 = convBlock_input(out_channels=channels)
        self.relu = reluBlock()
        
        # Setting up Base Layer Pass
        # 6 ResBlocks
        for i in range(1, 7):
            self.__setattr__('rb_base_{}'.format(i), ResBlock(channels=channels))
        
        # Setting up Detail Layer Pass
        self.rb_detail_1 = ResBlock(channels=channels)
        
        # 5 ResSkipBlocks
        for i in range(1, 6):
            self.__setattr__('rsb_detail_{}'.format(i), ResSkipBlock(channels=channels))
            
        # Fusion
        self.dr1 = convBlock_dr(out_channels=channels)
        self.conv_fusion_1 = convBlock_c(channels=channels)
        
        # 10 ResBlocks for fusion
        for i in range(1, 11):
            self.__setattr__('rb_fusion_{}'.format(i), ResBlock(channels=channels))
        
        self.conv_fusion_2 = convBlock_c(channels=channels)
        self.conv_fusion_3 = convBlock_scalec(in_channels=channels, scale=scale)
        
        self.ps = pixel_shuffle(scale=scale)
        self.conv_fusion_4 = convBlock_c_end(in_channels=channels)
        
        self.bicubic = bicubic(scale=scale)
        
        # params_init
        for m in self.named_children():
            if isinstance(m, nn.Conv2D):
                param_init.xavier_uniform(m.weight, gain=1.0)
                param_init.constant_init(m.bias_attr, 0)
        
    def forward(self, im, im_base):
        r"""
        Args:
            im (Tensor): The original Image
            im_base (Tensor): The filtered-map after guided-filtering 
        """
        # Pre-processing
        im_detail = im / (im_base + eps)
        input_b = paddle.concat([im, im_base], axis=1)
        input_d = paddle.concat([im, im_detail], axis=1)
        
        out_b = self.conv_base_1(input_b)
        out_d = self.conv_detail_1(input_d)
        out_d = self.rb_detail_1(out_d)
        
        # 2 Passes
        for i in range(1, 6):
            out_b = self.__getattr__('rb_base_{}'.format(i))(out_b)
            out_d = self.__getattr__('rsb_detail_{}'.format(i))(x=out_d, RMB_out_feature=out_b)
         
        out_b = self.rb_base_6(out_b)
        
        # Fusion
        fused_features = paddle.concat([out_b, out_d], axis=1)
        out = self.relu(fused_features)
        out = self.dr1(out)
        out = self.conv_fusion_1(out)
        
        for i in range(1, 11):
            out = self.__getattr__('rb_fusion_{}'.format(i))(out)
            
        out = self.relu(out)
        out = self.conv_fusion_2(out)
        out = self.relu(out)
        out = self.conv_fusion_3(out)
        out = self.relu(out)
        
        out = self.ps(out)
        out = self.conv_fusion_4(out)
        
        out += self.bicubic(im)
        
        return out
    


class SR_ITM_full_net(nn.Layer):
    r""" Full SR_ITM Net for training and refining"""
    
    def __init__(self, channels, scale):
        super(SR_ITM_full_net, self).__init__()
        
        # Setting up shared layers for modulation(SMF)
        self.conv_base_shared_1 = convBlock_input(out_channels=channels)
        self.conv_base_shared_2 = convBlock_c(channels=channels)
        self.conv_base_shared_3 = convBlock_c(channels=channels)
                
        self.conv_detail_shared_1 = convBlock_input(out_channels=channels)
        self.conv_detail_shared_2 = convBlock_c(channels=channels)
        self.conv_detail_shared_3 = convBlock_c(channels=channels)
        
        self.conv_base_1 = convBlock_input(out_channels=channels)
        self.conv_detail_1 = convBlock_input(out_channels=channels)
        self.relu = reluBlock()
        
        # Setting up Base Layer Pass
        # 3 ResBlocks and 3 ResModBlocks, interweaving
        for i in range(1, 4):
            self.__setattr__('rb_base_{}'.format(i), ResBlock(channels=channels))
            self.__setattr__('rmb_base_{}'.format(i), ResModBlock(channels=channels))
        
        # Setting up Detail Layer Pass
        self.rb_detail_1 = ResBlock(channels=channels)
        
        # (RSMB, RSB, RSMB, RSB, RSMB) 5 blcoks
        for i in range(1, 3):
            self.__setattr__('rsmb_detail_{}'.format(i), ResSkipModBlock(channels=channels))
            self.__setattr__('rsb_detail_{}'.format(i), ResSkipBlock(channels=channels))
        
        self.rsmb_detail_3 = ResSkipModBlock(channels=channels)
        
        # Fusion
        self.dr1 = convBlock_dr(out_channels=channels)
        self.conv_fusion_1 = convBlock_c(channels=channels)
        
        # 10 ResBlocks for fusion
        for i in range(1, 11):
            self.__setattr__('rb_fusion_{}'.format(i), ResBlock(channels=channels))
        
        self.conv_fusion_2 = convBlock_c(channels=channels)
        self.conv_fusion_3 = convBlock_scalec(in_channels=channels, scale=scale)
        
        self.ps = pixel_shuffle(scale=scale)
        self.conv_fusion_4 = convBlock_c_end(in_channels=channels)
        
        self.bicubic = bicubic(scale=scale)
        
        # params_init
        for m in self.named_children():
            if isinstance(m, nn.Conv2D):
                param_init.xavier_uniform(m.weight, gain=1.0)
                param_init.constant_init(m.bias_attr, 0)
        
    def forward(self, im, im_base):
        r"""
        Args:
            im (Tensor): The original Image
            im_base (Tensor): The filtered-map after guided-filtering 
        """
        # Pre-processing
        im_detail = im / (im_base + eps)
        input_b = paddle.concat([im, im_base], axis=1)
        input_d = paddle.concat([im, im_detail], axis=1)
        
        out_b = self.conv_base_1(input_b)
        out_d = self.conv_detail_1(input_d)
        out_d = self.rb_detail_1(out_d)
        
        # shared layers for modulation
        SMF_b = self.conv_base_shared_1(input_b)
        SMF_b = self.relu(SMF_b)
        SMF_b = self.conv_base_shared_2(SMF_b)
        SMF_b = self.relu(SMF_b)
        SMF_b = self.conv_base_shared_3(SMF_b)
        SMF_b = self.relu(SMF_b)
        
        SMF_d = self.conv_detail_shared_1(input_d)
        SMF_d = self.relu(SMF_d)
        SMF_d = self.conv_detail_shared_2(SMF_d)
        SMF_d = self.relu(SMF_d)
        SMF_d = self.conv_detail_shared_3(SMF_d)
        SMF_d = self.relu(SMF_d)
        
        # 2 Passes
        for i in range(1, 4):
            out_b = self.__getattr__('rb_base_{}'.format(i))(out_b)
            out_d = self.__getattr__('rsmb_detail_{}'.format(i))(x=out_d, RB_out_feature=out_b, SMF=SMF_d)
            out_b = self.__getattr__('rmb_base_{}'.format(i))(x=out_b, SMF=SMF_b)
            # rule out the 3rd (i.e. last) rsb
            if i != 3:
                out_d = self.__getattr__('rsb_detail_{}'.format(i))(x=out_d, RMB_out_feature=out_b)
         
        
        # Fusion
        fused_features = paddle.concat([out_b, out_d], axis=1)
        out = self.relu(fused_features)
        out = self.dr1(out)
        out = self.conv_fusion_1(out)
        
        for i in range(1, 11):
            out = self.__getattr__('rb_fusion_{}'.format(i))(out)
            
        out = self.relu(out)
        out = self.conv_fusion_2(out)
        out = self.relu(out)
        out = self.conv_fusion_3(out)
        out = self.relu(out)
        
        out = self.ps(out)
        out = self.conv_fusion_4(out)
        
        out += self.bicubic(im)
        
        return out