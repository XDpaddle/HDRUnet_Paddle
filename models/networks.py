import paddle
import models.archs.classSR_rcan_arch as classSR_rcan_arch
import models.archs.HDR_arch as HDR_arch
#import models.archs.csrnet as CSRNet
import models.archs.Condition_arch as Condition_arch
import models.archs.deep_sritm as deep_sritm
import models.archs.UNet_arch as UNet_arch

# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    # image restoration
    if which_model == 'deep_sritm_full':
        netG = deep_sritm.SR_ITM_full_net(channels=opt_net['channels'], scale=opt_net['scale'])
    elif which_model == 'HDRUNet':
        netG = UNet_arch.HDRUNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], act_type=opt_net['act_type'])
    elif which_model == 'deep_sritm_base':
        netG = deep_sritm.SR_ITM_base_net(channels=opt_net['channels'], scale=opt_net['scale'])    
    elif which_model == 'ConditionNet':
        netG = Condition_arch.ConditionNet(classifier=opt_net['classifier'], cond_c=opt_net['cond_c'])
    elif which_model == 'HyCondITMv1':
        netG = HDR_arch.HyCondITMv1(transform_channels=opt_net['transform_channels'], global_cond_channels=opt_net['global_cond_channels'],
                              merge_cond_channels=opt_net['merge_cond_channels'], in_channels=opt_net['in_channels'])

    #elif which_model == 'CSRNet':
    #    netG = CSRNet.CSRNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], base_nf=opt_net['base_nf'], cond_nf=opt_net['cond_nf'])
        
    elif which_model == 'classSR_3class_rcan':
        netG = classSR_rcan_arch.classSR_3class_rcan(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'])

    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG