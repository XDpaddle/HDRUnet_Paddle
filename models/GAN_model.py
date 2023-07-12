import logging
from collections import OrderedDict
import paddle
import paddle.nn.functional as F
from paddle.fluid.dygraph import learning_rate_scheduler 
import paddle.nn as nn
from paddle.distributed import fleet
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss,class_loss_3class,average_loss_3class,GANLoss
from models.archs import arch_util
import cv2
import numpy as np
from utils import util
from data import util as ut
import os.path as osp
import os


logger = logging.getLogger('base')


class GANModel(BaseModel):
    def __init__(self, opt):
        super(GANModel, self).__init__(opt)



        if opt['dist']:
            self.rank = paddle.distributed.ParallelEnv().rank
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt)#.to(self.device)


        if opt['dist']:
            self.netG = fleet.distributed_model(self.netG)
        self.is_train = True
        if self.is_train:
            self.netD = networks.define_D(opt)
            if opt['dist']:
                self.netD = fleet.distributed_model(self.netD)
            self.netG.train()
            self.netD.train()

            # loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss()#.to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss()#.to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss()#.to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']
            
            self.cri_fea = nn.L1Loss()
            self.netF = networks.define_F(opt, use_bn=False)
            
            # GD gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            # D_update_ratio and D_init_iters
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

            # optimizers
            self.l_fea_w = train_opt['feature_weight']
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0.0
            wd_D = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0.0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                # if v.requires_grad:
                if not v.stop_gradient:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))

            # schedulers
            # if train_opt['lr_scheme'] == 'MultiStepLR':
            #     for optimizer in self.optimizers:
            #         self.schedulers.append(
            #             lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['T_period'],
            #                                              restarts=train_opt['restarts'],
            #                                              weights=train_opt['restart_weights'],
            #                                              gamma=train_opt['lr_gamma'],
            #                                              clear_state=train_opt['clear_state']))
            # elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
            #     for optimizer in self.optimizers:
            #         self.schedulers.append(
            #             # lr_scheduler.CosineAnnealingLR_Restart(
            #                 # optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
            #             lr_scheduler.CosineAnnealingDecay(
            #                 train_opt['lr_G'], train_opt['T_period'], eta_min=train_opt['eta_min'],
            #                 restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            # else:
            #     raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            
            if train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingDecay(train_opt['lr_G'],
                        train_opt['T_period'], eta_min=train_opt['eta_min'],
                        restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
                    #optimizer._learning_rate = self.schedulers[-1]
            elif train_opt['lr_scheme'] == 'MultiStepLR_Restart':
                self.schedulers.append(
                    lr_scheduler.MultiStepLR_Restart(train_opt['T_period'],
                                                    restarts=train_opt['restarts'],
                                                    weights=train_opt['restart_weights'],
                                                    gamma=train_opt['lr_gamma'],
                                                    clear_state=train_opt['clear_state']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')
            # self.optimizer_G = paddle.optimizer.SGD(learning_rate=self.schedulers[0], parameters=optim_params,
                                                    # weight_decay=wd_G)
            self.optimizer_G = paddle.optimizer.Adam(learning_rate=self.schedulers[0], parameters=optim_params,
                                                     weight_decay=wd_G,
                                                     beta1=train_opt['beta1'], beta2=train_opt['beta2'])
            self.optimizers.append(self.optimizer_G)
            self.optimizer_D = paddle.optimizer.Adam(learning_rate=self.schedulers[0], parameters=optim_params,
                                                     weight_decay=wd_D,
                                                     beta1=train_opt['beta1'], beta2=train_opt['beta2'])
            self.optimizers.append(self.optimizer_D)
            if opt['dist']:
                self.optimizer_G = fleet.distributed_optimizer(self.optimizer_G)

            self.log_dict = OrderedDict()
            
        # print network
        self.print_network()
        self.load()

    def feed_data(self, data, need_GT=True):
        paddle.device.set_device("gpu")
        self.var_L = data['LQ']#.to(self.device)  # LQ
        #self.SDR_base = data['SDR_base']#.to(self.device) # condition
        self.var_mask = data['mask']
        #self.var_L = F.normalize(self.var_L, axis=2)
        if need_GT:
            self.real_H = data['GT']#.to(self.device)  # GT
            self.var_ref = data['GT']
        #    self.real_H = F.normalize(self.real_H, axis=2)
        # print(self.var_L, self.real_H)

    def optimize_parameters(self, step):
        # 归一化
        self.optimizer_G.clear_grad()
        self.fake_H = self.netG((self.var_L, self.var_mask)) # HDRTV格式
        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_g_total = 0
        l_g_total += l_pix
        
        
        real_fea = self.netF(self.real_H).detach()
        fake_fea = self.netF(self.fake_H)
        
        l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
        
        pred_g_fake = self.netD(self.fake_H)
        #label = paddle.ones(shape=[1], dtype='float32')
        l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
        l_g_total += l_g_gan
        pred_d_real = self.netD(self.var_ref)
        l_d_real = self.cri_gan(pred_d_real, True)
        l_d_real.backward()
        # fake
        pred_d_fake = self.netD(self.fake_H.detach())  # detach to avoid BP to G
        l_d_fake = self.cri_gan(pred_d_fake, False) #gai
        l_g_total += l_d_fake
        #l_d_fake.backward()
        #l_pix.backward()
        #l_g_gan.backward()
        l_g_total.backward()
        
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self):
        self.netG.eval()
        with paddle.no_grad():
            self.fake_H = self.netG((self.var_L, self.var_mask))
        self.netG.train()


    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].astype('float')#().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].astype('float')#().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].astype('float')#().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        # if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
        #     net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
        #                                      self.netG.module.__class__.__name__)
        # else:
        net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n.item()))
            logger.info(s)
        if self.is_train:
            s, n = self.get_network_description(self.netG)
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
            if self.rank <= 0:
                logger.info('Network D structure: {}, with parameters: {:,d}'.format(net_struc_str, n.item()))
                logger.info(s) 
            """
            s, n = self.get_network_description(self.netF)
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
            if self.rank <= 0:
                logger.info('Network F structure: {}, with parameters: {:,d}'.format(net_struc_str, n.item()))
                logger.info(s)"""

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        load_path_G = '/root/autodl-tmp/ClassSR_paddle-main/experiments/highlight_generation_GAN_v2/models/140000_G.pdparams'
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
            
        load_path_D = self.opt['path']['pretrain_model_D']
        load_path_D = '/root/autodl-tmp/ClassSR_paddle-main/experiments/highlight_generation_GAN_v2/models/140000_D.pdparams'
        if load_path_D is not None:
            logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
        self.save_network(self.netD, 'D', iter_label)

