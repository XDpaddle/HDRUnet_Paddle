import random
import numpy as np
import cv2
import lmdb
import paddle
from paddle.io import Dataset
import data.util as util
import os.path as osp
# from cv2.ximgproc import guidedFilter


class LQGTDataset_rcan(Dataset):
    """
    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, etc) and GT image pairs.
    If only GT images are provided, generate LQ images on-the-fly.
    """

    def __init__(self, opt):
        super(LQGTDataset_rcan, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.LQ_env, self.GT_env = None, None  # environments for lmdb

        self.paths_GT, self.sizes_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])
        self.paths_LQ, self.sizes_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])
        #self.paths_cond, self.sizes_cond = util.get_image_paths(self.data_type, opt['dataroot_cond'])
        #self.cond_folder = opt['dataroot_cond']

        # self.paths_GT=[self.paths_GT[400000],self.paths_GT[800000],self.paths_GT[1200000]]
        # self.paths_LQ=[self.paths_LQ[400000],self.paths_LQ[800000],self.paths_LQ[1200000]]


        assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))
        self.random_scale_list = [1]

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def __getitem__(self, index):
        if self.data_type == 'lmdb' and (self.GT_env is None or self.LQ_env is None):
            self._init_lmdb()
        GT_path, LQ_path = None, None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        # get GT image
        GT_path = self.paths_GT[index]
        resolution = [int(s) for s in self.sizes_GT[index].split('_')
                      ] if self.data_type == 'lmdb' else None
        #img_GT = util.read_img_rcan(self.GT_env, GT_path, resolution)
        img_GT = util.imread(GT_path, float32=True, bit_depth=16)
        #print(np.array(img_GT).astype('float32'))
        if self.opt['phase'] != 'train':  # modcrop in the validation / test phase
            img_GT = util.modcrop(img_GT, scale)
        if self.opt['color']:  # change color space if necessary
            img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]
        
        #print(np.array(img_GT).astype('float32'))

        # get LQ image
        if self.paths_LQ:
            LQ_path = self.paths_LQ[index]
            resolution = [int(s) for s in self.sizes_LQ[index].split('_')
                          ] if self.data_type == 'lmdb' else None
            #img_LQ = util.read_img_rcan(self.LQ_env, LQ_path, resolution)
            img_LQ = util.imread(LQ_path, float32=True, bit_depth=8)
            
        else:  # down-sampling on-the-fly
            # randomly scale during training
            if self.opt['phase'] == 'train':
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_GT.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, GT_size)
                W_s = _mod(W_s, random_scale, scale, GT_size)
                img_GT = cv2.resize(img_GT, (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                if img_GT.ndim == 2:
                    img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_GT.shape
            # using matlab imresize
            img_LQ = util.imresize_np(img_GT, 1 / scale, True)
            if img_LQ.ndim == 2:
                img_LQ = np.expand_dims(img_LQ, axis=2)

        #print(np.array(img_LQ).astype('float32'))
        
        # # get condition
        """
        cond_scale = self.opt['cond_scale']
        if self.cond_folder is not None:
            if '_' in osp.basename(LQ_path):
                cond_name = '_'.join(osp.basename(LQ_path).split('_')[:-1])+'_bicx'+str(cond_scale)+'.png'
            else: cond_name = osp.basename(LQ_path).split('.')[0]+'_bicx'+str(cond_scale)+'.png'
            cond_path = osp.join(self.cond_folder, cond_name)
        """
        #SDR_base = guidedFilter(guide=img_LQ, src=img_LQ, radius=5, eps=0.01)
        cond = img_LQ.copy()
        
        if self.opt['phase'] == 'train':
            # if the image size is too small
            H, W, _ = img_GT.shape
            if H < GT_size or W < GT_size:
                img_GT = cv2.resize(img_GT, (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                img_LQ = util.imresize_np(img_GT, 1 / scale, True)
                if img_LQ.ndim == 2:
                    img_LQ = np.expand_dims(img_LQ, axis=2)

            H, W, C = img_LQ.shape
            #LQ_size = GT_size // scale
            
            #print(np.array(img_GT).astype('float32'))
            #print(np.array(img_LQ).astype('float32'))

            """
            # randomly crop
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]
            """
            #print(np.array(img_GT).astype('float32'))
            #print(np.array(img_LQ).astype('float32'))

            # augmentation - flip, rotate
            img_LQ, img_GT = util.augment([img_LQ, img_GT], self.opt['use_flip'],
                                          self.opt['use_rot'])
        
        #print(np.array(img_GT).astype('float32'))
        #print(np.array(img_LQ).astype('float32'))
        
        if self.opt['color']:  # change color space if necessary
            img_LQ = util.channel_convert(C, self.opt['color'],
                                          [img_LQ])[0]  

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]
            cond = cond[:, :, [2, 1, 0]]
            #SDR_base = SDR_base[:, :, [2, 1, 0]]
        #print(np.array(img_GT).astype('float32'))
        img_GT = paddle.to_tensor(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1))),paddle.float32)
        img_LQ = paddle.to_tensor(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1))),paddle.float32)
        cond = paddle.to_tensor(np.ascontiguousarray(np.transpose(cond, (2, 0, 1))),paddle.float32)
        #SDR_base = paddle.to_tensor(np.ascontiguousarray(np.transpose(SDR_base, (2, 0, 1))),paddle.float32)

        if LQ_path is None:
            LQ_path = GT_path
        return {'LQ': img_LQ, 'GT': img_GT, 'cond': cond, 'LQ_path': LQ_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)
