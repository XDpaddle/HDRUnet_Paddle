# HDRUNet_paddle

Paper: HDRUNet: Single Image HDR Reconstruction with Denoising and Dequantization.

[Paper](https://paperswithcode.com/paper/hdrunet-single-image-hdr-reconstruction-with)

作者: Xiangyu Chen等

Paddle 复现版本

## 数据集

https://pan.baidu.com/s/1OSLVoBioyen-zjvLmhbe2g

提取码: 2me9

## 训练模型

链接：https://pan.baidu.com/s/15v9e9vQ5RLk7P8y7TJyQqw?pwd=hh66 
提取码：hh66

## 训练步骤

### train 

```bash
python train.py -opt config/train/train_HDRUNet.yml
```

```
多卡仅需
​```bash
python -m paddle.distributed.launch train.py --launcher fleet -opt config_file_path
```

## 测试步骤

```bash
python test.py -opt config/test/test_HDRUNet.yml
```

## 复现指标

|        | PSNR  |
| ------ | ----- |
| Paddle | 37.35 |

注：因显存限制，测试结果为测试图片降采样到1080p的结果