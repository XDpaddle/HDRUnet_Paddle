# HG_paddle

Paper: A New Journey from SDRTV to HDRTV.

[Paper](https://paperswithcode.com/paper/a-new-journey-from-sdrtv-to-hdrtv)

作者: Xiangyu Chen等

Paddle 复现版本

## 数据集

https://pan.baidu.com/s/1OSLVoBioyen-zjvLmhbe2g

提取码: 2me9

## 训练模型

链接：https://pan.baidu.com/s/1PxihzPH8WCgdRAtP9fivBg?pwd=hh66 
提取码：hh66

## 训练步骤

### train 

```bash
#First
python train.py -opt options/train/train_pre_Gen.yml
#Then
python train.py -opt config/train/train_HG.yml
```

```
多卡仅需
​```bash
python -m paddle.distributed.launch train.py --launcher fleet -opt config_file_path
```

## 测试步骤

```bash
python test.py -opt config/test/test_HG.yml
```

## 复现指标

|        | PSNR  |
| ------ | ----- |
| Paddle | 37.60 |

注：因显存限制，测试结果为测试图片降采样到1080p的结果