import logging
from PIL import Image
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torchvision.transforms.transforms import CenterCrop
from mydataset import Nutrition_RGBD
import pdb
import random

def get_DataLoader(args):
    """根据训练参数构建 Nutrition5k 的 train/test DataLoader。

    该函数统一定义图像 transform、列表文件路径、食材向量格式和 DataLoader
    参数，是 train_rgbd.py 的数据入口。返回的 batch 字段顺序由
    mydataset.Nutrition_RGBD.__getitem__ 决定。
    """
    #image_sizes = ((256, 352), (320, 448))
    train_transform = transforms.Compose([
        #transforms.Resize((320, 448)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    test_transform = transforms.Compose([
        #transforms.Resize((320, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Nutrition5k 的图像和列表文件都放在 data_root/imagery 下。
    nutrition_rgbd_ims_root = os.path.join(args.data_root, 'imagery')
    nutrition_train_txt = os.path.join(args.data_root, 'imagery','rgbd_train_processed_refine.txt')
    nutrition_test_txt = os.path.join(args.data_root, 'imagery','rgbd_test_processed1_refine.txt') # depth_color.png
    nutrition_train_rgbd_txt = os.path.join(args.data_root, 'imagery','rgb_in_overhead_train_processed_refine.txt')
    nutrition_test_rgbd_txt = os.path.join(args.data_root, 'imagery','rgb_in_overhead_test_processed1_refine.txt') # rbg.png
    ingredients_mode = getattr(args, "ingredients_mode", "clip512")
    # 当前模型的食材输入维度由格式决定：CLIP 文本特征 512 维，binary 标签 255 维。
    ingredients_dim = 255 if ingredients_mode == "binary255" else 512
    ingredients_vocab_path = getattr(args, "ingredients_vocab", "")

    # training=True 会启用 Dataset 内部的同步翻转/旋转增强。
    trainset = Nutrition_RGBD(
        nutrition_rgbd_ims_root,
        nutrition_train_rgbd_txt,
        nutrition_train_txt,
        training=True,
        transform=train_transform,
        ingredients_mode=ingredients_mode,
        ingredients_dim=ingredients_dim,
        ingredients_vocab_path=ingredients_vocab_path,
    )
    # 验证/测试集不做随机增强，只做确定性的 Tensor 化和归一化。
    testset = Nutrition_RGBD(
        nutrition_rgbd_ims_root,
        nutrition_test_rgbd_txt,
        nutrition_test_txt,
        training=False,
        transform=test_transform,
        ingredients_mode=ingredients_mode,
        ingredients_dim=ingredients_dim,
        ingredients_vocab_path=ingredients_vocab_path,
    )

    train_loader = DataLoader(trainset,
                              batch_size=args.b,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True
                              )
    test_loader = DataLoader(testset,
                             batch_size=args.b,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True
                             )

    return train_loader, test_loader



