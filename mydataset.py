import os
import json
from os.path import join

import numpy as np
import scipy
from scipy import io
import scipy.misc
from PIL import Image
import pandas as pd
# import matplotlib.pyplot as plt
import random

import torch
from torch.utils.data import Dataset


import imageio
import cv2
import pdb
from typing import Optional, Dict
import csv


"""
Nutrition5k RGB-D 数据集读取与食材特征构造。

Dataset 每个样本返回：
RGB 图像、dish_id/label、五个营养真值、depth/RGB-D 图像、食材向量。
食材向量支持两种格式：
- clip512：从 ingredients_fea_nonorm/{dish_id}.pth 读取 512 维 CLIP 文本特征。
- binary255：从 Nutrition5k 元数据 CSV、JSON 或索引列表构造 255 维二值向量。
"""


_NUTRITION5K_CACHE = {
    # key: (csv_dir, dim) -> (vocab: {name->idx}, dish_map: {dish_id->list[idx]})
}


def _load_ingredients_vocab(vocab_path: str) -> Optional[Dict[str, int]]:
    """读取外部食材词表。

    JSON 可以是 {name: idx} 字典，也可以是 [name0, name1, ...] 列表。
    返回统一的 name -> index 映射，供 binary255 模式把名称转成二值位置。
    """
    if not vocab_path:
        return None
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    if isinstance(vocab, dict):
        return {str(k): int(v) for k, v in vocab.items()}
    if isinstance(vocab, list):
        return {str(name): int(i) for i, name in enumerate(vocab)}
    raise ValueError(f"Unsupported vocab format in {vocab_path}: expected dict or list, got {type(vocab)}")


def _ingredients_to_binary_vector(obj, dim: int, vocab: Optional[Dict[str, int]]):
    """把多种食材表示转换成固定维度的 0/1 向量。

    支持输入：
    - 已经是 dim 维的 Tensor；
    - ingredient 名称字符串或名称列表；
    - ingredient index 列表；
    - 包含 ingredients/indices 等字段的字典。
    """
    vec = torch.zeros(dim, dtype=torch.float32)
    if obj is None:
        return vec

    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu()
        if t.numel() == dim and t.dtype != torch.long:
            # 已经是 dense/binary 向量时直接拉平成 dim 维。
            return t.reshape(dim).float()
        if t.dtype.is_floating_point and t.numel() != dim:
            raise ValueError(
                f"binary255 expects ingredient indices/names or a 255-d vector, got a float tensor with {t.numel()} elements."
            )
        obj = t.reshape(-1).tolist()

    if isinstance(obj, dict):
        for k in ("ingredients", "ingredient", "indices", "ids", "list"):
            if k in obj:
                return _ingredients_to_binary_vector(obj[k], dim=dim, vocab=vocab)
        return vec

    if isinstance(obj, str):
        # CSV/JSON 里常见的 "a,b,c" 或 "a;b;c" 形式统一拆成列表。
        parts = [p.strip() for p in obj.replace(";", ",").split(",")]
        obj = [p for p in parts if p]

    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            if item is None:
                continue
            if isinstance(item, str):
                if vocab is None:
                    raise ValueError("Binary ingredients needs --ingredients_vocab when items are strings.")
                idx = vocab.get(item)
                if idx is None:
                    # 不在词表中的食材跳过，避免因少量未知名称中断整个数据集。
                    continue
            else:
                if isinstance(item, float):
                    # Avoid accidentally treating dense float features as indices.
                    if abs(item - round(item)) > 1e-6:
                        raise ValueError(
                            "binary255 expects ingredient indices (ints) or names (strings), got non-integer floats."
                        )
                idx = int(item)
            if 0 <= idx < dim:
                vec[idx] = 1.0
        return vec

    return vec


def _find_first_existing(paths):
    """按优先级返回第一个存在的文件路径。"""
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return ""


def _load_nutrition5k_binary_labels_from_csv(csv_dir: str, dim: int):
    """从 Nutrition5k 官方 metadata CSV 中构建 binary255 标签。

    ingredients_metadata 提供食材名称与原始 id；两个 dish_metadata 文件提供
    每个 dish 的食材列表。这里按原始 id 和名称排序生成稳定词表，并缓存结果，
    避免 train/test Dataset 重复解析大 CSV。
    """
    cache_key = (os.path.abspath(csv_dir), int(dim))
    if cache_key in _NUTRITION5K_CACHE:
        return _NUTRITION5K_CACHE[cache_key]

    ingredients_meta = os.path.join(csv_dir, "nutrition5k_dataset_metadata_ingredients_metadata.csv")
    cafe1 = os.path.join(csv_dir, "nutrition5k_dataset_metadata_dish_metadata_cafe1.csv")
    cafe2 = os.path.join(csv_dir, "nutrition5k_dataset_metadata_dish_metadata_cafe2.csv")

    if not (os.path.isfile(ingredients_meta) and os.path.isfile(cafe1) and os.path.isfile(cafe2)):
        raise FileNotFoundError(
            "Missing Nutrition5k CSVs. Expected:\n"
            f"- {ingredients_meta}\n- {cafe1}\n- {cafe2}"
        )

    name_to_id = {}
    with open(ingredients_meta, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            name = (row.get("ingr") or "").strip()
            if not name:
                continue
            try:
                name_to_id[name] = int(row.get("id"))
            except Exception:
                continue

    def iter_dish_rows(path):
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            for row in r:
                if row:
                    yield row

    dish_to_names = {}
    used_names = set()
    for path in (cafe1, cafe2):
        for row in iter_dish_rows(path):
            dish_id = (row[0] or "").strip()
            if not dish_id:
                continue
            names = []
            i = 6  # dish_id + 5 totals
            while i + 1 < len(row):
                # 每个食材块结构为 ingr_code, ingr_name, calorie, mass, fat, carb, protein。
                name = (row[i + 1] or "").strip()
                if name:
                    names.append(name)
                i += 7  # ingr_code, ingr_name, then 5 numbers
            if not names:
                continue
            uniq = sorted(set(names))
            dish_to_names[dish_id] = uniq
            used_names.update(uniq)

    vocab_names = sorted(used_names, key=lambda n: (name_to_id.get(n, 10**9), n))
    if len(vocab_names) > dim:
        raise ValueError(f"Nutrition5k unique ingredients={len(vocab_names)} exceeds dim={dim}.")

    vocab = {name: i for i, name in enumerate(vocab_names)}
    dish_to_indices = {d: [vocab[n] for n in names if n in vocab] for d, names in dish_to_names.items()}

    _NUTRITION5K_CACHE[cache_key] = (vocab, dish_to_indices)
    return vocab, dish_to_indices

def random_unit(p):
    """按给定概率 p 返回 True，用于简单数据增强开关。"""
    assert p >= 0 and p <= 1, "概率P的值应该处在[0,1]之间！"
    if p == 0:  # 概率为0，直接返回False
        return False
    if p == 1:  # 概率为1，直接返回True
        return True
    p_digits = len(str(p).split(".")[1])
    interval_begin = 1
    interval__end = pow(10, p_digits)
    R = random.randint(interval_begin, interval__end)
    if float(R)/interval__end < p:
        return True
    else:
        return False

# RGB-D
class Nutrition_RGBD(Dataset):
    """Nutrition5k RGB-D 样本读取器。

    rgb_txt_dir 与 rgbd_txt_dir 分别对应两个模态的样本列表；列表行里包含图像
    相对路径、dish_id/label 以及五个营养真值。__getitem__ 会按相同 index
    取 RGB 与 depth/RGB-D 图像，并附加对应 dish 的食材向量。
    """

    def __init__(
        self,
        image_path,
        rgb_txt_dir,
        rgbd_txt_dir,
        training,
        transform=None,
        ingredients_mode: str = "clip512",
        ingredients_dim: int = 255,
        ingredients_vocab_path: str = "",
    ):
        # 两个列表必须一一对应：同一个 index 表示同一道菜的 RGB 与 depth/RGB-D 视角。
        file_rgb = open(rgb_txt_dir, 'r')
        file_rgbd = open(rgbd_txt_dir, 'r')
        lines_rgb = file_rgb.readlines()
        lines_rgbd = file_rgbd.readlines()
        self.images = []
        self.labels = []
        self.total_calories = []
        self.total_mass = []
        self.total_fat = []
        self.total_carb = []
        self.total_protein = []
        self.images_rgbd = []
        # pdb.set_trace()
        for line in lines_rgb:
            image_rgb = line.split()[0]  # side_angles/dish_1550862840/frames_sampled5/camera_A_frame_010.jpeg
            label = line.strip().split()[1]  # 类别 1-
            calories = line.strip().split()[2]
            mass = line.strip().split()[3]
            fat = line.strip().split()[4]
            carb = line.strip().split()[5]
            protein = line.strip().split()[6]

            self.images += [os.path.join(image_path, image_rgb)]  # 每张图片路径
            self.labels += [str(label)]
            self.total_calories += [np.array(float(calories))]
            self.total_mass += [np.array(float(mass))]
            self.total_fat += [np.array(float(fat))]
            self.total_carb += [np.array(float(carb))]
            self.total_protein += [np.array(float(protein))]
        for line in lines_rgbd:
            # depth/RGB-D 列表只需要图像路径，营养真值以 RGB 列表为准。
            image_rgbd = line.split()[0]
            self.images_rgbd += [os.path.join(image_path, image_rgbd)]

            # pdb.set_trace()
        # self.transform_rgb = transform[0]
        self.training = training
        self.transform = transform
        self.image_path = image_path
        self.ingredients_mode = ingredients_mode
        self.ingredients_dim = int(ingredients_dim)
        self.ingredients_vocab = _load_ingredients_vocab(ingredients_vocab_path)
        self._dish_to_ingredients = None
        if self.ingredients_mode == "binary255" and self.ingredients_vocab is None:
            # 未显式提供 vocab 时，尝试在项目根、数据根和 imagery 目录附近寻找官方 CSV。
            candidates = [
                os.getcwd(),
                os.path.dirname(os.path.abspath(self.image_path)),
                os.path.abspath(self.image_path),
            ]
            meta_path = _find_first_existing(
                [os.path.join(d, "nutrition5k_dataset_metadata_ingredients_metadata.csv") for d in candidates]
            )
            if meta_path:
                csv_dir = os.path.dirname(meta_path)
                self.ingredients_vocab, self._dish_to_ingredients = _load_nutrition5k_binary_labels_from_csv(
                    csv_dir=csv_dir,
                    dim=self.ingredients_dim,
                )

    # RGB-D  20210805
    def my_loader(path, Type):
        """按指定通道数读取 PIL 图像；Type=3 为 RGB，Type=1 为灰度。"""
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                if Type == 3:
                    img = img.convert('RGB')
                elif Type == 1:
                    img = img.convert('L')
                return img

    def __getitem__(self, index):
        # 读取成 PIL Image，后续统一交给 torchvision transform 转 Tensor/Normalize。
        img_rgb = Image.open(self.images[index])
        img_rgbd = Image.open(self.images_rgbd[index])
        dish_id = os.path.basename(os.path.dirname(self.images[index]))
        ingredients_path = os.path.join(self.image_path, "ingredients_fea_nonorm", f"{dish_id}.pth")
        # ingredients_path = os.path.join(self.image_path, "ingredients_fea", f"{dish_id}.pth")

        ingredients_obj = None
        if self._dish_to_ingredients is not None and dish_id in self._dish_to_ingredients:
            # binary255 模式优先使用从 Nutrition5k CSV 解析出的 dish -> ingredient indices。
            ingredients_obj = self._dish_to_ingredients[dish_id]
        elif os.path.isfile(ingredients_path):
            # clip512 模式常用 .pth 文件保存 CLIP 文本编码；也兼容保存索引/向量的 .pth。
            ingredients_obj = torch.load(ingredients_path)
        else:
            json_path = os.path.splitext(ingredients_path)[0] + ".json"
            if os.path.isfile(json_path):
                # JSON 作为轻量 sidecar，可保存名称列表或索引列表。
                with open(json_path, "r", encoding="utf-8") as f:
                    ingredients_obj = json.load(f)

        if self.ingredients_mode == "binary255":
            if ingredients_obj is None:
                raise FileNotFoundError(
                    f"Missing binary ingredients for {dish_id}. Provide Nutrition5k CSVs, a sidecar JSON, or an indices list."
                )
            ingredients_tensor = _ingredients_to_binary_vector(
                ingredients_obj,
                dim=self.ingredients_dim,
                vocab=self.ingredients_vocab,
            )
        else:
            if ingredients_obj is None:
                raise FileNotFoundError(f"Missing ingredients feature: {ingredients_path}")
            ingredients_tensor = ingredients_obj
            if isinstance(ingredients_tensor, torch.Tensor):
                # CLIP 文本特征常保存为 [1, 512]，训练时需要压成 [512]。
                ingredients_tensor = torch.squeeze(ingredients_tensor)
        #print(ingredients_tensor.shape)
        # try:
        #     # img = cv2.resize(img, (self.imsize, self.imsize))
        #     img_rgb = Image.fromarray(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))  # cv2转PIL
        #     img_rgbd = Image.fromarray(cv2.cvtColor(img_rgbd, cv2.COLOR_BGR2RGB))  # cv2转PIL
        # except:
        #     print("图片有误：", self.images[index])

        if self.training == True:
            if random_unit(0.5) == True:
                # RGB 与 depth/RGB-D 必须同步做几何增强，保持跨模态像素对应。
                img_rgb = img_rgb.transpose(Image.FLIP_LEFT_RIGHT)
                img_rgbd = img_rgbd.transpose(Image.FLIP_LEFT_RIGHT)

            if random_unit(0.5) == True:
                rotate_degree = random.randint(0, 360)
                img_rgb = img_rgb.rotate(rotate_degree, expand = 1)
                img_rgbd = img_rgbd.rotate(rotate_degree, expand = 1)
                img_rgb = img_rgb.resize((416,416))
                img_rgbd = img_rgbd.resize((416, 416))


        if self.transform is not None:
            # transform 内通常包含 ToTensor + ImageNet Normalize。
            img_rgb = self.transform(img_rgb)
            img_rgbd = self.transform(img_rgbd)



        return img_rgb, self.labels[index], self.total_calories[index], self.total_mass[index], self.total_fat[index], \
        self.total_carb[index], self.total_protein[index], img_rgbd, ingredients_tensor

    def __len__(self):
        return len(self.images)
