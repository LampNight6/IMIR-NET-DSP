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


_NUTRITION5K_CACHE = {
    # key: (csv_dir, dim) -> (vocab: {name->idx}, dish_map: {dish_id->list[idx]})
}


def _load_ingredients_vocab(vocab_path: str) -> Optional[Dict[str, int]]:
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
    vec = torch.zeros(dim, dtype=torch.float32)
    if obj is None:
        return vec

    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu()
        if t.numel() == dim and t.dtype != torch.long:
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
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return ""


def _load_nutrition5k_binary_labels_from_csv(csv_dir: str, dim: int):
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
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                if Type == 3:
                    img = img.convert('RGB')
                elif Type == 1:
                    img = img.convert('L')
                return img

    def __getitem__(self, index):
        img_rgb = Image.open(self.images[index])
        img_rgbd = Image.open(self.images_rgbd[index])
        dish_id = os.path.basename(os.path.dirname(self.images[index]))
        ingredients_path = os.path.join(self.image_path, "ingredients_fea_nonorm", f"{dish_id}.pth")
        # ingredients_path = os.path.join(self.image_path, "ingredients_fea", f"{dish_id}.pth")

        ingredients_obj = None
        if self._dish_to_ingredients is not None and dish_id in self._dish_to_ingredients:
            ingredients_obj = self._dish_to_ingredients[dish_id]
        elif os.path.isfile(ingredients_path):
            ingredients_obj = torch.load(ingredients_path)
        else:
            json_path = os.path.splitext(ingredients_path)[0] + ".json"
            if os.path.isfile(json_path):
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
                img_rgb = img_rgb.transpose(Image.FLIP_LEFT_RIGHT)
                img_rgbd = img_rgbd.transpose(Image.FLIP_LEFT_RIGHT)

            if random_unit(0.5) == True:
                rotate_degree = random.randint(0, 360)
                img_rgb = img_rgb.rotate(rotate_degree, expand = 1)
                img_rgbd = img_rgbd.rotate(rotate_degree, expand = 1)
                img_rgb = img_rgb.resize((416,416))
                img_rgbd = img_rgbd.resize((416, 416))


        if self.transform is not None:
            img_rgb = self.transform(img_rgb)
            img_rgbd = self.transform(img_rgbd)



        return img_rgb, self.labels[index], self.total_calories[index], self.total_mass[index], self.total_fat[index], \
        self.total_carb[index], self.total_protein[index], img_rgbd, ingredients_tensor

    def __len__(self):
        return len(self.images)
