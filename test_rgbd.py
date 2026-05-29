from __future__ import print_function

import argparse
import os
import csv
import subprocess
import types
from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from mydataset import Nutrition_RGBD
from mynetwork import MyResNetRGBD, MyResNetRGBDLegacy


"""
IMIR-Net RGB-D 推理与评估脚本。

脚本根据 checkpoint 自动选择当前 fusion 模型、旧版 legacy 模型，或历史
CA_SA+ingredients 模型；随后对测试集逐样本预测，并在 dish 级别聚合多帧/
多视角结果，最后计算 PMAE = sum(|pred-gt|) / sum(gt) * 100。
"""

METRICS = ["calories", "mass", "fat", "carb", "protein"]
HIST_CASA_ING_COMMIT = "8ed3b23"
_HIST_MODULE_CACHE = {}


def to_scalar(x):
    """把 Tensor 或普通数值统一转成 Python float，方便 numpy/CSV 聚合。"""
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def _strip_module_prefix(state_dict):
    """去掉 DataParallel 保存权重时常见的 module. 前缀。"""
    if not isinstance(state_dict, (dict, OrderedDict)):
        return state_dict
    out = OrderedDict()
    for k, v in state_dict.items():
        if isinstance(k, str) and k.startswith("module."):
            k = k[7:]
        out[k] = v
    return out


def load_checkpoint(model, checkpoint_path, device):
    """加载 checkpoint 到指定模型。

    该函数兼容训练 checkpoint {"state_dict": ...}、裸 state_dict 以及仅包含
    Food2K backbone 的预训练权重。当前 main 路径主要使用 build_model_for_checkpoint，
    这里保留给手动指定模型的场景。
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Common training checkpoint format: {"epoch": ..., "state_dict": ...}
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = _strip_module_prefix(ckpt["state_dict"])
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"[WARN] load_state_dict(strict=True) failed: {e}")
            print(f"[WARN] Retried with strict=False. missing={len(missing)} unexpected={len(unexpected)}")
        return

    # Some checkpoints are directly a state_dict (OrderedDict of parameter tensors).
    if isinstance(ckpt, (dict, OrderedDict)):
        state_dict = _strip_module_prefix(ckpt)

        # If this is a plain ResNet state_dict (e.g., Food2K) without model_rgb/model_depth prefixes,
        # load it into both backbones (this is what MyResNetRGBD expects as pretrain).
        if (
            hasattr(model, "model_rgb")
            and hasattr(model, "model_depth")
            and any(isinstance(k, str) and not k.startswith(("model_rgb.", "model_depth.")) for k in state_dict.keys())
        ):
            model.model_rgb.load_state_dict(state_dict, strict=False)
            model.model_depth.load_state_dict(state_dict, strict=False)
            print(f"[INFO] Loaded backbone weights into model_rgb/model_depth from: {checkpoint_path}")
            print("[WARN] This checkpoint does not include the IMIR-Net regression head; use ./saved/ckpt_best.pth for meaningful nutrition predictions.")
            return

        # Fallback: try to load into the full model.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded checkpoint with strict=False from: {checkpoint_path} (missing={len(missing)} unexpected={len(unexpected)})")
        return

    raise ValueError(f"Unrecognized checkpoint format at: {checkpoint_path}")


def _extract_state_dict(ckpt):
    """从训练 checkpoint 或裸权重中抽取统一的 state_dict。"""
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return _strip_module_prefix(ckpt["state_dict"])
    if isinstance(ckpt, (dict, OrderedDict)):
        return _strip_module_prefix(ckpt)
    raise ValueError("Unrecognized checkpoint format (no state_dict).")


def _infer_ingredients_dim_from_state_dict(state_dict):
    """通过 ingredients_fc1.weight 的输入维度判断 checkpoint 需要的食材向量维度。"""
    w = state_dict.get("ingredients_fc1.weight")
    if isinstance(w, torch.Tensor) and w.dim() == 2:
        return int(w.shape[1])
    return None


def _has_prefix(state_dict, prefixes):
    """检查权重 key 中是否存在指定模块前缀，用于自动识别模型变体。"""
    return any(isinstance(k, str) and k.startswith(prefixes) for k in state_dict.keys())


def _load_historical_mynetwork_module(commit):
    """从 git 历史中加载旧版 mynetwork.py，兼容早期 CA_SA+ingredients checkpoint。"""
    if commit in _HIST_MODULE_CACHE:
        return _HIST_MODULE_CACHE[commit]
    try:
        raw = subprocess.check_output(["git", "show", f"{commit}:mynetwork.py"])
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load historical mynetwork.py at commit {commit}. "
            "Please ensure this repo has git history available."
        ) from exc
    source = raw.decode("utf-8", "ignore")
    module = types.ModuleType(f"hist_mynetwork_{commit}")
    module.__file__ = f"<git:{commit}:mynetwork.py>"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    _HIST_MODULE_CACHE[commit] = module
    return module


def _build_historical_casa_ingredients_model(ingredients_dim, device):
    """实例化历史版本中的 MyResNetRGBD。"""
    hist_module = _load_historical_mynetwork_module(HIST_CASA_ING_COMMIT)
    model_cls = getattr(hist_module, "MyResNetRGBD", None)
    if model_cls is None:
        raise RuntimeError(
            f"Historical mynetwork.py at {HIST_CASA_ING_COMMIT} does not define MyResNetRGBD."
        )
    return model_cls(ingredients_dim=ingredients_dim).to(device)


def build_model_for_checkpoint(args, device):
    """根据 checkpoint key 自动选择并加载匹配的模型定义。

    判断依据：
    - 有 ingredients_fc/fusion_fc：说明 checkpoint 接收食材输入。
    - 有 CA_SA_Enhance 且没有 MBFM：说明属于历史 CA_SA+ingredients 版本。
    - 没有食材融合头：使用 legacy RGB-D 模型。
    """
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = _extract_state_dict(ckpt)
    has_ing_fusion_head = _has_prefix(state_dict, ("ingredients_fc1.", "fusion_fc."))
    has_casa = _has_prefix(state_dict, ("CA_SA_Enhance_3.", "CA_SA_Enhance_4."))
    has_mbfm = _has_prefix(state_dict, ("mbfm_l3.", "mbfm_l4."))

    if args.model_variant == "legacy":
        variant = "legacy"
    elif args.model_variant == "fusion":
        variant = "fusion"
    elif args.model_variant == "ca_sa_ingredients":
        variant = "ca_sa_ingredients"
    else:
        if has_ing_fusion_head and has_casa and not has_mbfm:
            variant = "ca_sa_ingredients"
        elif has_ing_fusion_head:
            variant = "fusion"
        else:
            variant = "legacy"

    if variant == "legacy":
        model = MyResNetRGBDLegacy().to(device)
    elif variant == "ca_sa_ingredients":
        inferred_dim = _infer_ingredients_dim_from_state_dict(state_dict)
        ingredients_dim = inferred_dim if inferred_dim is not None else (255 if args.ingredients_mode == "binary255" else 512)
        model = _build_historical_casa_ingredients_model(ingredients_dim=ingredients_dim, device=device)
    else:
        inferred_dim = _infer_ingredients_dim_from_state_dict(state_dict)
        ingredients_dim = inferred_dim if inferred_dim is not None else (255 if args.ingredients_mode == "binary255" else 512)
        model = MyResNetRGBD(ingredients_dim=ingredients_dim).to(device)

    # load weights
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        payload = _strip_module_prefix(ckpt["state_dict"])
    else:
        payload = state_dict

    try:
        model.load_state_dict(payload, strict=True)
    except RuntimeError as e:
        print(f"[WARN] load_state_dict(strict=True) failed: {e}")
        # strict=False 仍会因 shape mismatch 报错，所以先过滤掉尺寸不匹配的权重。
        # Remove incompatible shapes (strict=False still errors on size mismatch).
        model_sd = model.state_dict()
        filtered = {k: v for k, v in payload.items() if k in model_sd and getattr(v, "shape", None) == getattr(model_sd[k], "shape", None)}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"[WARN] Retried with filtered strict=False. missing={len(missing)} unexpected={len(unexpected)}")

    inferred_dim = None if variant == "legacy" else _infer_ingredients_dim_from_state_dict(state_dict)
    inferred_mode = "clip512"
    if inferred_dim == 255:
        inferred_mode = "binary255"
    elif inferred_dim == 512:
        inferred_mode = "clip512"
    return model, variant, inferred_mode, inferred_dim


def main():
    """推理主流程：构建 Dataset/DataLoader，预测，dish 级聚合并输出 PMAE/CSV。"""
    parser = argparse.ArgumentParser(description="IMIR-Net RGB-D inference")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root containing imagery/")
    parser.add_argument("--checkpoint", type=str, default="./saved/ckpt_best.pth", help="Path to checkpoint .pth")
    parser.add_argument("--out_csv", type=str, default="", help="If set, write dish-level predictions CSV")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--ingredients_mode",
        type=str,
        default="clip512",
        choices=["clip512", "binary255"],
        help="Must match the checkpoint's ingredients head (clip512 or binary255).",
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default="auto",
        choices=["auto", "legacy", "fusion", "ca_sa_ingredients"],
        help="Model definition to use: legacy, fusion (MBFM), or ca_sa_ingredients (historical CA_SA + ingredients).",
    )
    parser.add_argument(
        "--ingredients_vocab",
        type=str,
        default="",
        help="(binary255 only) JSON vocab: list[255] of names or {name: idx}.",
    )
    args = parser.parse_args()

    test_transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    nutrition_rgbd_ims_root = os.path.join(args.data_root, "imagery")
    nutrition_test_txt = os.path.join(args.data_root, "imagery", "rgbd_test_processed1_refine.txt")  # depth_color.png
    nutrition_test_rgbd_txt = os.path.join(args.data_root, "imagery", "rgb_in_overhead_test_processed1_refine.txt")  # rgb.png

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, variant, inferred_ingredients_mode, inferred_ingredients_dim = build_model_for_checkpoint(args, device)
    print(f"[INFO] Selected model_variant: {variant}")
    model.eval()

    # For fusion checkpoints, auto-align dataset ingredients format/dim to checkpoint.
    # checkpoint 的 ingredients_fc1 输入维度优先级最高，避免命令行参数与权重不一致。
    dataset_ingredients_mode = inferred_ingredients_mode if variant in ("fusion", "ca_sa_ingredients") else args.ingredients_mode
    dataset_ingredients_dim = (inferred_ingredients_dim if inferred_ingredients_dim is not None else (255 if dataset_ingredients_mode == "binary255" else 512))

    testset = Nutrition_RGBD(
        nutrition_rgbd_ims_root,
        nutrition_test_rgbd_txt,
        nutrition_test_txt,
        training=False,
        transform=test_transform,
        ingredients_mode=dataset_ingredients_mode,
        ingredients_dim=dataset_ingredients_dim,
        ingredients_vocab_path=args.ingredients_vocab,
    )

    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    epoch_iterator = tqdm(
        testloader,
        desc="Testing... (loss=X.X)",
        bar_format="{l_bar}{r_bar}",
        dynamic_ncols=True,
    )

    pred_sum = {}
    pred_cnt = {}
    gt_sum = {}
    gt_cnt = {}

    with torch.no_grad():
        for _, x in enumerate(epoch_iterator):
            # Dataset 字段顺序与训练保持一致；推理时 label/dish_id 用于 dish 级聚合。
            inputs = x[0].to(device)
            total_calories = x[2].to(device).float().view(-1)
            total_mass = x[3].to(device).float().view(-1)
            total_fat = x[4].to(device).float().view(-1)
            total_carb = x[5].to(device).float().view(-1)
            total_protein = x[6].to(device).float().view(-1)
            inputs_rgbd = x[7].to(device)
            inputs_ingredients = x[8].to(device)

            if variant == "legacy":
                # legacy forward 不接收食材向量，并可能返回 (results, aux_feature)。
                out = model(inputs, inputs_rgbd)
                outputs = out[0] if isinstance(out, (tuple, list)) and len(out) == 2 else out
            else:
                outputs = model(inputs, inputs_rgbd, inputs_ingredients)

            preds = []
            for k in range(5):
                # batch_size=1 时 squeeze 后可能是 0 维 Tensor，这里统一恢复成一维。
                out_k = outputs[k]
                if isinstance(out_k, torch.Tensor) and out_k.dim() == 0:
                    out_k = out_k.view(1)
                else:
                    out_k = out_k.view(-1)
                preds.append(out_k)

            ids = x[1]
            if not isinstance(ids, (list, tuple)):
                ids = [ids]

            bs = int(total_calories.shape[0])
            for i in range(bs):
                dish_id = str(ids[i])
                # 同一个 dish 可能有多帧/多视角，先累加，后面再取平均。
                pred = np.array([to_scalar(preds[k][i]) for k in range(5)], dtype=np.float64)
                gt = np.array(
                    [
                        to_scalar(total_calories[i]),
                        to_scalar(total_mass[i]),
                        to_scalar(total_fat[i]),
                        to_scalar(total_carb[i]),
                        to_scalar(total_protein[i]),
                    ],
                    dtype=np.float64,
                )

                pred_sum[dish_id] = pred_sum.get(dish_id, 0.0) + pred
                pred_cnt[dish_id] = pred_cnt.get(dish_id, 0) + 1
                gt_sum[dish_id] = gt_sum.get(dish_id, 0.0) + gt
                gt_cnt[dish_id] = gt_cnt.get(dish_id, 0) + 1

    dishes = sorted(set(pred_sum.keys()) & set(gt_sum.keys()))
    # 按 dish 平均后再评估，避免多视角样本数量影响单道菜权重。
    pred_mean = {d: (pred_sum[d] / max(1, pred_cnt[d])) for d in dishes}
    gt_mean = {d: (gt_sum[d] / max(1, gt_cnt[d])) for d in dishes}

    # PMAE definition (the one you validated earlier): sum(|pred-gt|)/sum(gt) * 100, dish-level.
    pmae = {}
    for j, name in enumerate(METRICS):
        # PMAE 按指标分别计算：所有 dish 的绝对误差总和 / 真值总和。
        se = 0.0
        sg = 0.0
        for d in dishes:
            p = float(pred_mean[d][j])
            g = float(gt_mean[d][j])
            se += abs(p - g)
            sg += g
        pmae[name] = (se / max(sg, 1e-12)) * 100.0

    avg_pmae = sum(pmae.values()) / len(METRICS)
    print("Checkpoint:", args.checkpoint)
    print("PMAE sum(|e|)/sum(gt) (%):", pmae)
    print("AVG_PMAE (%):", avg_pmae)

    if args.out_csv:
        headers = ["dish_id"] + METRICS
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for d in dishes:
                row = [d] + [float(x) for x in pred_mean[d].tolist()]
                writer.writerow(row)

        print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
