from __future__ import print_function

import argparse
import os
import csv
from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from mydataset import Nutrition_RGBD
from mynetwork import MyResNetRGBD, MyResNetRGBDLegacy


METRICS = ["calories", "mass", "fat", "carb", "protein"]


def to_scalar(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, (dict, OrderedDict)):
        return state_dict
    out = OrderedDict()
    for k, v in state_dict.items():
        if isinstance(k, str) and k.startswith("module."):
            k = k[7:]
        out[k] = v
    return out


def load_checkpoint(model, checkpoint_path, device):
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

        # If this looks like a bare backbone checkpoint (not a full IMIR-Net checkpoint),
        # we can only attempt to load what matches and ignore the rest.
        if any(isinstance(k, str) and not k.startswith(("clip_", "mbfm_", "encoder_", "con2d", "ingredients_", "fusion_", "calorie", "mass", "fat", "carb", "protein")) for k in state_dict.keys()):
            print(
                "[WARN] Checkpoint may not be a full IMIR-Net checkpoint; loading with strict=False and ignoring mismatched keys."
            )

        # Fallback: try to load into the full model.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded checkpoint with strict=False from: {checkpoint_path} (missing={len(missing)} unexpected={len(unexpected)})")
        return

    raise ValueError(f"Unrecognized checkpoint format at: {checkpoint_path}")


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return _strip_module_prefix(ckpt["state_dict"])
    if isinstance(ckpt, (dict, OrderedDict)):
        return _strip_module_prefix(ckpt)
    raise ValueError("Unrecognized checkpoint format (no state_dict).")


def _infer_ingredients_dim_from_state_dict(state_dict):
    w = state_dict.get("ingredients_fc1.weight")
    if isinstance(w, torch.Tensor) and w.dim() == 2:
        return int(w.shape[1])
    return None


def build_model_for_checkpoint(args, device):
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = _extract_state_dict(ckpt)

    if args.model_variant == "legacy":
        variant = "legacy"
    elif args.model_variant == "fusion":
        variant = "fusion"
    else:
        variant = "fusion" if any(k.startswith(("ingredients_fc1.", "fusion_fc.")) for k in state_dict.keys()) else "legacy"

    if variant == "legacy":
        model = MyResNetRGBDLegacy().to(device)
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
        choices=["auto", "legacy", "fusion"],
        help="Which model definition to use for the checkpoint: legacy (old checkpoints) or fusion (ingredients-gated).",
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
    model.eval()

    # For fusion checkpoints, auto-align dataset ingredients format/dim to checkpoint.
    dataset_ingredients_mode = inferred_ingredients_mode if variant == "fusion" else args.ingredients_mode
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
            inputs = x[0].to(device)
            total_calories = x[2].to(device).float().view(-1)
            total_mass = x[3].to(device).float().view(-1)
            total_fat = x[4].to(device).float().view(-1)
            total_carb = x[5].to(device).float().view(-1)
            total_protein = x[6].to(device).float().view(-1)
            inputs_rgbd = x[7].to(device)
            inputs_ingredients = x[8].to(device)

            if variant == "legacy":
                out = model(inputs, inputs_rgbd)
                outputs = out[0] if isinstance(out, (tuple, list)) and len(out) == 2 else out
            else:
                outputs = model(inputs, inputs_rgbd, inputs_ingredients)

            preds = []
            for k in range(5):
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
    pred_mean = {d: (pred_sum[d] / max(1, pred_cnt[d])) for d in dishes}
    gt_mean = {d: (gt_sum[d] / max(1, gt_cnt[d])) for d in dishes}

    # PMAE definition (the one you validated earlier): sum(|pred-gt|)/sum(gt) * 100, dish-level.
    pmae = {}
    for j, name in enumerate(METRICS):
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
