from __future__ import print_function

import argparse
import csv
import os
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from mydataset import Nutrition_RGBD
from mynetwork import MyResNetRGBD

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

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        payload = _strip_module_prefix(ckpt["state_dict"])
    elif isinstance(ckpt, (dict, OrderedDict)):
        payload = _strip_module_prefix(ckpt)
    else:
        raise RuntimeError("Unsupported checkpoint format")

    try:
        model.load_state_dict(payload, strict=True)
    except RuntimeError as e:
        print(f"[WARN] load_state_dict(strict=True) failed: {e}")
        model_sd = model.state_dict()
        filtered = {
            k: v
            for k, v in payload.items()
            if k in model_sd and getattr(v, "shape", None) == getattr(model_sd[k], "shape", None)
        }
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"[WARN] Retried with filtered strict=False. missing={len(missing)} unexpected={len(unexpected)}")


def main():
    parser = argparse.ArgumentParser(description="IMIR-Net RGB-D inference")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root containing imagery/")
    parser.add_argument("--checkpoint", type=str, default="./saved/ckpt_best.pth", help="Path to checkpoint .pth")
    parser.add_argument("--out_csv", type=str, default="epoch_result_dish.csv", help="Write dish-level predictions CSV")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)

    # Compatibility args: accepted but not used in this repository version.
    parser.add_argument("--model_variant", type=str, default="auto")
    parser.add_argument("--ingredients_mode", type=str, default="clip512")
    parser.add_argument("--ingredients_vocab", type=str, default="")

    args, _ = parser.parse_known_args()

    test_transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    nutrition_rgbd_ims_root = os.path.join(args.data_root, "imagery")
    nutrition_test_txt = os.path.join(args.data_root, "imagery", "rgbd_test_processed1_refine.txt")
    nutrition_test_rgbd_txt = os.path.join(args.data_root, "imagery", "rgb_in_overhead_test_processed1_refine.txt")

    testset = Nutrition_RGBD(
        nutrition_rgbd_ims_root,
        nutrition_test_rgbd_txt,
        nutrition_test_txt,
        training=False,
        transform=test_transform,
    )

    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyResNetRGBD().to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

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

            out = model(inputs, inputs_rgbd)
            outputs = out[0] if isinstance(out, (tuple, list)) and len(out) == 2 else out

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
