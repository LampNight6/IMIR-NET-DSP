import argparse
import os

import numpy as np
import torch


def get_weights_vector(model_state_dict, key_suffix):
    for key in model_state_dict.keys():
        if isinstance(key, str) and key.endswith(key_suffix):
            return model_state_dict[key].detach().cpu().numpy().ravel()
    return None


def main():
    parser = argparse.ArgumentParser(description="Check whether a checkpoint was initialized from Food2K pretrain.")
    parser.add_argument("--checkpoint", default=r".\\saved\\ckpt_best.pth")
    parser.add_argument("--food2k", default=r".\\food2k_resnet101_0.0001.pth")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    meta = ckpt.get("meta") if isinstance(ckpt, dict) else None
    if isinstance(meta, dict):
        print(f"checkpoint: {args.checkpoint}")
        print(f"pretrained_backbone: {meta.get('pretrained_backbone','')}")
        print(f"pretrained_path: {meta.get('pretrained_path','')}")
        print(f"run_name: {meta.get('run_name','')}")
        print(f"model: {meta.get('model','')}")
        return

    if not os.path.isfile(args.food2k):
        raise FileNotFoundError(f"Food2K weights not found: {args.food2k}")
    food2k_sd = torch.load(args.food2k, map_location="cpu")

    layers = [
        "conv1.weight",
        "layer1.0.conv1.weight",
        "layer2.0.conv1.weight",
        "layer3.0.conv1.weight",
    ]

    print(f"checkpoint: {args.checkpoint}")
    print("meta: (missing)")
    print(f"food2k: {args.food2k}")

    for prefix in ("model_rgb.", "model_depth."):
        printed = False
        for layer in layers:
            target_vec = get_weights_vector(state_dict, prefix + layer)
            ref_vec = get_weights_vector(food2k_sd, layer)
            if target_vec is None or ref_vec is None:
                continue
            dist = float(np.linalg.norm(target_vec - ref_vec))
            rel = dist / (float(np.linalg.norm(ref_vec)) + 1e-12)
            print(f"{prefix}{layer}: L2={dist:.6g}, rel={rel:.6g}")
            printed = True
        if not printed:
            print(f"{prefix}*: no comparable layers found in checkpoint.")

    print("NOTE: without meta, this is only a similarity heuristic (fine-tuning changes weights).")


if __name__ == "__main__":
    main()
