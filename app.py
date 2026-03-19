import argparse
import json
import os
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from mynetwork import MyResNetRGBD

# 兼容 huggingface_hub>=1.x（该版本移除了 HfFolder，gradio 旧代码仍会导入）
try:
    import huggingface_hub as _hf_hub

    if not hasattr(_hf_hub, "HfFolder"):
        class _HfFolderCompat:
            @staticmethod
            def path_token():
                from huggingface_hub.constants import HF_TOKEN_PATH
                return HF_TOKEN_PATH

            @staticmethod
            def get_token():
                from huggingface_hub import get_token
                return get_token()

            @staticmethod
            def save_token(token: str):
                token_path = Path(_HfFolderCompat.path_token())
                token_path.parent.mkdir(parents=True, exist_ok=True)
                token_path.write_text(token, encoding="utf-8")

            @staticmethod
            def delete_token():
                token_path = Path(_HfFolderCompat.path_token())
                if token_path.exists():
                    token_path.unlink()

        _hf_hub.HfFolder = _HfFolderCompat
except Exception:
    pass

try:
    import gradio as gr
except ImportError:
    gr = None

try:
    import gradio_client.utils as _gr_client_utils

    _ORIG_GET_TYPE = _gr_client_utils.get_type

    def _safe_get_type(schema):
        # 兼容某些版本中 additionalProperties 为 bool 的 JSON Schema
        # 原 get_type 默认按 dict 处理，遇到 bool 会触发 TypeError。
        if isinstance(schema, bool):
            return "boolean"
        return _ORIG_GET_TYPE(schema)

    _gr_client_utils.get_type = _safe_get_type
except Exception:
    pass

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# 默认路径（均为相对路径）
DEFAULT_CHECKPOINT_PATH = "saved/ckpt_best.pth"
DEFAULT_VOCAB_PATH = "ingredients_vocab.json"
BACKBONE_FILENAME = "food2k_resnet101_0.0001.pth"


# 全局缓存：模型与词表只在系统启动时加载一次，后续复用
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL: Union[MyResNetRGBD, None] = None
_MODEL_CKPT_PATH: Union[Path, None] = None
_VOCAB_LIST: Union[List[str], None] = None
_VOCAB_PATH: Union[Path, None] = None
_EXAMPLE_GT_MAP: Dict[Tuple[str, str], Dict[str, float]] = {}
_EXAMPLE_INFO_GT_MAP: Dict[str, Dict[str, float]] = {}


# 与 test_rgbd.py 保持一致的推理预处理流程
_TEST_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def _strip_module_prefix(state_dict: Dict[str, Any]) -> OrderedDict:
    """移除 DataParallel 常见的 module. 前缀。"""
    out = OrderedDict()
    for k, v in state_dict.items():
        if isinstance(k, str) and k.startswith("module."):
            k = k[7:]
        out[k] = v
    return out


@contextmanager
def _temporary_cwd(path: Path):
    """临时切换工作目录，确保 mynetwork.py 内相对路径能找到 backbone 权重。"""
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def _find_backbone_parent() -> Union[Path, None]:
    """
    查找包含 food2k_resnet101_0.0001.pth 的目录。
    优先当前目录；若不存在，再在上级和常见同级项目目录中寻找。
    """
    local = Path(BACKBONE_FILENAME)
    if local.is_file():
        return Path.cwd()

    script_dir = Path(__file__).resolve().parent
    candidates: List[Path] = []

    # 先查当前脚本目录及其上级
    for p in [script_dir, *list(script_dir.parents)[:3]]:
        candidates.append(p / BACKBONE_FILENAME)

    # 再查常见目录名（如 IMIR-Net / IMIR-Net 2）
    for p in [script_dir, *list(script_dir.parents)[:3]]:
        candidates.extend(p.glob(f"IMIR-Net*/{BACKBONE_FILENAME}"))
        candidates.extend(p.glob(f"IMIR-Net*/IMIR-Net/{BACKBONE_FILENAME}"))

    for cp in candidates:
        if cp.is_file():
            return cp.parent
    return None


def load_model(
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    device: Union[str, torch.device] = _DEVICE,
) -> MyResNetRGBD:
    """
    加载训练好的模型（仅实例化 MyResNetRGBD(ingredients_dim=255)）。
    支持：
    1) checkpoint['state_dict'] 嵌套
    2) module. 前缀
    3) strict=True 失败后的形状过滤回退加载
    """
    global _MODEL, _MODEL_CKPT_PATH, _DEVICE

    device = torch.device(device)
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")

    # 若已加载同一 checkpoint，直接复用
    if (
        _MODEL is not None
        and _MODEL_CKPT_PATH is not None
        and ckpt_path.resolve() == _MODEL_CKPT_PATH.resolve()
        and _DEVICE == device
    ):
        return _MODEL

    # mynetwork.py 在模型构造时会用相对路径加载 food2k_resnet101_0.0001.pth
    # 因此这里先定位该文件，再在对应目录中实例化模型。
    backbone_parent = _find_backbone_parent()
    if backbone_parent is None:
        raise FileNotFoundError(
            "找不到 food2k_resnet101_0.0001.pth。"
            "请将该文件放在项目根目录，或放在上级 IMIR-Net 目录中。"
        )

    with _temporary_cwd(backbone_parent):
        model = MyResNetRGBD(ingredients_dim=255).to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        payload = _strip_module_prefix(ckpt["state_dict"])
    elif isinstance(ckpt, (dict, OrderedDict)):
        payload = _strip_module_prefix(ckpt)
    else:
        raise ValueError(f"不支持的 checkpoint 格式: {ckpt_path}")

    try:
        model.load_state_dict(payload, strict=True)
    except RuntimeError as err:
        # 参考 test_rgbd.py: strict 失败时，过滤掉 shape 不匹配的 key 再 strict=False
        model_sd = model.state_dict()
        filtered = {
            k: v
            for k, v in payload.items()
            if k in model_sd and getattr(v, "shape", None) == getattr(model_sd[k], "shape", None)
        }
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"[WARN] strict=True 加载失败: {err}")
        print(f"[WARN] 已回退到过滤后 strict=False，missing={len(missing)} unexpected={len(unexpected)}")

    model.eval()
    _MODEL = model
    _MODEL_CKPT_PATH = ckpt_path
    _DEVICE = device
    return model


def _to_pil_image(image_input: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
    """兼容 Gradio 常见输入类型：路径 / PIL / numpy。"""
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input)
        return img.convert("RGB")

    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    if isinstance(image_input, np.ndarray):
        arr = image_input

        # 灰度图扩展为 3 通道，保持与模型输入一致
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)

        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"不支持的 numpy 图像形状: {arr.shape}")

        # Gradio 可能给 float32（0~1）或 uint8（0~255），统一转 uint8
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

        return Image.fromarray(arr).convert("RGB")

    raise TypeError(f"不支持的图像输入类型: {type(image_input)}")


def preprocess_image(image_input: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
    """
    图像预处理：
    Resize(416,416) -> ToTensor -> Normalize
    输出 (1,3,H,W) 并放到当前 device。
    """
    img = _to_pil_image(image_input)
    tensor = _TEST_TRANSFORM(img).unsqueeze(0).to(_DEVICE)
    return tensor


def load_vocab(vocab_path: str = DEFAULT_VOCAB_PATH) -> List[str]:
    """加载 ingredients 词表（有序列表），并做全局缓存。"""
    global _VOCAB_LIST, _VOCAB_PATH

    vp = Path(vocab_path)
    if not vp.is_file():
        raise FileNotFoundError(f"找不到词表文件: {vp}")

    if _VOCAB_LIST is not None and _VOCAB_PATH is not None and vp.resolve() == _VOCAB_PATH.resolve():
        return _VOCAB_LIST

    data = json.loads(vp.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"词表格式错误，期望 list，实际为: {type(data)}")

    _VOCAB_LIST = [str(x) for x in data]
    _VOCAB_PATH = vp
    return _VOCAB_LIST


def build_ingredients_vector(
    selected_names: Sequence[str],
    vocab_list: Sequence[str],
    device: Union[str, torch.device],
) -> torch.Tensor:
    """
    将用户勾选成分名转成 (1,255) 二值向量：
    - 勾选项为 1.0
    - 未勾选为 0.0
    """
    device = torch.device(device)
    if len(vocab_list) > 255:
        raise ValueError(f"vocab 长度为 {len(vocab_list)}，超过 255，无法映射到 255 维向量。")

    name_to_idx = {name: idx for idx, name in enumerate(vocab_list)}
    vec = torch.zeros(255, dtype=torch.float32, device=device)
    for name in selected_names:
        idx = name_to_idx.get(name)
        if idx is not None and idx < 255:
            vec[idx] = 1.0
    return vec.unsqueeze(0)


def _scalar(x: Any) -> float:
    """将张量/数值安全转为 Python float。"""
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def predict(
    rgb_path: Union[str, Path, Image.Image, np.ndarray],
    depth_path: Union[str, Path, Image.Image, np.ndarray],
    selected_ingredients: Sequence[str],
) -> Dict[str, float]:
    """
    单次推理：
    1) 预处理 RGB/Depth
    2) 构建 255 维成分向量
    3) 模型推理
    4) 返回 5 项营养值字典
    """
    global _MODEL, _VOCAB_LIST

    if _MODEL is None:
        load_model(DEFAULT_CHECKPOINT_PATH, _DEVICE)
    if _VOCAB_LIST is None:
        load_vocab(DEFAULT_VOCAB_PATH)

    rgb_tensor = preprocess_image(rgb_path)
    depth_tensor = preprocess_image(depth_path)
    ingredients_vector = build_ingredients_vector(selected_ingredients, _VOCAB_LIST, _DEVICE)

    with torch.no_grad():
        outputs = _MODEL(rgb_tensor, depth_tensor, ingredients_vector)

    if not isinstance(outputs, (list, tuple)) or len(outputs) != 5:
        raise RuntimeError(f"模型输出格式不符合预期，期望长度为5的 list/tuple，实际为: {type(outputs)} len={getattr(outputs, '.__len__', lambda: 'N/A')}")

    return {
        "calories": _scalar(outputs[0]),
        "mass": _scalar(outputs[1]),
        "fat": _scalar(outputs[2]),
        "carb": _scalar(outputs[3]),
        "protein": _scalar(outputs[4]),
    }


# =========================
# Gradio 界面：公共配置与工具函数
# =========================
METRIC_KEYS = ["calories", "mass", "fat", "carb", "protein"]
METRIC_LABELS_ZH = {
    "calories": "热量(kcal)",
    "mass": "质量(g)",
    "fat": "脂肪(g)",
    "carb": "碳水(g)",
    "protein": "蛋白质(g)",
}
METRIC_UNITS = {
    "calories": "kcal",
    "mass": "g",
    "fat": "g",
    "carb": "g",
    "protein": "g",
}


def _scan_examples(examples_dir: Union[str, Path] = "examples") -> List[List[Any]]:
    """
    扫描 examples 目录，返回 gr.Examples 所需格式：
    [rgb路径, depth路径, 成分名列表, info.json路径]
    同时构建 ground_truth 索引，供界面结果表与图表做真值对比。
    """
    global _EXAMPLE_GT_MAP, _EXAMPLE_INFO_GT_MAP
    _EXAMPLE_GT_MAP = {}
    _EXAMPLE_INFO_GT_MAP = {}

    root = Path(examples_dir)
    if not root.exists() or not root.is_dir():
        return []

    rows: List[List[Any]] = []
    vocab_set = set(_VOCAB_LIST or [])
    sample_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    for sample_dir in sample_dirs:
        rgb_file = sample_dir / "rgb.png"
        depth_file = sample_dir / "depth.png"
        info_file = sample_dir / "info.json"
        if not rgb_file.is_file() or not depth_file.is_file() or not info_file.is_file():
            continue

        try:
            info = json.loads(info_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        ingredients = info.get("ingredients", [])
        if not isinstance(ingredients, list):
            ingredients = []
        ingredients = [str(x) for x in ingredients]
        if vocab_set:
            ingredients = [x for x in ingredients if x in vocab_set]

        gt = info.get("ground_truth", {})
        if isinstance(gt, dict):
            try:
                gt_clean = {k: float(gt[k]) for k in METRIC_KEYS if k in gt}
            except Exception:
                gt_clean = {}
        else:
            gt_clean = {}

        rgb_abs = str(rgb_file.resolve())
        depth_abs = str(depth_file.resolve())
        info_abs = str(info_file.resolve())
        if gt_clean:
            _EXAMPLE_GT_MAP[(rgb_abs, depth_abs)] = gt_clean
            _EXAMPLE_INFO_GT_MAP[info_abs] = gt_clean

        rows.append([str(rgb_file), str(depth_file), ingredients, str(info_file)])
    return rows


def _resolve_if_pathlike(value: Any) -> Optional[str]:
    """仅当输入是路径类型时解析为绝对路径；其余类型返回 None。"""
    if isinstance(value, (str, Path)) and value:
        try:
            return str(Path(value).resolve())
        except Exception:
            return None
    return None


def _lookup_ground_truth(
    rgb_path: Union[str, Path, Image.Image, np.ndarray],
    depth_path: Union[str, Path, Image.Image, np.ndarray],
) -> Optional[Dict[str, float]]:
    """根据当前输入图像路径查找对应样例真值（如果存在）。"""
    rgb_abs = _resolve_if_pathlike(rgb_path)
    depth_abs = _resolve_if_pathlike(depth_path)
    if not rgb_abs or not depth_abs:
        return None
    return _EXAMPLE_GT_MAP.get((rgb_abs, depth_abs))


def _load_ground_truth_from_info(info_path: Union[str, Path]) -> Optional[Dict[str, float]]:
    """优先根据样例 info.json 读取真值。"""
    info_abs = _resolve_if_pathlike(info_path)
    if not info_abs:
        return None

    if info_abs in _EXAMPLE_INFO_GT_MAP:
        return _EXAMPLE_INFO_GT_MAP[info_abs]

    p = Path(info_abs)
    if not p.is_file():
        return None

    try:
        info = json.loads(p.read_text(encoding="utf-8"))
        gt = info.get("ground_truth", {})
        if not isinstance(gt, dict):
            return None
        gt_clean = {k: float(gt[k]) for k in METRIC_KEYS if k in gt}
        return gt_clean or None
    except Exception:
        return None


def _build_result_table(pred: Dict[str, float], gt: Optional[Dict[str, float]] = None):
    """
    构建表格数据（固定五列，兼容旧版 Gradio Dataframe 更新逻辑）：
    指标名称 / 预测值 / 真实值 / 绝对误差 / PMAE(%)
    """
    rows = []
    for k in METRIC_KEYS:
        p = float(pred[k])
        if gt is not None and k in gt:
            g = float(gt[k])
            err = abs(p - g)
            if abs(g) > 1e-8:
                pmae = round(err / abs(g) * 100.0, 2)
            else:
                # 真值为 0 时，百分比误差无定义，这里置空
                pmae = None
            rows.append([METRIC_LABELS_ZH[k], round(p, 3), round(g, 3), round(err, 3), pmae])
        else:
            rows.append([METRIC_LABELS_ZH[k], round(p, 3), None, None, None])
    return rows


def _build_bar_plot(pred: Dict[str, float], gt: Optional[Dict[str, float]] = None):
    """绘制营养成分柱状图并返回 matplotlib Figure。"""
    if plt is None:
        raise RuntimeError("未安装 matplotlib，请先执行: pip install matplotlib")

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    # 有真值时：按量纲拆成两个子图并列展示，避免热量/质量压缩其他柱子
    if gt:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        groups = [
            (["calories", "mass"], "高量纲指标"),
            (["fat", "carb", "protein"], "低量纲指标"),
        ]
        width = 0.36

        for ax, (keys, subtitle) in zip(axes, groups):
            labels = [METRIC_LABELS_ZH[k] for k in keys]
            pred_vals = [float(pred[k]) for k in keys]
            gt_vals = [float(gt.get(k, 0.0)) for k in keys]
            x = np.arange(len(keys))

            pred_bars = ax.bar(x - width / 2, pred_vals, width=width, color="#4E79A7", label="预测值")
            gt_bars = ax.bar(x + width / 2, gt_vals, width=width, color="#9E9E9E", label="真实值")

            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_title(subtitle)
            ax.set_ylabel("数值")
            ax.grid(False)
            ymax = max(max(pred_vals), max(gt_vals), 1.0)
            ax.set_ylim(0, ymax * 1.25)

            for bars in (pred_bars, gt_bars):
                for b in bars:
                    val = b.get_height()
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        val,
                        f"{val:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

        # 从所有子图收集图例并按标签去重，避免重复显示
        legend_items: Dict[str, Any] = {}
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            for h, l in zip(handles, labels):
                if l and l not in legend_items:
                    legend_items[l] = h
        fig.legend(
            list(legend_items.values()),
            list(legend_items.keys()),
            loc="center left",
            bbox_to_anchor=(0.99, 0.5),
            ncol=1,
            frameon=False,
            handlelength=1.8,
            handletextpad=0.6,
            columnspacing=1.2,
        )
        fig.suptitle("预测值与真实值对比", fontsize=13, y=0.98)
        # 右侧预留图例空间，避免与标题或柱状图文字重叠
        fig.tight_layout(rect=[0, 0, 0.86, 0.9])
        return fig

    # 无真值时：保持原单柱图
    labels = [METRIC_LABELS_ZH[k] for k in METRIC_KEYS]
    values = [float(pred[k]) for k in METRIC_KEYS]
    colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F"]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("营养成分预测结果")
    ax.set_ylabel("预测值")
    ax.grid(False)

    upper = max(values) * 1.15 if max(values) > 0 else 1.0
    ax.set_ylim(0, upper)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    return fig


def _predict_for_ui(
    rgb_image_path: Union[str, Path, Image.Image, np.ndarray],
    depth_image_path: Union[str, Path, Image.Image, np.ndarray],
    selected_ingredients: List[str],
    sample_info_path: Union[str, Path],
):
    """Gradio 按钮回调：输入校验 + 推理 + 表格/图像输出。"""
    if gr is None:
        raise RuntimeError("未安装 gradio，请先执行: pip install gradio")

    if not rgb_image_path:
        raise gr.Error("请先上传 RGB 图像。")
    if not depth_image_path:
        raise gr.Error("请先上传深度图像。")
    if not selected_ingredients:
        raise gr.Error("请至少选择一个成分。")

    pred = predict(rgb_image_path, depth_image_path, selected_ingredients)
    gt = _load_ground_truth_from_info(sample_info_path)
    if gt is None:
        gt = _lookup_ground_truth(rgb_image_path, depth_image_path)
    rows = _build_result_table(pred, gt)
    fig = _build_bar_plot(pred, gt)
    return rows, fig


def _clear_sample_info_state(_):
    """用户手动上传图片后，清空样例 info 绑定状态。"""
    return ""


def build_demo():
    """构建 Gradio Blocks 界面。"""
    if gr is None:
        raise RuntimeError("未安装 gradio，请先执行: pip install gradio")

    # 样例数据（用于底部 gr.Examples）
    example_data = _scan_examples("examples")

    with gr.Blocks(title="基于 RGB-D 图像的食物营养成分估计系统") as demo:
        # 顶部标题区
        gr.Markdown("## 基于 RGB-D 图像的食物营养成分估计系统")
        gr.Markdown(
            "本系统基于改进的 IMIR-Net 模型，通过 RGB 图像、深度图像和食材成分信息预测食物的营养成分。"
            "您可以上传自己的图像，或点击下方样例数据快速体验。"
        )

        # 主体：左右两栏
        with gr.Row():
            # 左栏：输入区
            with gr.Column(scale=1):
                # 使用 PIL 输入可避免 filepath 模式下的本地路径访问限制导致的报错
                rgb_input = gr.Image(label="RGB 图像", type="pil")
                depth_input = gr.Image(label="深度图像", type="pil")
                gr.Markdown("**请选择该菜品包含的食材成分（可多选）**")
                ingredients_input = gr.Dropdown(
                    choices=_VOCAB_LIST or [],
                    label="成分选择",
                    multiselect=True,
                    allow_custom_value=False,
                    filterable=True,
                    info="支持搜索和多选",
                )
                sample_info_path = gr.Textbox(value="", visible=False, label="sample_info_path")
                predict_btn = gr.Button("开始预测", variant="primary")

            # 右栏：输出区
            with gr.Column(scale=1):
                result_df = gr.Dataframe(
                    headers=["指标名称", "预测值", "真实值", "绝对误差", "PMAE(%)"],
                    datatype=["str", "number", "number", "number", "number"],
                    row_count=(5, "fixed"),
                    col_count=(5, "fixed"),
                    label="预测结果",
                    interactive=False,
                )
                result_plot = gr.Plot(label="营养成分柱状图")

        # 交互绑定：点击预测按钮
        predict_btn.click(
            fn=_predict_for_ui,
            inputs=[rgb_input, depth_input, ingredients_input, sample_info_path],
            outputs=[result_df, result_plot],
        )

        # 手动上传时清空样例 info 状态（样例点击不触发 upload 事件）
        rgb_input.upload(fn=_clear_sample_info_state, inputs=[rgb_input], outputs=[sample_info_path])
        depth_input.upload(fn=_clear_sample_info_state, inputs=[depth_input], outputs=[sample_info_path])

        # 底部：样例区
        gr.Markdown("### 样例数据")
        gr.Markdown("点击下方任一样例可自动填充输入数据")
        if example_data:
            gr.Examples(
                examples=example_data,
                inputs=[rgb_input, depth_input, ingredients_input, sample_info_path],
                label="点击样例可自动填充 RGB、深度图、成分与真实值来源",
            )
        else:
            gr.Markdown("当前未检测到可用样例，请检查 `examples/` 目录。")

    return demo


# =========================
# 启动入口：test / server
# =========================
def run_cli_test():
    """命令行测试模式（保留原有逻辑）。"""
    load_vocab(DEFAULT_VOCAB_PATH)
    load_model(DEFAULT_CHECKPOINT_PATH, _DEVICE)

    examples_root = Path("examples")
    sample_dirs = sorted([p for p in examples_root.iterdir() if p.is_dir() and p.name.startswith("sample_")])
    if not sample_dirs:
        raise FileNotFoundError("examples/ 目录下没有找到 sample_xxx 样例目录。")

    sample_dir = sample_dirs[0]
    rgb_file = sample_dir / "rgb.png"
    depth_file = sample_dir / "depth.png"
    info_file = sample_dir / "info.json"

    if not rgb_file.is_file() or not depth_file.is_file() or not info_file.is_file():
        raise FileNotFoundError(f"样例文件不完整: {sample_dir}")

    info = json.loads(info_file.read_text(encoding="utf-8"))
    selected_ingredients = info.get("ingredients", [])
    ground_truth = info.get("ground_truth", {})

    t0 = time.perf_counter()
    pred = predict(rgb_file, depth_file, selected_ingredients)
    t1 = time.perf_counter()

    print(f"[INFO] device: {_DEVICE}")
    print(f"[INFO] sample: {sample_dir}")
    print(f"[INFO] dish_id: {info.get('dish_id', '')}")
    print("[PRED]")
    print(json.dumps(pred, ensure_ascii=False, indent=2))
    print("[GROUND_TRUTH]")
    print(json.dumps(ground_truth, ensure_ascii=False, indent=2))
    print(f"[TIME] inference_total: {(t1 - t0) * 1000:.2f} ms")


def run_server():
    """服务模式：启动前加载一次模型和词表，再启动 Gradio。"""
    if gr is None:
        raise RuntimeError("未安装 gradio，请先执行: pip install gradio")
    load_vocab(DEFAULT_VOCAB_PATH)
    load_model(DEFAULT_CHECKPOINT_PATH, _DEVICE)
    demo = build_demo()
    # 允许从本地 examples 目录读取样例文件（修复点击样例后预测报错）
    allowed_paths = [str(Path("examples").resolve())]

    # 优先端口（可通过环境变量覆盖），若占用则自动向后尝试
    preferred_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    candidate_ports = [preferred_port] + list(range(preferred_port + 1, preferred_port + 20))
    last_error: Optional[Exception] = None

    for port in candidate_ports:
        try:
            print(f"[INFO] 尝试启动 Gradio: 0.0.0.0:{port}")
            demo.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=False,
                inbrowser=True,
                allowed_paths=allowed_paths,
            )
            return
        except OSError as e:
            # 端口冲突：继续尝试下一个端口
            msg = str(e)
            if "Cannot find empty port" in msg or "error while attempting to bind" in msg:
                print(f"[WARN] 端口 {port} 被占用，尝试下一个端口...")
                last_error = e
                continue
            raise
        except ValueError as e:
            # 某些环境会误判 localhost 不可访问；回退到本机环回地址重试
            if "localhost is not accessible" in str(e):
                print(f"[WARN] 0.0.0.0:{port} 不可访问，回退到 127.0.0.1:{port}")
                demo.launch(
                    server_name="127.0.0.1",
                    server_port=port,
                    share=False,
                    inbrowser=True,
                    allowed_paths=allowed_paths,
                )
                return
            raise

    if last_error is not None:
        raise last_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMIR-Net Gradio 演示系统")
    parser.add_argument(
        "--mode",
        type=str,
        default="server",
        choices=["server", "test"],
        help="server: 启动 Gradio 服务（默认）；test: 执行命令行推理测试。",
    )
    args = parser.parse_args()

    if args.mode == "test":
        run_cli_test()
    else:
        run_server()
