import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# 设置全局字体为 SimHei（黑体），确保中文正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布
fig, ax = plt.subplots(figsize=(11, 14), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')


def wrap_bullet_text(text: str, width: int = 30) -> str:
    """对每条 bullet 文本做自动换行，避免超出色块宽度。"""
    wrapped_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            wrapped_lines.append("")
            continue

        prefix = ""
        content = line
        if line.startswith("- "):
            prefix = "- "
            content = line[2:]

        chunks = textwrap.wrap(
            content,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
        )

        if not chunks:
            wrapped_lines.append(prefix)
            continue

        wrapped_lines.append(prefix + chunks[0])
        for chunk in chunks[1:]:
            wrapped_lines.append("  " + chunk)

    return "\n".join(wrapped_lines)


layers = [
    {
        "name": "用户交互层",
        "content": "\n".join(
            [
                "- Gradio Web 框架",
                "- 图像上传组件（RGB / Depth）",
                "- 成分多选下拉列表（255 种食材）",
                "- 预测按钮 / 样例数据加载",
                "- 左栏输入区 + 右栏输出区布局",
            ]
        ),
        "color_fill": "#E6F3FF",
        "color_edge": "#0066CC",
        "y": 76,
        "h": 13,
        "wrap_width": 30,
    },
    {
        "name": "数据预处理层",
        "content": "\n".join(
            [
                "- PIL Image 读取与 RGB 转换",
                "- Resize(416×416) → ToTensor → ImageNet 归一化",
                "- 深度图复用相同预处理管线",
                "- 成分名称 → 255 维二值向量编码",
                "- 与训练阶段数据管线保持一致",
            ]
        ),
        "color_fill": "#E6FFE6",
        "color_edge": "#009900",
        "y": 56,
        "h": 16,
        "wrap_width": 33,
    },
    {
        "name": "模型推理层",
        "content": "\n".join(
            [
                "- MyResNetRGBD（ingredients_dim=255）",
                "- 加载 .pth 检查点 → eval() 模式",
                "- CLIP 分支参数冻结",
                "- torch.no_grad() 上下文推理",
                "- 输入：RGB 张量 + Depth 张量 + 成分向量",
            ]
        ),
        "color_fill": "#FFF2E6",
        "color_edge": "#CC6600",
        "y": 36,
        "h": 16,
        "wrap_width": 33,
    },
    {
        "name": "结果展示层",
        "content": "\n".join(
            [
                "- Tensor → Python float 数值转换",
                "- 指标名称与单位格式化",
                "- Matplotlib 柱状图绘制",
                "- 预测值与真实值对比展示（若有标注）",
            ]
        ),
        "color_fill": "#F2E6FF",
        "color_edge": "#6600CC",
        "y": 16,
        "h": 13,
        "wrap_width": 30,
    },
]

# 矩形参数
box_w = 64
box_x = 15
center_x = box_x + box_w / 2
right_x = box_x + box_w

# 绘制层级矩形与文本
for layer in layers:
    layer_h = layer["h"]
    patch = FancyBboxPatch(
        (box_x, layer["y"]),
        box_w,
        layer_h,
        boxstyle="round,pad=0.5,rounding_size=1.5",
        ec=layer["color_edge"],
        fc=layer["color_fill"],
        lw=2,
    )
    ax.add_patch(patch)

    # 标题
    ax.text(
        center_x,
        layer["y"] + layer_h - 1.2,
        layer["name"],
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
        color="#222222",
    )

    # 分隔线
    sep_y = layer["y"] + layer_h - 3.0
    ax.plot(
        [box_x + 1.8, box_x + box_w - 1.8],
        [sep_y, sep_y],
        color=layer["color_edge"],
        lw=1.2,
        alpha=0.6,
    )

    # 内容（自动换行后绘制）
    wrapped = wrap_bullet_text(layer["content"], width=layer.get("wrap_width", 30))
    ax.text(
        box_x + 2.8,
        layer["y"] + layer_h - 4.1,
        wrapped,
        ha="left",
        va="top",
        fontsize=10,
        color="#333333",
        linespacing=1.28,
    )

# 箭头样式
arrow_args = dict(arrowstyle="-|>", color="#555555", lw=1.8, mutation_scale=14)

connections = [
    {"text": "RGB/Depth 图像\n+ 成分选择列表"},
    {"text": "RGB(1,3,416,416) + Depth(1,3,416,416)\n+ Vec(1,255)"},
    {"text": "五项预测值\n[calories, mass, fat, carb, protein]"},
]

# 绘制层间数据流
for i, conn in enumerate(connections):
    y_gap_top = layers[i]["y"]
    y_gap_bottom = layers[i + 1]["y"] + layers[i + 1]["h"]

    ax.annotate(
        "",
        xy=(center_x, y_gap_bottom + 0.5),
        xytext=(center_x, y_gap_top - 0.5),
        arrowprops=arrow_args,
    )

    ax.text(
        center_x + 2,
        (y_gap_top + y_gap_bottom) / 2,
        conn["text"],
        ha="left",
        va="center",
        fontsize=11,
        color="#222222",
        linespacing=1.15,
    )

# 右侧闭环反馈
y_start_return = layers[3]["y"] + layers[3]["h"] / 2
y_end_return = layers[0]["y"] + layers[0]["h"] / 2
x_turn = 92

ax.plot([right_x + 0.5, x_turn], [y_start_return, y_start_return], color="#555555", lw=2)
ax.plot([x_turn, x_turn], [y_start_return, y_end_return], color="#555555", lw=2)
ax.annotate(
    "",
    xy=(right_x + 0.5, y_end_return),
    xytext=(x_turn, y_end_return),
    arrowprops=arrow_args,
)

return_text = "格式化数值\n+\n可视化图表\n(返回展示结果)"
ax.text(
    x_turn + 1.5,
    (y_start_return + y_end_return) / 2,
    return_text,
    ha="left",
    va="center",
    fontsize=11,
    color="#222222",
    linespacing=1.2,
)

plt.savefig("system_architecture.png", bbox_inches="tight")


