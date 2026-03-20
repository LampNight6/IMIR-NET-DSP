import textwrap

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 10), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')


def wrap_bullet_text(text: str, width: int) -> str:
    """将模块正文按行自动换行，避免中文溢出色块。"""
    wrapped_lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
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


def draw_module(
    ax,
    x,
    y,
    w,
    h,
    header_h,
    title,
    content,
    bg_color,
    head_color,
    border_color,
    content_width,
    content_fontsize,
):
    """绘制功能模块，含标题黑色空心框、正文换行与外边框。"""
    radius = 1.8

    # 主体与标题背景
    main_box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        ec="none",
        fc=bg_color,
    )
    ax.add_patch(main_box)

    head_box = patches.FancyBboxPatch(
        (x, y + h - header_h),
        w,
        header_h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        ec="none",
        fc=head_color,
    )
    ax.add_patch(head_box)

    # 修补标题区域底部圆角
    head_fill = patches.Rectangle((x, y + h - header_h), w, radius, fc=head_color, ec="none")
    ax.add_patch(head_fill)

    # 标题与正文分隔线
    split_y = y + h - header_h
    ax.plot([x, x + w], [split_y, split_y], color=border_color, lw=1)

    # 模块外边框
    border_box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        ec=border_color,
        fc="none",
        lw=1.6,
    )
    ax.add_patch(border_box)

    # 标题黑色空心框（按你的要求保留并校正）
    title_box_w = w - 8
    title_box_h = min(3.2, header_h - 0.7)
    title_box_x = x + (w - title_box_w) / 2
    title_box_y = y + h - header_h + (header_h - title_box_h) / 2
    title_box = patches.FancyBboxPatch(
        (title_box_x, title_box_y),
        title_box_w,
        title_box_h,
        boxstyle="round,pad=0.12,rounding_size=0.8",
        ec="#111111",
        fc="none",
        lw=1.1,
    )
    ax.add_patch(title_box)

    ax.text(
        x + w / 2,
        y + h - header_h / 2,
        title,
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color="#111111",
    )

    wrapped_content = wrap_bullet_text(content, width=content_width)
    ax.text(
        x + 2.2,
        y + h - header_h - 1.7,
        wrapped_content,
        ha="left",
        va="top",
        fontsize=content_fontsize,
        color="#333333",
        linespacing=1.25,
    )


# 全局尺寸
w = 35
header_h = 5.2
h_top = 21
h_mid = 24
h_bottom = 21

# 坐标（拉开模型推理与结果展示模块）
x1, y1 = 7, 72    # 图像输入
x2, y1 = 58, 72   # 成分输入
x3, y2 = 31, 42   # 数据预处理
x4, y3 = 7, 10    # 模型推理
x5, y3 = 58, 10   # 结果展示

# 1 图像输入
draw_module(
    ax,
    x1,
    y1,
    w,
    h_top,
    header_h,
    "图像输入模块",
    "\n".join(
        [
            "- RGB 图像上传与预览",
            "- 深度图像上传与预览",
            "- 图像格式校验",
        ]
    ),
    "#E6F3FF",
    "#CCE6FF",
    "#0066CC",
    content_width=24,
    content_fontsize=10,
)

# 2 成分输入
draw_module(
    ax,
    x2,
    y1,
    w,
    h_top,
    header_h,
    "成分输入模块",
    "\n".join(
        [
            "- 255 种食材成分多选列表",
            "- 成分词表加载（ingredients_vocab.json）",
            "- 成分名称映射为 255 维向量",
        ]
    ),
    "#E6FFE6",
    "#CCFFCC",
    "#009900",
    content_width=24,
    content_fontsize=9.7,
)

# 3 数据预处理（单独增高 + 更强换行）
draw_module(
    ax,
    x3,
    y2,
    w,
    h_mid,
    header_h,
    "数据预处理模块",
    "\n".join(
        [
            "- Resize(416×416) → ToTensor → ImageNet 归一化",
            "- RGB / Depth 图像统一预处理管线",
            "- 成分向量转为 float32 张量",
            "- 与训练阶段数据管线保持一致",
        ]
    ),
    "#FFF2E6",
    "#FFE6CC",
    "#CC6600",
    content_width=20,
    content_fontsize=9.5,
)

# 4 模型推理
draw_module(
    ax,
    x4,
    y3,
    w,
    h_bottom,
    header_h,
    "模型推理模块",
    "\n".join(
        [
            "- MyResNetRGBD 模型实例化与权重加载",
            "- eval() 模式 + CLIP 分支冻结",
            "- torch.no_grad() 前向推理",
            "- 输出：热量 / 质量 / 脂肪 / 碳水 / 蛋白质",
        ]
    ),
    "#F2E6FF",
    "#E6CCFF",
    "#6600CC",
    content_width=21,
    content_fontsize=9.2,
)

# 5 结果展示
draw_module(
    ax,
    x5,
    y3,
    w,
    h_bottom,
    header_h,
    "结果展示模块",
    "\n".join(
        [
            "- Tensor → float 数值转换与单位格式化",
            "- 数值结果表格（含真值对比）",
            "- Matplotlib 柱状图可视化",
            "- 预测值与真实值对比展示",
        ]
    ),
    "#FFFFE6",
    "#FFFFCC",
    "#CCCC00",
    content_width=21,
    content_fontsize=9.2,
)

# 箭头样式
arrow_style = dict(arrowstyle="-|>", color="#333333", lw=1.5, mutation_scale=14)

# 图像输入 -> 数据预处理
ax.annotate(
    "",
    xy=(x3 + 8, y2 + h_mid + 0.6),
    xytext=(x1 + w / 2, y1 - 0.6),
    arrowprops=arrow_style,
)
ax.text(
    (x1 + w / 2 + x3 + 8) / 2 - 2,
    (y1 + y2 + h_mid) / 2 + 0.2,
    "RGB/Depth 图像",
    ha="right",
    va="center",
    color="#555555",
    fontsize=10.5,
)

# 成分输入 -> 数据预处理
ax.annotate(
    "",
    xy=(x3 + w - 8, y2 + h_mid + 0.6),
    xytext=(x2 + w / 2, y1 - 0.6),
    arrowprops=arrow_style,
)
ax.text(
    (x2 + w / 2 + x3 + w - 8) / 2 + 2,
    (y1 + y2 + h_mid) / 2 + 0.2,
    "255 维成分向量",
    ha="left",
    va="center",
    color="#555555",
    fontsize=10.5,
)

# 数据预处理 -> 模型推理
ax.annotate(
    "",
    xy=(x4 + w / 2, y3 + h_bottom + 0.6),
    xytext=(x3 + 8, y2 - 0.6),
    arrowprops=arrow_style,
)
ax.text(
    (x3 + 8 + x4 + w / 2) / 2 + 2,
    (y2 + y3 + h_bottom) / 2,
    "标准化张量",
    ha="left",
    va="center",
    color="#555555",
    fontsize=10.5,
)

# 模型推理 -> 结果展示（模块拉开后，标签放在模块上方避免覆盖色块）
ax.annotate(
    "",
    xy=(x5 - 0.5, y3 + h_bottom / 2),
    xytext=(x4 + w + 0.5, y3 + h_bottom / 2),
    arrowprops=arrow_style,
)
ax.text(
    (x4 + w + x5) / 2,
    y3 + h_bottom / 2 + 1.8,
    "五项营养预测值",
    ha="center",
    va="bottom",
    color="#555555",
    fontsize=10.5,
)

# 导出图片（按要求移除底部“图 4-X 功能模块图”）
plt.savefig("function_module_diagram.png", bbox_inches="tight")
plt.close()
