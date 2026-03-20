import textwrap

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 优先选择常见中文字体，避免中文显示为方块
plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei',
    'SimHei',
    'Noto Sans CJK SC',
    'WenQuanYi Zen Hei',
    'Arial Unicode MS',
    'DejaVu Sans',
]
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 18), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(-14, 112)
ax.axis('off')


def wrap_lines(text: str, width: int = 26) -> str:
    """对多行文本自动换行，避免溢出节点边界。"""
    lines = []
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
            lines.append(prefix)
            continue

        lines.append(prefix + chunks[0])
        for chunk in chunks[1:]:
            lines.append("  " + chunk)

    return "\n".join(lines)


def draw_node(x, y, w, h, title, content, bg_color, border_color, content_width=26, content_size=10.8):
    """绘制处理节点（圆角矩形 + 标题分隔线）。"""
    radius = 2.0
    box = patches.FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        ec=border_color,
        fc=bg_color,
        lw=1.6,
    )
    ax.add_patch(box)

    # 标题
    ax.text(
        x,
        y + h / 2 - 1.35,
        title,
        ha='center',
        va='top',
        fontsize=13,
        fontweight='bold',
        color='#111111',
    )

    # 分隔线
    split_y = y + h / 2 - 3.6
    ax.plot([x - w / 2 + 1.8, x + w / 2 - 1.8], [split_y, split_y], color=border_color, lw=1.0, alpha=0.55)

    # 正文
    content_wrapped = wrap_lines(content, width=content_width)
    ax.text(
        x - w / 2 + 2.2,
        y + h / 2 - 4.8,
        content_wrapped,
        ha='left',
        va='top',
        fontsize=content_size,
        color='#222222',
        linespacing=1.3,
    )


def draw_shape_node(x, y, w, h, text):
    """绘制数据形态节点（虚线框）。"""
    box = patches.FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.05,rounding_size=1.4",
        ec='#808080',
        fc='#F7F7F7',
        ls='--',
        lw=1.3,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=10.2, color='#333333')


def draw_arrow(x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle='-|>', color='#333333', lw=1.5, mutation_scale=14),
    )


def draw_text_label(x, y, text):
    bbox_props = dict(boxstyle='round,pad=0.22', fc='white', ec='none', alpha=0.88)
    ax.text(x, y, text, ha='center', va='center', fontsize=10.2, color='#555555', bbox=bbox_props)


# 节点布局
s1_x, s1_y, s1_w, s1_h = 35, 102, 40, 13
sh1_x, sh1_y, sh1_w, sh1_h = 79, 102, 30, 4.4

s2_x, s2_y, s2_w, s2_h = 35, 88, 40, 9.5
sh2_x, sh2_y, sh2_w, sh2_h = 79, 88, 34, 4.4

s3l_x, s3l_y, s3l_w, s3l_h = 22, 67, 32, 16
s3r_x, s3r_y, s3r_w, s3r_h = 76, 67, 32, 16
merge_sh_x, merge_sh_y, merge_sh_w, merge_sh_h = 49, 49, 82, 4.6

s4_x, s4_y, s4_w, s4_h = 49, 33, 46, 18
sh4_x, sh4_y, sh4_w, sh4_h = 49, 20, 56, 4.6

s5_x, s5_y, s5_w, s5_h = 49, 7.5, 46, 13
sh5_x, sh5_y, sh5_w, sh5_h = 49, -3.2, 48, 4.6

end_x, end_y, end_w, end_h = 49, -11.5, 42, 4.8


# 1) 绘制节点
draw_node(
    s1_x,
    s1_y,
    s1_w,
    s1_h,
    "用户输入",
    "\n".join(
        [
            "- 上传 RGB 食物图像（.jpeg/.png）",
            "- 上传伪彩色深度图像（depth_color.png）",
            "- 从 255 种成分列表中勾选食材",
        ]
    ),
    "#E6F3FF",
    "#0066CC",
    content_width=24,
    content_size=10.2,
)
draw_shape_node(sh1_x, sh1_y, sh1_w, sh1_h, "PIL Image × 2 + List[str]")

draw_node(
    s2_x,
    s2_y,
    s2_w,
    s2_h,
    "输入校验",
    "\n".join(
        [
            "- 图像格式有效性检查",
            "- 成分勾选非空检查",
        ]
    ),
    "#E6FFE6",
    "#009900",
    content_width=24,
    content_size=10.4,
)
draw_shape_node(sh2_x, sh2_y, sh2_w, sh2_h, "已校验的 PIL Image × 2 + List[str]")

draw_node(
    s3l_x,
    s3l_y,
    s3l_w,
    s3l_h,
    "图像预处理",
    "\n".join(
        [
            "- Resize(416×416)",
            "- ToTensor() → [0,1]",
            "- ImageNet Normalize",
            "- unsqueeze(0) 扩展批次维度",
            "- 移至 GPU/CPU",
        ]
    ),
    "#FFF2E6",
    "#CC6600",
    content_width=18,
    content_size=9.7,
)

draw_node(
    s3r_x,
    s3r_y,
    s3r_w,
    s3r_h,
    "成分编码",
    "\n".join(
        [
            "- 成分名称映射到词表索引",
            "- 生成 255 维二值向量",
            "- 转为 float32 张量",
            "- 移至 GPU/CPU",
        ]
    ),
    "#FFF2E6",
    "#CC6600",
    content_width=18,
    content_size=9.7,
)

draw_shape_node(
    merge_sh_x,
    merge_sh_y,
    merge_sh_w,
    merge_sh_h,
    "RGB 张量(1,3,416,416) + Depth 张量(1,3,416,416) + 成分向量(1,255)",
)

draw_node(
    s4_x,
    s4_y,
    s4_w,
    s4_h,
    "模型推理",
    "\n".join(
        [
            "- torch.no_grad() 上下文",
            "- model(rgb, depth, ingredients_vec)",
            "- ResNet-101 + CLIP 特征融合",
            "- MBFM 多分支 RGB-D 融合",
            "- 成分引导通道注意力",
            "- 输出五个独立预测头",
        ]
    ),
    "#F2E6FF",
    "#6600CC",
    content_width=28,
    content_size=9.8,
)

draw_shape_node(sh4_x, sh4_y, sh4_w, sh4_h, "五项标量预测值 [calories, mass, fat, carb, protein]")

draw_node(
    s5_x,
    s5_y,
    s5_w,
    s5_h,
    "结果后处理与展示",
    "\n".join(
        [
            "- Tensor → Python float 转换",
            "- 指标名称与单位格式化",
            "- Matplotlib 柱状图生成",
            "- 返回 Gradio 前端渲染",
        ]
    ),
    "#FFFFE6",
    "#CCCC00",
    content_width=27,
    content_size=10.0,
)

draw_shape_node(sh5_x, sh5_y, sh5_w, sh5_h, "格式化数值表格 + Matplotlib Figure")

end_patch = patches.FancyBboxPatch(
    (end_x - end_w / 2, end_y - end_h / 2),
    end_w,
    end_h,
    boxstyle="round,pad=0.2,rounding_size=1.6",
    ec='#666666',
    fc='#DADADA',
    lw=1.5,
)
ax.add_patch(end_patch)
ax.text(end_x, end_y, "用户在浏览器中查看预测结果", ha='center', va='center', fontsize=12, fontweight='bold', color='#222222')


# 2) 绘制数据流连线
ax.plot([s1_x + s1_w / 2, sh1_x - sh1_w / 2], [s1_y, sh1_y], color='#888888', ls=':', lw=1.4)
ax.plot([s2_x + s2_w / 2, sh2_x - sh2_w / 2], [s2_y, sh2_y], color='#888888', ls=':', lw=1.4)

draw_arrow(s1_x, s1_y - s1_h / 2, s2_x, s2_y + s2_h / 2)
draw_text_label(s1_x, (s1_y - s1_h / 2 + s2_y + s2_h / 2) / 2, "原始图像与成分列表")

split_y = s2_y - s2_h / 2 - 3.5
ax.plot([s2_x, s2_x], [s2_y - s2_h / 2, split_y], color='#333333', lw=1.5)
ax.plot([s3l_x, s3r_x], [split_y, split_y], color='#333333', lw=1.5)
draw_arrow(s3l_x, split_y, s3l_x, s3l_y + s3l_h / 2)
draw_arrow(s3r_x, split_y, s3r_x, s3r_y + s3r_h / 2)
draw_text_label(s2_x, split_y + 1.4, "校验通过的有效输入")

merge_y = s3l_y - s3l_h / 2 - 3.2
ax.plot([s3l_x, s3l_x], [s3l_y - s3l_h / 2, merge_y], color='#333333', lw=1.5)
ax.plot([s3r_x, s3r_x], [s3r_y - s3r_h / 2, merge_y], color='#333333', lw=1.5)
ax.plot([s3l_x, s3r_x], [merge_y, merge_y], color='#333333', lw=1.5)
draw_arrow(merge_sh_x, merge_y, merge_sh_x, merge_sh_y + merge_sh_h / 2)

draw_arrow(merge_sh_x, merge_sh_y - merge_sh_h / 2, s4_x, s4_y + s4_h / 2)
draw_text_label(merge_sh_x, (merge_sh_y - merge_sh_h / 2 + s4_y + s4_h / 2) / 2, "三路标准化张量")

draw_arrow(s4_x, s4_y - s4_h / 2, sh4_x, sh4_y + sh4_h / 2)
draw_arrow(sh4_x, sh4_y - sh4_h / 2, s5_x, s5_y + s5_h / 2)
draw_text_label(s4_x + 13, (s4_y - s4_h / 2 + sh4_y + sh4_h / 2) / 2, "五项营养预测值")

draw_arrow(s5_x, s5_y - s5_h / 2, sh5_x, sh5_y + sh5_h / 2)
draw_arrow(sh5_x, sh5_y - sh5_h / 2, end_x, end_y + end_h / 2)
draw_text_label(s5_x + 14, (s5_y - s5_h / 2 + sh5_y + sh5_h / 2) / 2, "格式化结果与可视化图表")

# 输出到仓库根目录
plt.savefig('data_flow_diagram.png', bbox_inches='tight')
plt.close()
