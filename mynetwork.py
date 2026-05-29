import torch
import torch.nn as nn
import clip
from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1, conv3x3
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse


"""
IMIR-Net 的核心网络定义。

本文件包含 RGB-D 双分支 ResNet、CLIP 视觉特征补充、频域融合模块 MBFM、
膨胀卷积编码器以及营养指标回归头。整体 forward 路径是：
RGB/Depth 图像 -> 双分支特征提取 -> CLIP/RGB 与 RGB-D 融合 -> 多尺度特征聚合
-> 食材向量门控 -> 输出 calories/mass/fat/carb/protein 五个回归结果。
"""


class CLIPVisionBranch(nn.Module):
    """冻结的 CLIP RN 视觉分支，用来提供额外的语义视觉特征。

    主模型训练时不会更新 CLIP 参数，只通过 forward hook 取出 layer3/layer4
    中间特征。输入默认已按 ImageNet 均值方差归一化，因此进入 CLIP 前先反
    归一化到 [0, 1]，再按 CLIP 自己的均值方差归一化。
    """

    def __init__(self, clip_model_name: str = "RN101", assume_imagenet_normalized_input: bool = True):
        super().__init__()
        clip_model, _ = clip.load(clip_model_name, jit=False)
        self.clip_model = clip_model
        self.clip_model.to("cpu")
        self.clip_model.eval()
        self.clip_model.float()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.register_buffer(
            "clip_mean",
            torch.tensor([0.481, 0.457, 0.408], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "clip_std",
            torch.tensor([0.268, 0.261, 0.275], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.assume_imagenet_normalized_input = bool(assume_imagenet_normalized_input)

        self._features = {}
        self._hook_handles = []

        def _save(name: str):
            def _hook(_module, _inputs, output):
                # forward hook 保存指定 stage 的特征，避免改动 CLIP 原始 forward。
                self._features[name] = output

            return _hook

        visual = self.clip_model.visual
        self._hook_handles.append(visual.layer3.register_forward_hook(_save("s3")))
        self._hook_handles.append(visual.layer4.register_forward_hook(_save("s4")))

    def train(self, mode: bool = True):
        super().train(mode)
        # 即使外部调用 model.train()，CLIP 仍保持 eval，避免 BN/统计量变化。
        self.clip_model.eval()
        return self

    def _forward_clip_visual_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """只执行 CLIP ResNet 的卷积骨干，跳过最终 attention pool/head。"""
        visual = self.clip_model.visual
        x = visual.conv1(x)
        x = visual.bn1(x)
        x = visual.relu1(x)
        x = visual.conv2(x)
        x = visual.bn2(x)
        x = visual.relu2(x)
        x = visual.conv3(x)
        x = visual.bn3(x)
        x = visual.relu3(x)
        x = visual.avgpool(x)
        x = visual.layer1(x)
        x = visual.layer2(x)
        x = visual.layer3(x)
        x = visual.layer4(x)
        return x

    def forward(self, x_rgb: torch.Tensor):
        x_rgb = x_rgb.float()
        if self.assume_imagenet_normalized_input:
            # 数据集 transform 使用 ImageNet normalize，这里恢复到图像值域。
            x_rgb = x_rgb * self.imagenet_std + self.imagenet_mean
        x_rgb = x_rgb.clamp(0.0, 1.0)
        # CLIP 使用不同的均值方差；这里转换到 CLIP 预训练分布。
        x_rgb = (x_rgb - self.clip_mean) / self.clip_std

        self._features.clear()
        with torch.no_grad():
            _ = self._forward_clip_visual_backbone(x_rgb)

        clip_s3 = self._features.get("s3")
        clip_s4 = self._features.get("s4")
        if clip_s3 is None or clip_s4 is None:
            raise RuntimeError("CLIPVisionBranch hooks did not capture layer3/layer4 features.")
        return clip_s3.float(), clip_s4.float()

class CA_Enhance(nn.Module):
    """通道注意力增强模块。

    将 RGB 与 depth 特征在通道维拼接，通过全局池化生成 depth 通道权重，
    用于旧版模型中的跨模态 depth 增强。
    """

    def __init__(self, in_planes, ratio=16):
        super(CA_Enhance, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes // 2, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb, depth):
        x = torch.cat((rgb, depth), dim=1)
        #print(x.shape)
        max_pool_x = self.max_pool(x)
        #print(self.relu1(self.fc1(max_pool_x)).shape)

        max_out = self.fc2(self.relu1(self.fc1(max_pool_x)))
        out = max_out
        depth = depth.mul(self.sigmoid(out))
        return depth

class SA_Enhance(nn.Module):
    """空间注意力增强模块，根据通道最大响应生成单通道空间权重图。"""

    def __init__(self, kernel_size=7):
        super(SA_Enhance, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class CA_SA_Enhance(nn.Module):
    """旧版 CA + SA 组合模块，先做通道增强，再做空间增强。"""

    def __init__(self, in_planes, ratio=16):
        super(CA_SA_Enhance, self).__init__()

        self.self_CA_Enhance = CA_Enhance(in_planes)
        self.self_SA_Enhance = SA_Enhance()

    def forward(self, rgb, depth):
        x_d = self.self_CA_Enhance(rgb, depth)
        sa = self.self_SA_Enhance(x_d)
        depth_enhance = depth.mul(sa)
        return depth_enhance


class MBFM(nn.Module):
    """
    Multifrequency Bimodality Fusion Module (MBFM).

    Uses DWT to split features into low/high frequency; discards depth high-frequency
    components to suppress depth noise while keeping RGB texture details.

    输入的 RGB/depth 特征通道数相同，先各自降到一半通道，再做 Haar 小波
    分解。低频部分融合 RGB 与 depth 的结构信息，高频部分仅保留 RGB，
    以减少深度图噪声对纹理细节的影响。最后逆小波重建并与 depth 降维特征拼接。
    """

    def __init__(self, in_channels: int):
        super().__init__()
        if in_channels % 2 != 0:
            raise ValueError(f"MBFM expects even in_channels, got {in_channels}.")

        hidden_channels = in_channels // 2

        def cbr(in_ch: int, out_ch: int, k: int, padding: int):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=padding, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.rgb_reduce = cbr(in_channels, hidden_channels, k=1, padding=0)
        self.depth_reduce = cbr(in_channels, hidden_channels, k=1, padding=0)

        self.dwt = DWTForward(J=1, mode="zero", wave="haar")
        self.idwt = DWTInverse(mode="zero", wave="haar")

        self.rgb_ll_conv = cbr(hidden_channels, hidden_channels, k=3, padding=1)
        self.depth_ll_conv = cbr(hidden_channels, hidden_channels, k=3, padding=1)

        self.fuse_out = cbr(in_channels, in_channels, k=3, padding=1)

    def forward(self, x_rgb: torch.Tensor, x_depth: torch.Tensor) -> torch.Tensor:
        # 先用 1x1 卷积压缩通道，控制小波域融合的计算量。
        rgb_r = self.rgb_reduce(x_rgb)
        depth_r = self.depth_reduce(x_depth)

        # DWT 返回低频 LL 和高频 LH/HL/HH；这里只保留 RGB 高频。
        rgb_ll, rgb_high = self.dwt(rgb_r)  # rgb_high: list length J
        depth_ll, _depth_high = self.dwt(depth_r)

        fused_ll = self.rgb_ll_conv(rgb_ll) + self.depth_ll_conv(depth_ll)

        out_dwt = self.idwt((fused_ll, rgb_high))
        if out_dwt.shape[-2:] != depth_r.shape[-2:]:
            # 奇数尺寸经过 DWT/IDWT 可能产生一像素差异，回到 depth 特征尺寸。
            out_dwt = F.interpolate(out_dwt, size=depth_r.shape[-2:], mode="bilinear", align_corners=False)

        fused = torch.cat([out_dwt, depth_r], dim=1)
        return self.fuse_out(fused)

class DilatedEncoder(nn.Module):
    """
    Dilated Encoder for YOLOF.
    This module contains two types of components:
        - the original FPN lateral convolution layer and fpn convolution layer,
          which are 1x1 conv + 3x3 conv
        - the dilated residual block
    """

    def __init__(self,
                 in_channels=2048,
                 encoder_channels=512,
                 block_mid_channels=128,
                 #num_residual_blocks=4,
                 num_residual_blocks=3,
                 #block_dilations=[2, 4, 6, 8]
                 block_dilations = [2, 4, 6]
                 ):
        super(DilatedEncoder, self).__init__()
        # fmt: off
        self.in_channels = in_channels
        self.encoder_channels = encoder_channels
        self.block_mid_channels = block_mid_channels
        self.num_residual_blocks = num_residual_blocks
        self.block_dilations = block_dilations

        assert len(self.block_dilations) == self.num_residual_blocks

        # init
        self._init_layers()
        self._init_weight()

    def _init_layers(self):
        self.lateral_conv = nn.Conv2d(self.in_channels,
                                      self.encoder_channels,
                                      kernel_size=1)
        self.lateral_norm = nn.BatchNorm2d(self.encoder_channels)
        self.fpn_conv = nn.Conv2d(self.encoder_channels,
                                  self.encoder_channels,
                                  kernel_size=3,
                                  padding=1)
        self.fpn_norm = nn.BatchNorm2d(self.encoder_channels)
        encoder_blocks = []
        for i in range(self.num_residual_blocks):
            dilation = self.block_dilations[i]
            # 不同 dilation 的残差块扩大感受野，保留特征图分辨率。
            encoder_blocks.append(
                MyBottleneck(
                    self.encoder_channels,
                    self.block_mid_channels,
                    dilation=dilation
                )
            )
        self.dilated_encoder_blocks = nn.Sequential(*encoder_blocks)

    def xavier_init(self, layer):
        if isinstance(layer, nn.Conv2d):
            # print(layer.weight.data.type())
            # m.weight.data.fill_(1.0)
            nn.init.xavier_uniform_(layer.weight, gain=1)

    def _init_weight(self):
        self.xavier_init(self.lateral_conv)
        self.xavier_init(self.fpn_conv)
        for m in [self.lateral_norm, self.fpn_norm]:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        for m in self.dilated_encoder_blocks.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        # 先把 backbone 通道投影到 encoder_channels，再串联膨胀残差块。
        out = self.lateral_norm(self.lateral_conv(feature))
        out = self.fpn_norm(self.fpn_conv(out))
        return self.dilated_encoder_blocks(out)


class MyBottleneck(nn.Module):
    """DilatedEncoder 内部使用的轻量瓶颈残差块。"""

    def __init__(self,
                 in_channels: int = 512,
                 mid_channels: int = 128,
                 dilation: int = 1):
        super(MyBottleneck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels,
                      kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        # 残差连接稳定深层膨胀卷积训练。
        out = out + identity
        return out

class MyResNet(nn.Module):
    """去掉分类头的 ResNet 骨干，返回 layer3/layer4 两级特征。"""

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(MyResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 标准 ResNet 四个 stage；本项目只使用后两级特征做营养回归。
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)  这里将全连接层注释掉了
        #self.fc1 = nn.Linear(512, 256) #只用l4层的特征是512
        #self.fc1 = nn.Linear(512, 256) #
        # self.calorie = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))
        # self.mass = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))
        # self.fat = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))
        # self.carb = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))
        # self.protein = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))
        # self.encoder_l3 = DilatedEncoder(in_channels=1024)
        # self.encoder_l4 = DilatedEncoder(in_channels=2048)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        #print('11test')
        #print(block.expansion)
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            # 用 dilation 替代 stride 时保持输出分辨率不再下采样。
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        # stem + layer1/layer2 是共享的低中层纹理/形状特征提取。
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        l3_fea = self.layer3(x)
        #l3_fea = self.encoder_l3(x)
        l4_fea = self.layer4(l3_fea)
        #x = self.encoder_l4(x)
        #print(x.shape)
        #x = self.avgpool(x)
        #print(x.shape)
        #l3_fea_pool = self.avgpool(l3_fea)
        #print(l3_fea_pool.shape)
        #x = l3_fea_pool + x

        #x = torch.flatten(x, 1)


        # x = self.fc1(x)
        # embedding = F.relu(x)
        # # embedding = F.dropout(embedding, self.training)
        # results = []
        # results.append(self.calorie(embedding).squeeze())
        # results.append(self.mass(embedding).squeeze())
        # results.append(self.fat(embedding).squeeze())
        # results.append(self.carb(embedding).squeeze())
        # results.append(self.protein(embedding).squeeze())
        # return results
        # x = self.fc(x) 这里注释掉了最后一个全连接层，直接输出提取的特征

        return l3_fea, l4_fea

    def forward(self, x):
        return self._forward_impl(x)

class MyResNetRGBD(nn.Module):
    """当前 RGB-D + 食材信息融合模型。

    主要组成：
    1. RGB 与 depth 各一套 Food2K 预训练 ResNet101 backbone。
    2. 可选 CLIP 视觉分支补充 RGB 语义特征。
    3. MBFM 在 layer3/layer4 上融合 RGB 与 depth。
    4. DilatedEncoder 聚合多尺度上下文。
    5. 食材向量生成 visual gate，并与视觉向量拼接后回归五个营养指标。
    """

    def __init__(self, ingredients_dim: int = 255):
        super(MyResNetRGBD, self).__init__()
        self.model_rgb = MyResNet(Bottleneck, [3, 4, 23, 3])  # 这里具体的参数参考库中源代码
        self.model_rgb.load_state_dict(torch.load('food2k_resnet101_0.0001.pth'), strict=False)

        self.model_depth = MyResNet(Bottleneck, [3, 4, 23, 3])  # 这里具体的参数参考库中源代码
        self.model_depth.load_state_dict(torch.load('food2k_resnet101_0.0001.pth'), strict=False)

        # CLIP 分支本身冻结，只训练后面的 1x1 融合层。
        self.clip_branch = CLIPVisionBranch()
        self.clip_fusion_s3 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.clip_fusion_s4 = nn.Sequential(
            nn.Conv2d(in_channels=4096, out_channels=2048, kernel_size=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )
        self.enable_clip_fusion = True

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.ingredients_dim = int(ingredients_dim)
        # 食材输入可以是 512 维 CLIP 文本特征，也可以是 255 维二值食材向量。
        self.ingredients_fc1 = nn.Linear(self.ingredients_dim, 256)
        self.ingredients_fc2 = nn.Linear(256, 512)  # attention weights for 512-D visual vector (after GAP)
        self.fusion_fc = nn.Linear(512 + 256, 256)

        # 五个独立回归头，对应 Nutrition5k 的总热量、质量、脂肪、碳水、蛋白质。
        self.calorie = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))
        self.mass = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))
        self.fat = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))
        self.carb = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))
        self.protein = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))

        self.encoder_l3_rgb = DilatedEncoder(in_channels=1024)
        self.encoder_l4_rgb = DilatedEncoder(in_channels=2048)

        self.encoder_l3_depth = DilatedEncoder(in_channels=1024)
        self.encoder_l4_depth = DilatedEncoder(in_channels=2048)

        self.mbfm_l3 = MBFM(in_channels=1024)
        self.mbfm_l4 = MBFM(in_channels=2048)

        self.con2d = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.con2d_t = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1, bias=True)
        self.fc_t = nn.Linear(512,512)

    def load_state_dict(self, state_dict, strict: bool = True):
        """兼容旧 checkpoint。

        旧权重中没有 CLIP 分支和融合层时，不把这些缺失 key 视为错误，
        同时关闭 enable_clip_fusion，保证旧 checkpoint 仍能推理。
        """
        if not strict:
            incompatible = super().load_state_dict(state_dict, strict=False)
            if not any(k.startswith("clip_fusion_s3.") or k.startswith("clip_fusion_s4.") for k in state_dict.keys()):
                self.enable_clip_fusion = False
            return incompatible

        incompatible = super().load_state_dict(state_dict, strict=False)
        if not any(k.startswith("clip_fusion_s3.") or k.startswith("clip_fusion_s4.") for k in state_dict.keys()):
            self.enable_clip_fusion = False

        allowed_missing_prefixes = (
            "clip_branch.",
            "clip_fusion_s3.",
            "clip_fusion_s4.",
        )

        missing_keys = [
            k for k in incompatible.missing_keys
            if not any(k.startswith(p) for p in allowed_missing_prefixes)
        ]
        unexpected_keys = incompatible.unexpected_keys

        if missing_keys or unexpected_keys:
            error_msgs = []
            if missing_keys:
                error_msgs.append('Missing key(s) in state_dict: {}.'.format(
                    ', '.join('"{}"'.format(k) for k in missing_keys)
                ))
            if unexpected_keys:
                error_msgs.append('Unexpected key(s) in state_dict: {}.'.format(
                    ', '.join('"{}"'.format(k) for k in unexpected_keys)
                ))
            raise RuntimeError("Error(s) in loading state_dict for {}:\n\t{}".format(
                self.__class__.__name__, "\n\t".join(error_msgs)
            ))

        return incompatible

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        for k in list(state.keys()):
            # CLIP 预训练参数不随 checkpoint 保存，减少文件体积并避免重复存储冻结权重。
            if k.startswith("clip_branch.clip_model."):
                state.pop(k)
        return state

    def forward(self, x_rgb, x_depth, ingredients_vec):
        # RGB backbone 显式展开到 layer3/layer4，便于插入 CLIP 与 RGB-D 融合模块。
        x_fea = self.model_rgb.conv1(x_rgb)
        x_fea = self.model_rgb.bn1(x_fea)
        x_fea = self.model_rgb.relu(x_fea)
        x_fea = self.model_rgb.maxpool(x_fea)

        x_fea = self.model_rgb.layer1(x_fea)
        x_fea = self.model_rgb.layer2(x_fea)
        rgb_s3 = self.model_rgb.layer3(x_fea)
        rgb_s4 = self.model_rgb.layer4(rgb_s3)

        if self.enable_clip_fusion:
            # CLIP 与 ResNet 特征空间尺寸可能不同，先插值对齐后再按通道拼接。
            clip_s3, clip_s4 = self.clip_branch(x_rgb)
            if clip_s3.shape[-2:] != rgb_s3.shape[-2:]:
                clip_s3 = F.interpolate(clip_s3, size=rgb_s3.shape[-2:], mode='bilinear', align_corners=False)
            if clip_s4.shape[-2:] != rgb_s4.shape[-2:]:
                clip_s4 = F.interpolate(clip_s4, size=rgb_s4.shape[-2:], mode='bilinear', align_corners=False)
            l3_fea_rgb = self.clip_fusion_s3(torch.cat([rgb_s3, clip_s3], dim=1))
            l4_fea_rgb = self.clip_fusion_s4(torch.cat([rgb_s4, clip_s4], dim=1))
        else:
            l3_fea_rgb = rgb_s3
            l4_fea_rgb = rgb_s4

        ######################################
        # depth backbone 与 RGB backbone 结构相同，输入是 depth_color 图像。
        x_fea_depth = self.model_depth.conv1(x_depth)
        x_fea_depth = self.model_depth.bn1(x_fea_depth)
        x_fea_depth = self.model_depth.relu(x_fea_depth)
        x_fea_depth = self.model_depth.maxpool(x_fea_depth)

        x_fea_depth = self.model_depth.layer1(x_fea_depth)
        x_fea_depth = self.model_depth.layer2(x_fea_depth)
        l3_fea_depth = self.model_depth.layer3(x_fea_depth)

        # Layer 3 Fusion (RGB <- RGBD)
        # MBFM 输出仍保持 RGB stage 的通道数，后续继续作为主视觉特征。
        l3_fea_rgb = self.mbfm_l3(l3_fea_rgb, l3_fea_depth)

        l4_fea_depth = self.model_depth.layer4(l3_fea_depth)

        # Layer 4 Fusion (RGB <- RGBD)
        l4_fea_rgb = self.mbfm_l4(l4_fea_rgb, l4_fea_depth)


        l3_fea_rgb_dilate = self.encoder_l3_rgb(l3_fea_rgb)
        l4_fea_rgb_dilate = self.encoder_l4_rgb(l4_fea_rgb)

        # l3_fea_rgb_pool = self.avgpool(l3_fea_rgb_dilate)
        # l4_fea_rgb_pool = self.avgpool(l4_fea_rgb_dilate)
        # layer4 分辨率低于 layer3，上采样后和 layer3 特征拼接形成 1024 通道特征。
        l4_fea_rgb_dilate_up = torch.nn.functional.interpolate(l4_fea_rgb_dilate, scale_factor=2, mode='bilinear', align_corners=False)
        #rgb_fea = l3_fea_rgb_pool + l4_fea_rgb_pool_up
        rgb_fea_cat = torch.cat((l3_fea_rgb_dilate, l4_fea_rgb_dilate_up), dim=1)
        #print(rgb_fea.shape)
        # con2d 将多尺度拼接特征压到 512 通道，再全局池化得到视觉向量。
        rgb_fea = self.con2d(rgb_fea_cat)
        rgb_fea = self.relu(rgb_fea)
        rgb_fea = self.avgpool(rgb_fea)

        rgb_fea_t = self.con2d_t(rgb_fea_cat)
        rgb_fea_t = self.relu(rgb_fea_t)
        rgb_fea_t = self.avgpool(rgb_fea_t)
        rgb_fea_t = torch.flatten(rgb_fea_t, 1)
        rgb_fea_t = self.fc_t(rgb_fea_t)
        #rgb_fea_t /= rgb_fea_t.norm(dim=1, keepdim=True)
        #l3_fea = self.encoder_l3(x)
        #l4_fea = self.layer4(l3_fea)

        #l3_fea_rgb, l4_fea_rgb = self.model_rgb(x_rgb)
        #l3_fea_depth, l4_fea_depth = self.model_depth(x_depth)
        #print(l3_fea_rgb.shape)
        #print(l4_fea_rgb.shape)
        # l3_fea_depth_rgb = self.CA_SA_Enhance_3(l3_fea_depth, l3_fea_rgb)
        # l3_fea_depth = l3_fea_depth + l3_fea_depth_rgb
        #
        # l4_fea_rgb_depth = self.CA_SA_Enhance_4(l4_fea_rgb, l4_fea_depth)
        # l4_fea_rgb = l4_fea_rgb + l4_fea_rgb_depth
        #
        # l3_fea_rgb = self.encoder_l3_rgb(l3_fea_rgb)
        # l4_fea_rgb = self.encoder_l4_rgb(l4_fea_rgb)
        #
        # l3_fea_rgb_pool = self.avgpool(l3_fea_rgb)
        # l4_fea_rgb_pool = self.avgpool(l4_fea_rgb)
        # rgb_fea = l3_fea_rgb_pool + l4_fea_rgb_pool


        #l3_fea_depth = self.encoder_l3_depth(l3_fea_depth)
        #l4_fea_depth = self.encoder_l4_depth(l4_fea_depth)

        # l3_fea_depth_pool = self.avgpool(l3_fea_depth)
        # l4_fea_depth_pool = self.avgpool(l4_fea_depth)
        # depth_fea = l3_fea_depth_pool + l4_fea_depth_pool

        #rgbd_fea = rgb_fea + depth_fea
        #rgbd_fea = torch.cat((rgb_fea, depth_fea), dim=1)

        visual_vec = torch.flatten(rgb_fea, 1)  # (B, 512)
        if isinstance(ingredients_vec, torch.Tensor) and ingredients_vec.dim() == 1:
            ingredients_vec = ingredients_vec.view(1, -1)
        # 食材隐藏向量一方面作为独立语义特征，一方面生成视觉通道门控权重。
        ingredients_hidden = F.relu(self.ingredients_fc1(ingredients_vec.float()))  # (B, 256)
        attn = torch.sigmoid(self.ingredients_fc2(ingredients_hidden))  # (B, 512)
        visual_weighted = visual_vec * attn
        fused = torch.cat([visual_weighted, ingredients_hidden], dim=1)  # (B, 768)
        embedding = F.relu(self.fusion_fc(fused))  # (B, 256)
        results = []
        results.append(self.calorie(embedding).squeeze())
        results.append(self.mass(embedding).squeeze())
        results.append(self.fat(embedding).squeeze())
        results.append(self.carb(embedding).squeeze())
        results.append(self.protein(embedding).squeeze())
        return results


class MyResNetRGBDLegacy(nn.Module):
    """
    Legacy IMIR-Net RGB-D model (no ingredient-as-input fusion).
    Use this to evaluate older checkpoints trained without ingredients gating.

    该类保留旧 checkpoint 使用的 CA_SA 融合路径，forward 不接收
    ingredients_vec。推理脚本会根据 checkpoint key 自动选择当前模型或旧模型。
    """

    def __init__(self):
        super(MyResNetRGBDLegacy, self).__init__()
        self.model_rgb = MyResNet(Bottleneck, [3, 4, 23, 3])
        self.model_rgb.load_state_dict(torch.load('food2k_resnet101_0.0001.pth'), strict=False)

        self.model_depth = MyResNet(Bottleneck, [3, 4, 23, 3])
        self.model_depth.load_state_dict(torch.load('food2k_resnet101_0.0001.pth'), strict=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.calorie = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))
        self.mass = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))
        self.fat = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))
        self.carb = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))
        self.protein = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))

        self.encoder_l3_rgb = DilatedEncoder(in_channels=1024)
        self.encoder_l4_rgb = DilatedEncoder(in_channels=2048)

        # Present in some legacy checkpoints (even if not used in forward).
        self.encoder_l3_depth = DilatedEncoder(in_channels=1024)
        self.encoder_l4_depth = DilatedEncoder(in_channels=2048)

        self.CA_SA_Enhance_3 = CA_SA_Enhance(2048)
        self.CA_SA_Enhance_4 = CA_SA_Enhance(4096)

        self.con2d = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.con2d_t = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1, bias=True)
        self.fc_t = nn.Linear(512, 512)

    def forward(self, x_rgb, x_depth):
        # 旧版路径：先分别提取 RGB 与 depth 的 layer3 特征。
        x_fea = self.model_rgb.conv1(x_rgb)
        x_fea = self.model_rgb.bn1(x_fea)
        x_fea = self.model_rgb.relu(x_fea)
        x_fea = self.model_rgb.maxpool(x_fea)

        x_fea = self.model_rgb.layer1(x_fea)
        x_fea = self.model_rgb.layer2(x_fea)
        l3_fea_rgb = self.model_rgb.layer3(x_fea)

        x_fea_depth = self.model_depth.conv1(x_depth)
        x_fea_depth = self.model_depth.bn1(x_fea_depth)
        x_fea_depth = self.model_depth.relu(x_fea_depth)
        x_fea_depth = self.model_depth.maxpool(x_fea_depth)

        x_fea_depth = self.model_depth.layer1(x_fea_depth)
        x_fea_depth = self.model_depth.layer2(x_fea_depth)
        l3_fea_depth = self.model_depth.layer3(x_fea_depth)

        # CA_SA 先用 RGB/depth 互相生成注意力，再以残差形式增强 depth/RGB。
        l3_fea_depth_rgb = self.CA_SA_Enhance_3(l3_fea_depth, l3_fea_rgb)
        l3_fea_depth = l3_fea_depth + l3_fea_depth_rgb

        l4_fea_depth = self.model_depth.layer4(l3_fea_depth)
        l4_fea_rgb = self.model_rgb.layer4(l3_fea_rgb)

        l4_fea_rgb_depth = self.CA_SA_Enhance_4(l4_fea_rgb, l4_fea_depth)
        l4_fea_rgb = l4_fea_rgb + l4_fea_rgb_depth

        l3_fea_rgb_dilate = self.encoder_l3_rgb(l3_fea_rgb)
        l4_fea_rgb_dilate = self.encoder_l4_rgb(l4_fea_rgb)
        l4_fea_rgb_dilate_up = torch.nn.functional.interpolate(
            l4_fea_rgb_dilate, scale_factor=2, mode='bilinear', align_corners=False
        )
        rgb_fea_cat = torch.cat((l3_fea_rgb_dilate, l4_fea_rgb_dilate_up), dim=1)

        rgb_fea = self.con2d(rgb_fea_cat)
        rgb_fea = self.relu(rgb_fea)
        rgb_fea = self.avgpool(rgb_fea)

        rgb_fea_t = self.con2d_t(rgb_fea_cat)
        rgb_fea_t = self.relu(rgb_fea_t)
        rgb_fea_t = self.avgpool(rgb_fea_t)
        rgb_fea_t = torch.flatten(rgb_fea_t, 1)
        rgb_fea_t = self.fc_t(rgb_fea_t)

        embedding = torch.flatten(rgb_fea, 1)
        embedding = self.fc1(embedding)
        embedding = F.relu(embedding)
        # 返回五个营养回归结果，以及旧版训练中使用过的辅助视觉向量 rgb_fea_t。
        results = []
        results.append(self.calorie(embedding).squeeze())
        results.append(self.mass(embedding).squeeze())
        results.append(self.fat(embedding).squeeze())
        results.append(self.carb(embedding).squeeze())
        results.append(self.protein(embedding).squeeze())
        return results, rgb_fea_t
