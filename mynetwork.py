import torch
import torch.nn as nn
import clip
import timm
from typing import Tuple
from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1, conv3x3
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse


class CLIPVisionBranch(nn.Module):
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
                self._features[name] = output

            return _hook

        visual = self.clip_model.visual
        self._hook_handles.append(visual.layer3.register_forward_hook(_save("s3")))
        self._hook_handles.append(visual.layer4.register_forward_hook(_save("s4")))

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip_model.eval()
        return self

    def _forward_clip_visual_backbone(self, x: torch.Tensor) -> torch.Tensor:
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
            x_rgb = x_rgb * self.imagenet_std + self.imagenet_mean
        x_rgb = x_rgb.clamp(0.0, 1.0)
        x_rgb = (x_rgb - self.clip_mean) / self.clip_std

        self._features.clear()
        with torch.no_grad():
            _ = self._forward_clip_visual_backbone(x_rgb)

        clip_s3 = self._features.get("s3")
        clip_s4 = self._features.get("s4")
        if clip_s3 is None or clip_s4 is None:
            raise RuntimeError("CLIPVisionBranch hooks did not capture layer3/layer4 features.")
        return clip_s3.float(), clip_s4.float()


class RGB_Backbone_Swin(nn.Module):
    """
    RGB branch backbone: Swin Transformer Base (384x384).

    Returns stage3/stage4 features with channels (512, 1024).
    """

    def __init__(self, pretrained: bool = True, img_size: int = 384, in_chans: int = 3):
        super().__init__()
        self.img_size = int(img_size)
        self.backbone = timm.create_model(
            "swin_base_patch4_window12_384",
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3),
            img_size=img_size,
            in_chans=in_chans,
        )
        channels = list(self.backbone.feature_info.channels())
        if channels != [512, 1024]:
            raise RuntimeError(
                f"Unexpected Swin stage channels {channels}; expected [512, 1024] for stage3/stage4."
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        s3, s4 = self.backbone(x)
        # timm Swin features are often returned as NHWC; normalize to NCHW.
        if s3.dim() == 4 and s3.shape[1] != 512 and s3.shape[-1] == 512:
            s3 = s3.permute(0, 3, 1, 2).contiguous()
        if s4.dim() == 4 and s4.shape[1] != 1024 and s4.shape[-1] == 1024:
            s4 = s4.permute(0, 3, 1, 2).contiguous()
        return s3, s4


class Depth_Backbone_ConvNeXt(nn.Module):
    """
    Depth branch backbone: ConvNeXt-Tiny.

    Returns stage3/stage4 features with channels (384, 768).
    """

    def __init__(self, pretrained: bool = True, in_chans: int = 3):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3),
            in_chans=in_chans,
        )
        channels = list(self.backbone.feature_info.channels())
        if channels != [384, 768]:
            raise RuntimeError(
                f"Unexpected ConvNeXt stage channels {channels}; expected [384, 768] for stage3/stage4."
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s3, s4 = self.backbone(x)
        return s3, s4

class CA_Enhance(nn.Module):
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
        rgb_r = self.rgb_reduce(x_rgb)
        depth_r = self.depth_reduce(x_depth)

        rgb_ll, rgb_high = self.dwt(rgb_r)  # rgb_high: list length J
        depth_ll, _depth_high = self.dwt(depth_r)

        fused_ll = self.rgb_ll_conv(rgb_ll) + self.depth_ll_conv(depth_ll)

        out_dwt = self.idwt((fused_ll, rgb_high))
        if out_dwt.shape[-2:] != depth_r.shape[-2:]:
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
        out = self.lateral_norm(self.lateral_conv(feature))
        out = self.fpn_norm(self.fpn_conv(out))
        return self.dilated_encoder_blocks(out)


class MyBottleneck(nn.Module):

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
        out = out + identity
        return out

class MyResNet(nn.Module):
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
    def __init__(self, ingredients_dim: int = 255):
        super(MyResNetRGBD, self).__init__()
        self.rgb_encoder = RGB_Backbone_Swin(pretrained=True)
        self.depth_encoder = Depth_Backbone_ConvNeXt(pretrained=True)

        self.depth_proj_l3 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.depth_proj_l4 = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.clip_branch = CLIPVisionBranch()
        self.clip_fusion_s3 = nn.Sequential(
            nn.Conv2d(in_channels=1536, out_channels=512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.clip_fusion_s4 = nn.Sequential(
            nn.Conv2d(in_channels=3072, out_channels=1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.enable_clip_fusion = True

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.ingredients_dim = int(ingredients_dim)
        self.ingredients_fc1 = nn.Linear(self.ingredients_dim, 256)
        self.ingredients_fc2 = nn.Linear(256, 512)  # attention weights for 512-D visual vector (after GAP)
        self.fusion_fc = nn.Linear(512 + 256, 256)

        self.calorie = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))
        self.mass = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))
        self.fat = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))
        self.carb = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))
        self.protein = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 1))

        self.encoder_l3_rgb = DilatedEncoder(in_channels=512, encoder_channels=512, block_mid_channels=128)
        self.encoder_l4_rgb = DilatedEncoder(in_channels=1024, encoder_channels=1024, block_mid_channels=256)

        self.mbfm_l3 = MBFM(in_channels=512)
        self.mbfm_l4 = MBFM(in_channels=1024)

        self.con2d = nn.Conv2d(in_channels=1536, out_channels=512, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.con2d_t = nn.Conv2d(in_channels=1536, out_channels=512, kernel_size=3, padding=1, bias=True)
        self.fc_t = nn.Linear(512,512)

    def load_state_dict(self, state_dict, strict: bool = True):
        incompatible = super().load_state_dict(state_dict, strict=False)
        if not any(k.startswith("clip_fusion_s3.") or k.startswith("clip_fusion_s4.") for k in state_dict.keys()):
            self.enable_clip_fusion = False
        return incompatible

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        for k in list(state.keys()):
            if k.startswith("clip_branch.clip_model."):
                state.pop(k)
        return state

    def forward(self, x_rgb, x_depth, ingredients_vec):
        rgb_s3, rgb_s4 = self.rgb_encoder(x_rgb)

        if self.enable_clip_fusion:
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
        l3_fea_depth, l4_fea_depth = self.depth_encoder(x_depth)
        if l3_fea_depth.shape[-2:] != l3_fea_rgb.shape[-2:]:
            l3_fea_depth = F.interpolate(l3_fea_depth, size=l3_fea_rgb.shape[-2:], mode="bilinear", align_corners=False)
        if l4_fea_depth.shape[-2:] != l4_fea_rgb.shape[-2:]:
            l4_fea_depth = F.interpolate(l4_fea_depth, size=l4_fea_rgb.shape[-2:], mode="bilinear", align_corners=False)

        l3_fea_depth = self.depth_proj_l3(l3_fea_depth)
        l4_fea_depth = self.depth_proj_l4(l4_fea_depth)

        # Layer 3 Fusion (RGB <- RGBD)
        l3_fea_rgb = self.mbfm_l3(l3_fea_rgb, l3_fea_depth)

        # Layer 4 Fusion (RGB <- RGBD)
        l4_fea_rgb = self.mbfm_l4(l4_fea_rgb, l4_fea_depth)


        l3_fea_rgb_dilate = self.encoder_l3_rgb(l3_fea_rgb)
        l4_fea_rgb_dilate = self.encoder_l4_rgb(l4_fea_rgb)

        # l3_fea_rgb_pool = self.avgpool(l3_fea_rgb_dilate)
        # l4_fea_rgb_pool = self.avgpool(l4_fea_rgb_dilate)
        l4_fea_rgb_dilate_up = torch.nn.functional.interpolate(
            l4_fea_rgb_dilate, size=l3_fea_rgb_dilate.shape[-2:], mode='bilinear', align_corners=False
        )
        #rgb_fea = l3_fea_rgb_pool + l4_fea_rgb_pool_up
        rgb_fea_cat = torch.cat((l3_fea_rgb_dilate, l4_fea_rgb_dilate_up), dim=1)
        #print(rgb_fea.shape)
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
        results = []
        results.append(self.calorie(embedding).squeeze())
        results.append(self.mass(embedding).squeeze())
        results.append(self.fat(embedding).squeeze())
        results.append(self.carb(embedding).squeeze())
        results.append(self.protein(embedding).squeeze())
        return results, rgb_fea_t
