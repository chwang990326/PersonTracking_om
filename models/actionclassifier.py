import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import List, Tuple, Optional
from functools import partial
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from torchvision.transforms import v2

# --- 基础卷积组件 ---

def conv_3xnxn(inp, oup, kernel_size=3, stride=3, groups=1):
    # 此处时间维度的 stride 为 2，会将输入帧数减半
    return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (2, stride, stride), (1, 0, 0), groups=groups)

def conv_1xnxn(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (1, kernel_size, kernel_size), (1, stride, stride), (0, 0, 0), groups=groups)

def conv_1x1x1(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (1, 1, 1), (1, 1, 1), (0, 0, 0), groups=groups)

def conv_3x3x3(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (3, 3, 3), (1, 1, 1), (1, 1, 1), groups=groups)

def conv_5x5x5(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (5, 5, 5), (1, 1, 1), (2, 2, 2), groups=groups)

def bn_3d(dim):
    return nn.BatchNorm3d(dim)

# --- 核心模块 ---

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = conv_1x1x1(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = conv_1x1x1(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = bn_3d(dim)
        self.conv1 = conv_1x1x1(dim, dim, 1)
        self.conv2 = conv_1x1x1(dim, dim, 1)
        self.attn = conv_5x5x5(dim, dim, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = bn_3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, C, T, H, W)
        return x

class SpeicalPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = conv_3xnxn(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])

    def forward(self, x):
        x = self.proj(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = conv_1xnxn(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])

    def forward(self, x):
        x = self.proj(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x

# --- 主模型 (UniFormer) ---

class UniFormer(nn.Module):
    def __init__(self, depth=[3, 4, 8, 3], num_classes=7, img_size=224, in_chans=3, 
                 embed_dim=[64, 128, 320, 512], head_dim=64, mlp_ratio=4., 
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        self.num_classes = num_classes
        
        # Patch Embeddings
        self.patch_embed1 = SpeicalPatchEmbed(img_size=img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(img_size=img_size // 4, patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed(img_size=img_size // 8, patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 16, patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3])

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        num_heads = [dim // head_dim for dim in embed_dim]

        # Stages
        self.blocks1 = nn.ModuleList([CBlock(dim=embed_dim[0], mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i]) for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([CBlock(dim=embed_dim[1], mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i+depth[0]]) for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([SABlock(dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]]) for i in range(depth[2])])
        self.blocks4 = nn.ModuleList([SABlock(dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]]) for i in range(depth[3])])

        self.norm = bn_3d(embed_dim[-1])
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1: x = blk(x)
        x = self.patch_embed2(x)
        for blk in self.blocks2: x = blk(x)
        x = self.patch_embed3(x)
        for blk in self.blocks3: x = blk(x)
        x = self.patch_embed4(x)
        for blk in self.blocks4: x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        # 严格匹配 MMAction2 I3DHead 逻辑：
        # Global Average Pooling (N, C, T, H, W) -> (N, C, 1, 1, 1)
        # 无论输入帧数是 8 还是 16，池化层都会将其压缩为 1x1x1
        x = nn.functional.adaptive_avg_pool3d(x, (1, 1, 1))
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

# --- 辅助函数 ---

def uniformer_small(num_classes=6):
    return UniFormer(depth=[3, 4, 8, 3], embed_dim=[64, 128, 320, 512], 
                     head_dim=64, drop_path_rate=0.1, num_classes=num_classes)

def preprocess_crops_for_video_cls(crops: List[np.ndarray], debug_path: str = None) -> torch.Tensor:
    """
    完全对齐 MMAction2 Config 的预处理流水线，支持保存 CenterCrop 结果到本地
    已修改：目标帧数从 8 帧改为 16 帧
    """
    if debug_path and not os.path.exists(debug_path):
        os.makedirs(debug_path)

    # --- 修改点：target_frames 从 8 改为 16 ---
    target_frames = 16
    
    # 抽帧逻辑：如果输入帧数不等于16，则进行均匀采样
    if len(crops) != target_frames:
        indices = np.linspace(0, len(crops) - 1, target_frames).astype(int)
        crops = [crops[i] for i in indices]

    # 与测试集一致：先缩放到 [0, 1]，再使用 Kinetics 统计量归一化
    mean = [0.43216, 0.394666, 0.37645]
    std = [0.22803, 0.22145, 0.216989]
    to_float = v2.ToDtype(torch.float32, scale=True)
    resize = v2.Resize((224, 224), antialias=True)
    normalize = v2.Normalize(mean=mean, std=std)
    processed_tensors = []
    
    for idx, crop in enumerate(crops):
        # 重要：OpenCV 是 BGR，必须转 RGB 才能匹配训练配置
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # 与测试集评估保持一致：ToDtype(scale=True) -> Resize(antialias=True) -> Normalize
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        img_tensor = to_float(img_tensor)
        img_tensor = resize(img_tensor)
        
        if debug_path:
            save_img = (img_tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            save_name = os.path.join(debug_path, f"crop_frame_{idx:02d}.jpg")
            cv2.imwrite(save_name, save_img)

        # 归一化（与测试集一致）
        img_tensor = normalize(img_tensor)
        processed_tensors.append(img_tensor)
    
    # 形状变换: (T, C, H, W) -> (C, T, H, W) -> (1, C, T, H, W)
    # 这里的 T 现在是 16
    video_tensor = torch.stack(processed_tensors).permute(1, 0, 2, 3).unsqueeze(0)
    return video_tensor

def postprocess(outputs: torch.Tensor) -> Tuple[List[str], List[float]]:
    labels = ["climbing", "falling_down", "normal", "reaching_high", "sleeping", "smoking"]
    probs = torch.softmax(outputs, dim=1)
    conf, pred_class = torch.max(probs, dim=1)
    
    return [labels[pred_class.item()]], [conf.item()]

def crop_and_pad(frame, box, margin_percent=50):
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    margin_x, margin_y = int(w * margin_percent / 100), int(h * margin_percent / 100)
    
    x1, y1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
    x2, y2 = min(frame.shape[1], x2 + margin_x), min(frame.shape[0], y2 + margin_y)
    
    size = max(y2 - y1, x2 - x1)
    center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2
    half_size = size // 2
    
    y_min, y_max = max(0, center_y - half_size), min(frame.shape[0], center_y + half_size)
    x_min, x_max = max(0, center_x - half_size), min(frame.shape[1], center_x + half_size)
    square_crop = frame[y_min:y_max, x_min:x_max]

    # 仅负责裁剪，缩放统一交给 preprocess_crops_for_video_cls
    return square_crop