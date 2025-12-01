"""
U²M: Uncertainty U-Net with Multi-scale Modulation

A lightweight encoder-decoder network for dynamic uncertainty estimation
in both forward diffusion and reverse denoising processes.

Author: Enhanced UPSR
Date: 2025-11-05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.arch_util import default_init_weights


class ConvBlock(nn.Module):
    """基础卷积块：Conv3x3 + BN + ReLU"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DilatedResBlock(nn.Module):
    """膨胀卷积残差块 - 扩大感受野"""
    def __init__(self, channels, dilation=2):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class DeconvBlock(nn.Module):
    """反卷积上采样块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.deconv(x)))


class U2M(nn.Module):
    """
    U²M: Uncertainty U-Net with Multi-scale Modulation
    
    轻量级UNet架构，用于动态不确定性估计
    
    输入：
        - 6通道图像 [LR, Reference]
          * 正向阶段：[LR, SR_pre]
          * 反向阶段：[LR, x_t]
    
    输出：
        - 单通道不确定性图 σ_map ∈ ℝ^{1×H×W}
    
    Args:
        in_channels: 输入通道数，默认6 (LR:3 + Ref:3)
        base_channels: 基础通道数，默认32
        num_encoder_blocks: 编码器层数，默认4
        use_attention: 是否使用注意力机制
    """
    
    def __init__(self, in_channels=6, base_channels=32, num_encoder_blocks=4, use_attention=False):
        super().__init__()
        
        self.num_encoder_blocks = num_encoder_blocks
        self.use_attention = use_attention
        
        # === Encoder ===
        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        current_channels = in_channels
        for i in range(num_encoder_blocks):
            out_channels = base_channels * (2 ** i)
            self.encoder_blocks.append(
                nn.Sequential(
                    ConvBlock(current_channels, out_channels),
                    ConvBlock(out_channels, out_channels)
                )
            )
            # 下采样（除了最后一层）
            if i < num_encoder_blocks - 1:
                self.downsample_layers.append(
                    nn.Conv2d(out_channels, out_channels, 3, 2, 1)
                )
            current_channels = out_channels
        
        # === Bottleneck ===
        bottleneck_channels = base_channels * (2 ** (num_encoder_blocks - 1))
        self.bottleneck = nn.Sequential(
            DilatedResBlock(bottleneck_channels, dilation=2),
            DilatedResBlock(bottleneck_channels, dilation=4),
        )
        
        # === Global Head (新增: 输出global_v) ===
        # 从bottleneck全局池化得到图像级的自适应强度系数
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, C, H, W) -> (B, C, 1, 1)
            nn.Conv2d(bottleneck_channels, bottleneck_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels // 4, 1, 1),
            nn.Sigmoid()  # 输出0~1,表示全局噪声注入强度
        )
        
        # === Decoder ===
        self.decoder_blocks = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        
        for i in range(num_encoder_blocks - 1, 0, -1):
            in_ch = base_channels * (2 ** i)
            out_ch = base_channels * (2 ** (i - 1))
            
            # 上采样
            self.upsample_layers.append(DeconvBlock(in_ch, out_ch))
            
            # 解码器块（接收skip connection，所以输入是out_ch*2）
            self.decoder_blocks.append(
                nn.Sequential(
                    ConvBlock(out_ch * 2, out_ch),
                    ConvBlock(out_ch, out_ch)
                )
            )
        
        # === 可选注意力模块 ===
        if use_attention:
            self.attention = SpatialAttention(base_channels)
        
        # === Output Head ===
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, 1, 1),
            nn.Softplus()  # 保证输出为正值
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            default_init_weights(m, scale=0.1)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播 (改进版: 输出sigma_map和global_v)
        
        Args:
            x: 输入张量 (B, 6, H, W)
        
        Returns:
            sigma_map: 空间不确定性图 (B, 1, H, W) - 局部自适应
            global_v: 全局强度系数 (B, 1, 1, 1) - 图像级自适应
        """
        # 保存skip connections
        skip_connections = []
        
        # Encoder
        for i, encoder_block in enumerate(self.encoder_blocks):
            x = encoder_block(x)
            skip_connections.append(x)
            
            # 下采样（除了最后一层）
            if i < self.num_encoder_blocks - 1:
                x = self.downsample_layers[i](x)
        
        # Bottleneck
        bottleneck_features = self.bottleneck(x)
        
        # === 新增: Global Head (全局自适应强度) ===
        global_v = self.global_head(bottleneck_features)  # (B, 1, 1, 1)
        
        # Decoder（跳过bottleneck的skip connection）
        x = bottleneck_features
        skip_connections = skip_connections[:-1][::-1]  # 反转，从深到浅
        
        for i, (upsample, decoder_block) in enumerate(zip(self.upsample_layers, self.decoder_blocks)):
            x = upsample(x)
            
            # Skip connection
            skip = skip_connections[i]
            # 处理尺寸不匹配
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)
            x = decoder_block(x)
        
        # 可选注意力
        if self.use_attention:
            x = self.attention(x)
        
        # Output
        sigma_map = self.output_conv(x)
        
        return sigma_map, global_v


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention = self.conv(x)
        return x * attention


class TimeDecayScheduler(nn.Module):
    """
    时间衰减函数 γ(t)
    
    用于反向去噪阶段控制不确定性注入强度
    
    Args:
        decay_type: 衰减类型，'linear' 或 'cosine'
        min_value: 最小衰减值
    """
    def __init__(self, decay_type='cosine', min_value=0.0):
        super().__init__()
        self.decay_type = decay_type
        self.min_value = min_value
    
    def forward(self, t, T):
        """
        计算时间衰减系数
        
        Args:
            t: 当前时间步 (B,) 或标量
            T: 总时间步数
        
        Returns:
            gamma: 衰减系数 (B, 1, 1, 1) 或标量
        """
        if isinstance(t, torch.Tensor):
            normalized_t = t.float() / T
            
            if self.decay_type == 'linear':
                gamma = 1.0 - normalized_t
            elif self.decay_type == 'cosine':
                gamma = 0.5 * (1.0 + torch.cos(normalized_t * 3.14159))
            else:
                raise ValueError(f"Unknown decay type: {self.decay_type}")
            
            # 应用最小值约束
            gamma = torch.clamp(gamma, min=self.min_value)
            
            # 扩展维度以匹配特征图 (B,) -> (B, 1, 1, 1)
            if gamma.dim() == 1:
                gamma = gamma.view(-1, 1, 1, 1)
            
            return gamma
        else:
            # 标量输入
            normalized_t = float(t) / T
            
            if self.decay_type == 'linear':
                gamma = 1.0 - normalized_t
            elif self.decay_type == 'cosine':
                gamma = 0.5 * (1.0 + torch.cos(torch.tensor(normalized_t * 3.14159)))
            
            return max(gamma, self.min_value)


# ============ 测试代码 ============

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("U²M Network Testing")
    print("=" * 60)
    
    # 创建网络
    u2m = U2M(
        in_channels=6,
        base_channels=32,
        num_encoder_blocks=4,
        use_attention=True
    ).to(device)
    
    # 测试输入
    batch_size = 4
    H, W = 256, 256
    
    # 模拟输入：[LR, Reference]
    lr = torch.randn(batch_size, 3, H, W).to(device)
    ref = torch.randn(batch_size, 3, H, W).to(device)
    x_input = torch.cat([lr, ref], dim=1)
    
    print(f"\n输入形状: {x_input.shape}")
    
    # 前向传播
    with torch.no_grad():
        sigma_map = u2m(x_input)
    
    print(f"输出形状: {sigma_map.shape}")
    print(f"输出范围: [{sigma_map.min().item():.4f}, {sigma_map.max().item():.4f}]")
    
    # 参数统计
    total_params = sum(p.numel() for p in u2m.parameters())
    trainable_params = sum(p.numel() for p in u2m.parameters() if p.requires_grad)
    
    print(f"\n总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数: {trainable_params / 1e6:.2f}M")
    
    # 测试时间衰减调度器
    print("\n" + "=" * 60)
    print("Time Decay Scheduler Testing")
    print("=" * 60)
    
    scheduler_linear = TimeDecayScheduler(decay_type='linear', min_value=0.0)
    scheduler_cosine = TimeDecayScheduler(decay_type='cosine', min_value=0.0)
    
    T = 50  # 总步数
    t_samples = torch.tensor([0, 10, 25, 40, 49])
    
    print(f"\n总时间步: {T}")
    print(f"采样时间步: {t_samples.tolist()}")
    
    gamma_linear = scheduler_linear(t_samples, T)
    gamma_cosine = scheduler_cosine(t_samples, T)
    
    print(f"\n线性衰减: {gamma_linear.squeeze().tolist()}")
    print(f"余弦衰减: {gamma_cosine.squeeze().tolist()}")
    
    # 测试完整流程：正向 + 反向
    print("\n" + "=" * 60)
    print("Full Pipeline Testing")
    print("=" * 60)
    
    # 正向阶段
    print("\n[正向加噪阶段]")
    lr_fwd = torch.randn(2, 3, 256, 256).to(device)
    sr_pre = torch.randn(2, 3, 256, 256).to(device)
    input_fwd = torch.cat([lr_fwd, sr_pre], dim=1)
    
    with torch.no_grad():
        sigma_fwd = u2m(input_fwd)
    print(f"输入: [LR, SR_pre] -> {input_fwd.shape}")
    print(f"输出: σ_fwd -> {sigma_fwd.shape}")
    print(f"σ_fwd 范围: [{sigma_fwd.min():.4f}, {sigma_fwd.max():.4f}]")
    
    # 反向阶段
    print("\n[反向去噪阶段]")
    t_step = 30
    x_t = torch.randn(2, 3, 256, 256).to(device)
    input_rev = torch.cat([lr_fwd, x_t], dim=1)
    
    with torch.no_grad():
        sigma_t = u2m(input_rev)
        gamma_t = scheduler_cosine(torch.tensor([t_step]), T)
        sigma_inject = gamma_t * sigma_t
    
    print(f"时间步: t={t_step}")
    print(f"输入: [LR, x_t] -> {input_rev.shape}")
    print(f"输出: σ_t -> {sigma_t.shape}")
    print(f"衰减系数: γ(t) = {gamma_t.item():.4f}")
    print(f"注入强度: σ_inject = γ(t) * σ_t")
    print(f"σ_inject 范围: [{sigma_inject.min():.4f}, {sigma_inject.max():.4f}]")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
