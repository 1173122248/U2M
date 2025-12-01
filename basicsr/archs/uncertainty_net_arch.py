"""
Lightweight Uncertainty Prediction Network for Adaptive Diffusion
Author: UPSR-Enhanced
Date: 2025-10-22

This module implements a trainable uncertainty estimation network that predicts
pixel-wise uncertainty maps for adaptive noise scheduling in diffusion models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.arch_util import default_init_weights


class DepthwiseSeparableConv(nn.Module):
    """轻量级深度可分离卷积"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, 
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取器"""
    def __init__(self, in_channels, base_channels=32):
        super().__init__()
        
        # 三个并行分支提取不同尺度特征
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.GELU()
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 5, 1, 2),
            nn.GELU()
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, 1, 3),
            nn.GELU()
        )
        
        # 融合
        self.fusion = nn.Conv2d(base_channels * 3, base_channels, 1)
        
    def forward(self, x):
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)
        
        fused = torch.cat([f1, f2, f3], dim=1)
        out = self.fusion(fused)
        return out


class SpatialAttention(nn.Module):
    """空间注意力模块 - 聚焦不确定区域"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.GELU(),
            nn.Conv2d(channels // 8, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention = self.conv(x)
        return x * attention, attention


class FrequencyAwareModule(nn.Module):
    """频域感知模块 - 高频区域通常有更高不确定性"""
    def __init__(self, channels):
        super().__init__()
        self.conv_freq = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
        
    def forward(self, x):
        # 简单的高频提取（拉普拉斯算子）
        laplacian_kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        laplacian_kernel = laplacian_kernel.repeat(x.shape[1], 1, 1, 1)
        
        high_freq = F.conv2d(x, laplacian_kernel, padding=1, groups=x.shape[1])
        high_freq_feat = self.conv_freq(high_freq)
        
        return x + high_freq_feat


class UncertaintyPredictionNet(nn.Module):
    """
    轻量级不确定性预测网络
    
    输入:
        - LR图像 (B, 3, H/4, W/4)
        - SR图像 (B, 3, H, W) - 来自MSE模型
        - Bicubic上采样 (B, 3, H, W)
    
    输出:
        - 不确定性图 (B, 1, H, W) - 值域 [0, 1]
        - 中间特征用于辅助监督
    """
    
    def __init__(self, base_channels=32, num_blocks=4):
        super().__init__()
        
        self.base_channels = base_channels
        
        # === 输入处理 ===
        # LR特征提取
        self.lr_encoder = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, 1, 1),
            nn.GELU(),
            nn.PixelShuffle(4),  # 上采样到HR尺寸
            nn.Conv2d(base_channels // 16, base_channels, 3, 1, 1),
            nn.GELU()
        )
        
        # SR和Bicubic特征提取
        self.sr_encoder = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, 1, 1),
            nn.GELU()
        )
        
        self.bicubic_encoder = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, 1, 1),
            nn.GELU()
        )
        
        # 残差特征
        self.residual_encoder = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, 1, 1),
            nn.GELU()
        )
        
        # === 多尺度特征融合 ===
        self.multiscale_fusion = MultiScaleFeatureExtractor(
            base_channels * 4, base_channels * 2
        )
        
        # === 频域感知 ===
        self.freq_module = FrequencyAwareModule(base_channels * 2)
        
        # === 特征提取主干 ===
        self.backbone = nn.ModuleList([
            DepthwiseSeparableConv(base_channels * 2, base_channels * 2)
            for _ in range(num_blocks)
        ])
        
        # === 空间注意力 ===
        self.spatial_attention = SpatialAttention(base_channels * 2)
        
        # === 不确定性预测头 ===
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels // 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(base_channels // 2, 1, 1),
            nn.Sigmoid()  # 输出 [0, 1]
        )
        
        # === 辅助预测头（用于多任务学习）===
        # 预测边缘强度
        self.edge_head = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels // 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(base_channels // 2, 1, 1),
            nn.Sigmoid()
        )
        
        # 预测纹理复杂度
        self.texture_head = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels // 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(base_channels // 2, 1, 1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            default_init_weights(m, scale=0.1)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, lr, sr_mse, bicubic, return_features=False):
        """
        Args:
            lr: LR图像 (B, 3, H/4, W/4)
            sr_mse: MSE模型的SR结果 (B, 3, H, W)
            bicubic: Bicubic上采样 (B, 3, H, W)
            return_features: 是否返回中间特征用于可视化/分析
        
        Returns:
            uncertainty: 不确定性图 (B, 1, H, W)
            aux_outputs: 辅助输出字典（如果 return_features=True）
        """
        
        # 特征提取
        lr_feat = self.lr_encoder(lr)
        sr_feat = self.sr_encoder(sr_mse)
        bicubic_feat = self.bicubic_encoder(bicubic)
        
        # 残差特征
        residual = sr_mse - bicubic
        residual_feat = self.residual_encoder(residual)
        
        # 融合所有特征
        fused_feat = torch.cat([lr_feat, sr_feat, bicubic_feat, residual_feat], dim=1)
        fused_feat = self.multiscale_fusion(fused_feat)
        
        # 频域增强
        fused_feat = self.freq_module(fused_feat)
        
        # 主干网络
        for block in self.backbone:
            fused_feat = fused_feat + block(fused_feat)
        
        # 空间注意力
        fused_feat, attention_map = self.spatial_attention(fused_feat)
        
        # 预测不确定性
        uncertainty = self.uncertainty_head(fused_feat)
        
        if return_features:
            # 辅助输出
            edge_map = self.edge_head(fused_feat)
            texture_map = self.texture_head(fused_feat)
            
            aux_outputs = {
                'uncertainty': uncertainty,
                'edge_map': edge_map,
                'texture_map': texture_map,
                'attention_map': attention_map,
                'features': fused_feat
            }
            return aux_outputs
        else:
            return uncertainty


class AdaptiveNoiseScheduler(nn.Module):
    """
    自适应噪声调度器 - 将不确定性映射到加噪强度
    
    改进点：
    1. 非线性映射（可学习）
    2. 动态范围调整
    3. 平滑过渡
    """
    
    def __init__(self, min_noise=0.2, max_noise=1.0, sharpness=2.0):
        super().__init__()
        
        self.min_noise = min_noise
        self.max_noise = max_noise
        
        # 可学习的映射参数
        self.alpha = nn.Parameter(torch.tensor(sharpness))  # 控制曲线陡峭度
        self.beta = nn.Parameter(torch.tensor(0.5))  # 控制中心点
        
        # 平滑核（避免突变）
        self.smooth_kernel_size = 5
        
    def forward(self, uncertainty, smooth=True):
        """
        将不确定性映射到加噪强度
        
        使用sigmoid型非线性映射:
        noise_scale = min + (max - min) * sigmoid(alpha * (uncertainty - beta))
        
        Args:
            uncertainty: (B, 1, H, W) 不确定性图，值域 [0, 1]
            smooth: 是否进行空间平滑
        
        Returns:
            noise_scale: (B, 1, H, W) 噪声缩放因子
        """
        
        # 非线性映射
        centered = uncertainty - self.beta
        scaled = self.alpha * centered
        mapped = torch.sigmoid(scaled)
        
        # 映射到 [min_noise, max_noise]
        noise_scale = self.min_noise + (self.max_noise - self.min_noise) * mapped
        
        # 可选：空间平滑（避免噪声强度突变）
        if smooth:
            noise_scale = self._smooth_map(noise_scale)
        
        return noise_scale
    
    def _smooth_map(self, x):
        """使用高斯模糊平滑噪声强度图"""
        kernel_size = self.smooth_kernel_size
        sigma = kernel_size / 6.0
        
        # 创建高斯核
        channels = x.shape[1]
        kernel = self._get_gaussian_kernel(kernel_size, sigma, channels, x.device, x.dtype)
        
        # 应用平滑
        padding = kernel_size // 2
        smoothed = F.conv2d(x, kernel, padding=padding, groups=channels)
        
        return smoothed
    
    def _get_gaussian_kernel(self, kernel_size, sigma, channels, device, dtype):
        """生成高斯核"""
        x_cord = torch.arange(kernel_size, device=device, dtype=dtype)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        gaussian_kernel = (1. / (2. * 3.14159 * variance)) * \
                          torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
        
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
        
        return gaussian_kernel


# ============ 辅助函数：提取训练监督信号 ============

def compute_edge_map(image):
    """
    使用Sobel算子计算边缘强度
    
    Args:
        image: (B, C, H, W)
    Returns:
        edge_map: (B, 1, H, W)
    """
    gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    
    edge_x = F.conv2d(gray, sobel_x, padding=1)
    edge_y = F.conv2d(gray, sobel_y, padding=1)
    
    edge_map = torch.sqrt(edge_x ** 2 + edge_y ** 2)
    edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min() + 1e-8)
    
    return edge_map


def compute_texture_complexity(image, window_size=7):
    """
    使用局部标准差计算纹理复杂度
    
    Args:
        image: (B, C, H, W)
        window_size: 局部窗口大小
    Returns:
        texture_map: (B, 1, H, W)
    """
    gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    
    # 计算局部均值
    kernel = torch.ones(1, 1, window_size, window_size, 
                       dtype=image.dtype, device=image.device) / (window_size ** 2)
    local_mean = F.conv2d(gray, kernel, padding=window_size // 2)
    
    # 计算局部方差
    local_mean_sq = F.conv2d(gray ** 2, kernel, padding=window_size // 2)
    local_var = local_mean_sq - local_mean ** 2
    
    texture_map = torch.sqrt(torch.clamp(local_var, min=0))
    texture_map = (texture_map - texture_map.min()) / (texture_map.max() - texture_map.min() + 1e-8)
    
    return texture_map


# ============ 损失函数 ============

class UncertaintyLoss(nn.Module):
    """
    不确定性网络的复合损失函数
    
    包含:
    1. 自监督损失 - 基于扩散模型的预测误差
    2. 辅助任务损失 - 边缘检测、纹理估计
    3. 平滑正则化 - 避免噪声强度过于跳跃
    4. 范围约束 - 确保不确定性在合理范围
    """
    
    def __init__(self, 
                 lambda_self_sup=1.0,
                 lambda_edge=0.3,
                 lambda_texture=0.3,
                 lambda_smooth=0.1,
                 lambda_range=0.2):
        super().__init__()
        
        self.lambda_self_sup = lambda_self_sup
        self.lambda_edge = lambda_edge
        self.lambda_texture = lambda_texture
        self.lambda_smooth = lambda_smooth
        self.lambda_range = lambda_range
        
    def forward(self, outputs, targets):
        """
        Args:
            outputs: 网络输出字典
                - uncertainty: (B, 1, H, W)
                - edge_map: (B, 1, H, W)
                - texture_map: (B, 1, H, W)
            targets: 目标字典
                - gt_edge: (B, 1, H, W) GT的边缘图
                - gt_texture: (B, 1, H, W) GT的纹理图
                - prediction_error: (B, 3, H, W) 扩散模型的预测误差
        """
        
        uncertainty = outputs['uncertainty']
        total_loss = 0.0
        loss_dict = {}
        
        # 1. 自监督损失：不确定性应该与预测误差相关
        if 'prediction_error' in targets:
            pred_error = targets['prediction_error']
            # 计算像素级预测误差的强度
            error_magnitude = torch.mean(pred_error ** 2, dim=1, keepdim=True)
            error_magnitude = (error_magnitude - error_magnitude.min()) / \
                             (error_magnitude.max() - error_magnitude.min() + 1e-8)
            
            # 不确定性高的地方，应该允许更大的预测误差
            # 使用负相关损失
            self_sup_loss = F.mse_loss(uncertainty, error_magnitude)
            total_loss += self.lambda_self_sup * self_sup_loss
            loss_dict['self_sup'] = self_sup_loss
        
        # 2. 边缘检测辅助损失
        if 'edge_map' in outputs and 'gt_edge' in targets:
            edge_loss = F.l1_loss(outputs['edge_map'], targets['gt_edge'])
            total_loss += self.lambda_edge * edge_loss
            loss_dict['edge'] = edge_loss
        
        # 3. 纹理复杂度辅助损失
        if 'texture_map' in outputs and 'gt_texture' in targets:
            texture_loss = F.l1_loss(outputs['texture_map'], targets['gt_texture'])
            total_loss += self.lambda_texture * texture_loss
            loss_dict['texture'] = texture_loss
        
        # 4. 平滑正则化：避免不确定性图过于跳跃
        smooth_loss = self._total_variation_loss(uncertainty)
        total_loss += self.lambda_smooth * smooth_loss
        loss_dict['smooth'] = smooth_loss
        
        # 5. 范围约束：鼓励使用不确定性的全范围
        range_loss = self._range_diversity_loss(uncertainty)
        total_loss += self.lambda_range * range_loss
        loss_dict['range'] = range_loss
        
        loss_dict['total'] = total_loss
        
        return total_loss, loss_dict
    
    def _total_variation_loss(self, x):
        """全变分损失 - 鼓励空间平滑"""
        diff_i = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_j = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        tv_loss = torch.mean(diff_i) + torch.mean(diff_j)
        return tv_loss
    
    def _range_diversity_loss(self, x):
        """范围多样性损失 - 避免所有像素的不确定性都相同"""
        # 计算标准差，鼓励有差异
        std = torch.std(x)
        target_std = 0.2  # 期望标准差
        range_loss = F.mse_loss(std, torch.tensor(target_std, device=x.device))
        return range_loss


if __name__ == '__main__':
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建网络
    uncertainty_net = UncertaintyPredictionNet(base_channels=32, num_blocks=4).to(device)
    noise_scheduler = AdaptiveNoiseScheduler(min_noise=0.2, max_noise=1.0).to(device)
    
    # 测试输入
    batch_size = 2
    lr = torch.randn(batch_size, 3, 64, 64).to(device)
    sr_mse = torch.randn(batch_size, 3, 256, 256).to(device)
    bicubic = torch.randn(batch_size, 3, 256, 256).to(device)
    
    # 前向传播
    outputs = uncertainty_net(lr, sr_mse, bicubic, return_features=True)
    
    print("=== 不确定性网络测试 ===")
    print(f"输入 - LR: {lr.shape}, SR: {sr_mse.shape}, Bicubic: {bicubic.shape}")
    print(f"输出 - Uncertainty: {outputs['uncertainty'].shape}")
    print(f"       Edge Map: {outputs['edge_map'].shape}")
    print(f"       Texture Map: {outputs['texture_map'].shape}")
    print(f"       Attention Map: {outputs['attention_map'].shape}")
    
    # 测试噪声调度器
    noise_scale = noise_scheduler(outputs['uncertainty'])
    print(f"\n噪声缩放: {noise_scale.shape}")
    print(f"范围: [{noise_scale.min().item():.3f}, {noise_scale.max().item():.3f}]")
    
    # 参数统计
    total_params = sum(p.numel() for p in uncertainty_net.parameters())
    print(f"\n不确定性网络参数量: {total_params / 1e6:.2f}M")
    
    scheduler_params = sum(p.numel() for p in noise_scheduler.parameters())
    print(f"噪声调度器参数量: {scheduler_params}")
    
    print("\n✅ 测试通过！")
