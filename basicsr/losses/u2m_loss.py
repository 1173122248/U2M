"""
U²M Loss: Uncertainty Modulation Loss
基于residual alignment和calibration的不确定性训练损失

参考思想:
1. Residual Alignment: σ应该与重建残差对齐
2. Calibration: σ应该与实际误差相关
3. Smoothness: σ应该空间平滑
4. Magnitude: σ应该在合理范围内
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class U2MLoss(nn.Module):
    """
    U²M网络的训练损失 (统一框架版本)
    
    核心思想:
    - σ应该预测"难以重建"的区域 (高残差区域)
    - σ应该与实际重建误差相关 (calibration)
    - σ应该空间平滑但有变化
    
    统一框架:
    - 内部权重会被归一化 (lambda_r + lambda_c + lambda_s + lambda_m = 1.0)
    - 外部通过total_weight控制U²M损失相对于主干损失的比例
    """
    
    def __init__(self,
                 lambda_r=1.0,      # residual alignment权重
                 lambda_c=0.5,      # calibration权重
                 lambda_s=0.01,     # smoothness权重
                 lambda_m=0.01,     # magnitude权重
                 sigma_min=0.01,    # σ最小值
                 sigma_max=0.45,    # σ最大值 (相对值)
                 calib_samples=1024,  # calibration采样点数
                 gamma=1.0):        # residual weighting指数
        super().__init__()
        
        # ===== 原始权重 (不归一化) =====
        self.lambda_r = lambda_r
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.lambda_m = lambda_m
        
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.calib_samples = calib_samples
        self.gamma = gamma
    
    @staticmethod
    def normalize_map(x, eps=1e-8):
        """将张量归一化到[0,1]"""
        b = x.view(x.size(0), -1)
        mn = b.min(dim=1, keepdim=True)[0]
        mx = b.max(dim=1, keepdim=True)[0]
        norm = (b - mn) / (mx - mn + eps)
        return norm.view(x.size())
    
    def residual_alignment_loss(self, sigma, pred, gt, lr):
        """
        Residual Alignment Loss
        
        思想: σ应该与重建残差对齐
        - r_pred = |pred - lr|: 预测的残差
        - r_gt = |gt - lr|: 真实的残差
        - σ应该在高残差区域更大
        
        Args:
            sigma: (B,1,H,W) 预测的不确定性
            pred: (B,C,H,W) 网络预测的SR
            gt: (B,C,H,W) Ground truth
            lr: (B,C,H,W) LR上采样 (bicubic)
        """
        # 计算残差 (在RGB通道上平均)
        r_pred = torch.abs(pred - lr).mean(dim=1, keepdim=True)  # (B,1,H,W)
        r_gt = torch.abs(gt - lr).mean(dim=1, keepdim=True)      # (B,1,H,W)
        
        # 归一化残差作为权重
        W = self.normalize_map(r_gt).pow(self.gamma)
        
        # 归一化σ到[0,1]进行对齐
        sigma_norm = self.normalize_map(sigma)
        r_gt_norm = self.normalize_map(r_gt)
        
        # MSE loss: σ应该预测残差模式
        loss = (W * (sigma_norm - r_gt_norm) ** 2).mean()
        
        return loss
    
    def calibration_loss(self, sigma, pred, gt):
        """
        Calibration Loss
        
        思想: σ应该与实际重建误差正相关
        - 使用Pearson相关系数衡量σ和abs_error的关系
        - 鼓励高相关性 (相关系数接近1)
        
        Args:
            sigma: (B,1,H,W) 预测的不确定性
            pred: (B,C,H,W) 网络预测的SR
            gt: (B,C,H,W) Ground truth
        """
        # 计算实际重建误差
        abs_err = torch.abs(pred - gt).mean(dim=1, keepdim=True)  # (B,1,H,W)
        
        B = sigma.size(0)
        corr_loss = 0.0
        
        # 对每个样本计算相关系数
        for i in range(B):
            s = sigma[i].view(-1)
            e = abs_err[i].view(-1).detach()  # detach防止影响主网络
            
            # 采样子集以节省计算
            n = s.numel()
            if n > self.calib_samples:
                idx = torch.randperm(n, device=s.device)[:self.calib_samples]
                s = s[idx]
                e = e[idx]
            
            # 计算Pearson相关系数
            s_centered = s - s.mean()
            e_centered = e - e.mean()
            denom = (s_centered.norm() * e_centered.norm() + 1e-8)
            corr = (s_centered * e_centered).sum() / denom
            
            # 损失: 1 - corr (鼓励corr接近1)
            corr_loss += (1.0 - corr)
        
        corr_loss = corr_loss / B
        return corr_loss
    
    def smoothness_loss(self, sigma):
        """
        Smoothness Loss
        
        思想: σ应该空间平滑(避免噪声)
        - TV loss: 相邻像素差异的L1范数
        
        Args:
            sigma: (B,1,H,W) 预测的不确定性
        """
        # 水平方向
        dx = torch.abs(sigma[:, :, :, :-1] - sigma[:, :, :, 1:]).mean()
        # 垂直方向
        dy = torch.abs(sigma[:, :, :-1, :] - sigma[:, :, 1:, :]).mean()
        
        return dx + dy
    
    def magnitude_loss(self, sigma):
        """
        Magnitude Loss
        
        思想: σ应该在合理范围内
        - 惩罚过大或过小的σ
        - 防止σ均值爆炸
        
        Args:
            sigma: (B,1,H,W) 预测的不确定性
        """
        # 归一化σ (假设输出通过Softplus,范围[0,+∞))
        # 将其映射到相对范围
        sigma_mean = sigma.mean()
        
        # 上界惩罚
        upper_loss = F.relu(sigma - self.sigma_max).pow(2).mean()
        
        # 下界惩罚
        lower_loss = F.relu(self.sigma_min - sigma).pow(2).mean()
        
        # L2正则化防止均值爆炸
        l2_reg = (sigma ** 2).mean()
        
        return upper_loss + lower_loss + 1e-3 * l2_reg
    
    def forward(self, sigma, pred, gt, lr):
        """
        前向计算总损失
        
        Args:
            sigma: (B,1,H,W) U²M预测的不确定性图
            pred: (B,C,H,W) 主网络预测的SR
            gt: (B,C,H,W) Ground truth HR
            lr: (B,C,H,W) LR上采样 (bicubic)
        
        Returns:
            total_loss: 标量
            loss_dict: 各项损失的字典
        """
        losses = {}
        
        # 1. Residual Alignment Loss
        loss_r = self.residual_alignment_loss(sigma, pred, gt, lr)
        losses['res_align'] = loss_r
        
        # 2. Calibration Loss
        loss_c = self.calibration_loss(sigma, pred, gt)
        losses['calib'] = loss_c
        
        # 3. Smoothness Loss
        loss_s = self.smoothness_loss(sigma)
        losses['smooth'] = loss_s
        
        # 4. Magnitude Loss
        loss_m = self.magnitude_loss(sigma)
        losses['mag'] = loss_m
        
        # 总损失
        total_loss = (self.lambda_r * loss_r + 
                     self.lambda_c * loss_c + 
                     self.lambda_s * loss_s + 
                     self.lambda_m * loss_m)
        
        losses['total'] = total_loss
        
        return total_loss, losses


class U2MLossWithHeteroscedastic(nn.Module):
    """
    结合U2MLoss和Heteroscedastic Loss
    
    保留原有的data_term (异方差似然),同时添加新的约束
    """
    
    def __init__(self,
                 lambda_u2m=1.0,        # U2MLoss权重
                 lambda_het=0.1,        # Heteroscedastic data term权重
                 **u2m_kwargs):
        super().__init__()
        self.lambda_u2m = lambda_u2m
        self.lambda_het = lambda_het
        self.u2m_loss = U2MLoss(**u2m_kwargs)
    
    def forward(self, sigma, pred, gt, lr):
        """
        Args:
            sigma: (B,1,H,W) U²M预测的σ
            pred: (B,C,H,W) 主网络预测
            gt: (B,C,H,W) Ground truth
            lr: (B,C,H,W) LR上采样
        """
        # U2M Loss
        u2m_total, u2m_losses = self.u2m_loss(sigma, pred, gt, lr)
        
        # Heteroscedastic data term
        # -log p(gt|pred,σ) = 1/(2σ²)||gt-pred||² + log(σ)
        sigma_clamped = torch.clamp(sigma, min=0.01, max=3.0)
        residual = (gt - pred) ** 2
        data_term = (residual / (2 * sigma_clamped ** 2 + 1e-6)).mean()
        
        # 总损失
        total_loss = self.lambda_u2m * u2m_total + self.lambda_het * data_term
        
        # 构建损失字典
        losses = {f'u2m_{k}': v for k, v in u2m_losses.items()}
        losses['het_data'] = data_term
        losses['total'] = total_loss
        
        return total_loss, losses
