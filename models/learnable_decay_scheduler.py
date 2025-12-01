"""
可学习的时间步衰减调度器
用于动态U²M反向去噪的噪声衰减控制
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableDecayScheduler(nn.Module):
    """
    可学习的时间步衰减调度器
    
    输入: 时间步t (归一化到[0,1])
    输出: 衰减因子 decay ∈ [0,1]
    
    设计思路:
    1. 使用小型MLP学习时间步到衰减因子的非线性映射
    2. 初始化接近线性衰减，允许训练中调整
    3. 输出用sigmoid约束到[0,1]
    """
    
    def __init__(self, hidden_dim=32, init_strategy='linear'):
        """
        Args:
            hidden_dim: 隐藏层维度
            init_strategy: 初始化策略
                - 'linear': 初始化为线性衰减 f(t) = t
                - 'exponential': 初始化为指数衰减 f(t) = t²
                - 'learned': 随机初始化，完全由数据驱动
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.init_strategy = init_strategy
        
        # 小型MLP: t -> decay
        # 设计: 2层MLP足够表达各种衰减曲线
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """根据策略初始化权重"""
        if self.init_strategy == 'linear':
            # 初始化为恒等映射: f(t) ≈ t
            # 策略: 第一层放大，第二层缩小，第三层输出
            nn.init.constant_(self.fc1.weight, 1.0)
            nn.init.constant_(self.fc1.bias, 0.0)
            nn.init.constant_(self.fc2.weight, 0.01)  # 小权重，接近恒等
            nn.init.constant_(self.fc2.bias, 0.0)
            nn.init.constant_(self.fc3.weight, 1.0 / self.hidden_dim)
            nn.init.constant_(self.fc3.bias, 0.0)
            
        elif self.init_strategy == 'exponential':
            # 初始化为平方映射: f(t) ≈ t²
            # 通过激活函数的非线性实现
            nn.init.xavier_normal_(self.fc1.weight)
            nn.init.constant_(self.fc1.bias, 0.0)
            nn.init.xavier_normal_(self.fc2.weight)
            nn.init.constant_(self.fc2.bias, 0.5)  # bias偏移实现非线性
            nn.init.xavier_normal_(self.fc3.weight)
            nn.init.constant_(self.fc3.bias, 0.0)
            
        else:  # 'learned'
            # 完全随机初始化
            nn.init.kaiming_normal_(self.fc1.weight)
            nn.init.kaiming_normal_(self.fc2.weight)
            nn.init.kaiming_normal_(self.fc3.weight)
            nn.init.constant_(self.fc1.bias, 0.0)
            nn.init.constant_(self.fc2.bias, 0.0)
            nn.init.constant_(self.fc3.bias, 0.0)
    
    def forward(self, t_normalized):
        """
        Args:
            t_normalized: (B,) 归一化时间步 [0, 1]
                         t=1.0 表示开始去噪（需要强噪声）
                         t=0.0 表示接近完成（需要弱噪声）
        
        Returns:
            decay: (B,) 衰减因子 [0, 1]
        """
        # 扩展维度以匹配Linear层
        x = t_normalized.unsqueeze(-1)  # (B, 1)
        
        # MLP forward
        h1 = F.relu(self.fc1(x))       # (B, hidden_dim)
        h2 = F.relu(self.fc2(h1))      # (B, hidden_dim)
        out = self.fc3(h2)              # (B, 1)
        
        # Sigmoid约束到[0,1]，确保输出合理
        decay = torch.sigmoid(out).squeeze(-1)  # (B,)
        
        return decay
    
    def get_decay_curve(self, num_points=100):
        """
        获取完整的衰减曲线（用于可视化）
        
        Returns:
            t_values: (num_points,) 时间步
            decay_values: (num_points,) 对应的衰减因子
        """
        self.eval()
        with torch.no_grad():
            t_values = torch.linspace(0, 1, num_points)
            decay_values = self.forward(t_values)
        return t_values.cpu().numpy(), decay_values.cpu().numpy()


class SimplifiedDecayScheduler(nn.Module):
    """
    简化版：可学习的参数化衰减调度器
    
    使用少量参数表达常见衰减模式:
    decay(t) = a * t^b + c
    
    其中a, b, c是可学习参数
    """
    
    def __init__(self, init_mode='linear'):
        super().__init__()
        
        # 可学习参数
        if init_mode == 'linear':
            # 线性: t^1
            self.log_power = nn.Parameter(torch.tensor(0.0))  # b=exp(0)=1
            self.scale = nn.Parameter(torch.tensor(1.0))      # a=1
            self.bias = nn.Parameter(torch.tensor(0.0))       # c=0
        elif init_mode == 'exponential':
            # 指数: t^2
            self.log_power = nn.Parameter(torch.tensor(0.693))  # b=exp(0.693)≈2
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.bias = nn.Parameter(torch.tensor(0.0))
        else:  # sqrt
            # 平方根: t^0.5
            self.log_power = nn.Parameter(torch.tensor(-0.693))  # b=exp(-0.693)≈0.5
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.bias = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, t_normalized):
        """
        Args:
            t_normalized: (B,) 归一化时间步 [0, 1]
        
        Returns:
            decay: (B,) 衰减因子，约束到[0, 1]
        """
        # 计算幂指数 (用exp保证b>0)
        power = torch.exp(self.log_power).clamp(0.1, 5.0)  # 限制在[0.1, 5.0]
        
        # 参数化衰减: a * t^b + c
        decay = self.scale * torch.pow(t_normalized.clamp(min=1e-6), power) + self.bias
        
        # 约束到[0, 1]
        decay = torch.clamp(decay, 0.0, 1.0)
        
        return decay
    
    def get_params_summary(self):
        """返回当前参数的可读摘要"""
        power = torch.exp(self.log_power).item()
        return {
            'power_b': power,
            'scale_a': self.scale.item(),
            'bias_c': self.bias.item(),
            'formula': f'{self.scale.item():.3f} * t^{power:.3f} + {self.bias.item():.3f}'
        }


def create_decay_scheduler(scheduler_type='mlp', init_strategy='linear', **kwargs):
    """
    工厂函数：创建衰减调度器
    
    Args:
        scheduler_type: 'mlp' (灵活) 或 'parametric' (简洁)
        init_strategy: 初始化策略
        **kwargs: 传递给调度器的额外参数
    
    Returns:
        scheduler: nn.Module
    """
    if scheduler_type == 'mlp':
        return LearnableDecayScheduler(
            hidden_dim=kwargs.get('hidden_dim', 32),
            init_strategy=init_strategy
        )
    elif scheduler_type == 'parametric':
        return SimplifiedDecayScheduler(init_mode=init_strategy)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


if __name__ == '__main__':
    """测试代码"""
    import matplotlib.pyplot as plt
    
    # 测试不同的初始化策略
    schedulers = {
        'Linear Init': LearnableDecayScheduler(init_strategy='linear'),
        'Exponential Init': LearnableDecayScheduler(init_strategy='exponential'),
        'Learned Init': LearnableDecayScheduler(init_strategy='learned'),
        'Parametric Linear': SimplifiedDecayScheduler(init_mode='linear'),
        'Parametric Exponential': SimplifiedDecayScheduler(init_mode='exponential'),
    }
    
    plt.figure(figsize=(12, 5))
    
    # 测试1: 初始曲线
    plt.subplot(1, 2, 1)
    for name, scheduler in schedulers.items():
        t = torch.linspace(0, 1, 100)
        decay = scheduler(t).detach().numpy()
        plt.plot(t.numpy(), decay, label=name, linewidth=2)
    
    plt.xlabel('Normalized Timestep t', fontsize=12)
    plt.ylabel('Decay Factor', fontsize=12)
    plt.title('Initial Decay Curves', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 测试2: 模拟训练后的变化
    plt.subplot(1, 2, 2)
    scheduler = LearnableDecayScheduler(init_strategy='linear')
    
    # 模拟梯度更新
    optimizer = torch.optim.Adam(scheduler.parameters(), lr=0.01)
    t = torch.linspace(0, 1, 50)
    
    for epoch in [0, 10, 50, 100]:
        if epoch > 0:
            for _ in range(10):
                # 模拟损失：希望早期衰减慢，后期衰减快
                decay = scheduler(t)
                # 假设损失：鼓励t>0.5时decay更大
                loss = -torch.mean(decay[t > 0.5]) + torch.mean(decay[t < 0.5])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        decay = scheduler(t).detach().numpy()
        plt.plot(t.numpy(), decay, label=f'Epoch {epoch}', linewidth=2)
    
    plt.xlabel('Normalized Timestep t', fontsize=12)
    plt.ylabel('Decay Factor', fontsize=12)
    plt.title('Learned Adaptation Example', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learnable_decay_scheduler.png', dpi=150)
    print("Visualization saved to learnable_decay_scheduler.png")
    
    # 打印参数化调度器的参数
    print("\nParametric Scheduler Parameters:")
    param_scheduler = SimplifiedDecayScheduler(init_mode='linear')
    print(param_scheduler.get_params_summary())
