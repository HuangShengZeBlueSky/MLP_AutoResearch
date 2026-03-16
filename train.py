"""
MLP AutoResearch 训练脚本 (AI Agent 可修改的唯一文件)

这是 AI Agent 在自动迭代实验中唯一可以修改的文件。
包含：模型架构定义、超参数配置、优化器设置。

Usage: python train.py
"""

import torch
from torch import nn, optim
import torch.nn.functional as F

from prepare import (
    INPUT_DIM, OUTPUT_DIM, NUM_EPOCHS,
    get_device, run_training
)

# ---------------------------------------------------------------------------
# 超参数 (AI Agent 可自由修改这些参数)
# ---------------------------------------------------------------------------

# 模型架构
NUM_LAYERS = 3              # 网络层数（不含输入/输出层）
HIDDEN_DIM = 256            # 隐藏层维度
USE_BATCHNORM = True         # 是否使用 Batch Normalization
USE_RESIDUAL = False         # 是否使用残差连接
RESIDUAL_BLOCK_SIZE = 5      # 每个残差块包含的层数
ACTIVATION = 'gelu'          # 激活函数: 'relu', 'gelu', 'silu', 'leaky_relu'
INIT_METHOD = 'kaiming_normal'  # 权重初始化: 'kaiming_normal', 'kaiming_uniform',
                                #              'xavier_normal', 'xavier_uniform', 'default'
DROPOUT_RATE = 0.1           # Dropout 比率 (0.0 = 不使用)

# 优化器
OPTIMIZER_TYPE = 'adamw'       # 优化器: 'adam', 'adamw', 'sgd'
LEARNING_RATE = 1e-3         # 学习率
WEIGHT_DECAY = 1e-4          # 权重衰减 (L2 正则化)
BETAS = (0.9, 0.999)         # Adam/AdamW 的 beta1, beta2
MOMENTUM = 0.9               # SGD 的动量
LR_SCHEDULE = 'cosine'       # 学习率调度: 'none', 'cosine', 'step'
LR_STEP_SIZE = 10            # StepLR 的 step_size
LR_GAMMA = 0.1               # StepLR 的 gamma

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def get_activation():
    """根据 ACTIVATION 参数返回对应的激活函数模块"""
    activations = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'silu': nn.SiLU,
        'leaky_relu': lambda: nn.LeakyReLU(0.01),
    }
    act_cls = activations.get(ACTIVATION, nn.ReLU)
    return act_cls()


def init_weights(model):
    """根据 INIT_METHOD 参数初始化模型权重"""
    init_fns = {
        'kaiming_normal': lambda w: nn.init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu'),
        'kaiming_uniform': lambda w: nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu'),
        'xavier_normal': lambda w: nn.init.xavier_normal_(w),
        'xavier_uniform': lambda w: nn.init.xavier_uniform_(w),
    }
    
    init_fn = init_fns.get(INIT_METHOD)
    if init_fn is None:
        return  # 'default' 或未知方法，使用 PyTorch 默认初始化
    
    for module in model.modules():
        if isinstance(module, nn.Linear):
            init_fn(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# 模型定义
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """残差块：主路径 + 线性跳跃连接"""
    
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers):
        super().__init__()
        
        # 主路径
        layers = []
        current_dim = in_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if USE_BATCHNORM:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(get_activation())
            if DROPOUT_RATE > 0:
                layers.append(nn.Dropout(DROPOUT_RATE))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, out_dim))
        if USE_BATCHNORM:
            layers.append(nn.BatchNorm1d(out_dim))
        self.main_path = nn.Sequential(*layers)
        
        # 跳跃连接（线性投影）
        self.skip = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim) if USE_BATCHNORM else nn.Identity()
        )
    
    def forward(self, x):
        return self.main_path(x) + self.skip(x)


class MLP_AutoResearch(nn.Module):
    """
    可配置的 MLP 模型，支持：
    - 任意层数
    - 可选 BatchNorm
    - 可选残差连接
    - 可选 Dropout
    - 多种激活函数
    - 多种初始化策略
    """
    
    def __init__(self):
        super().__init__()
        self.name = f"MLP_{NUM_LAYERS}L"
        if USE_RESIDUAL:
            self.name += "_Residual"
        
        self.flatten = nn.Flatten()
        
        if USE_RESIDUAL:
            self._build_residual_model()
        else:
            self._build_plain_model()
        
        self.classify = nn.Softmax(dim=1)
    
    def _build_plain_model(self):
        """构建朴素 MLP（无残差连接）"""
        layers = []
        current_dim = INPUT_DIM
        
        for i in range(NUM_LAYERS):
            next_dim = HIDDEN_DIM
            layers.append(nn.Linear(current_dim, next_dim))
            if USE_BATCHNORM:
                layers.append(nn.BatchNorm1d(next_dim))
            layers.append(get_activation())
            if DROPOUT_RATE > 0:
                layers.append(nn.Dropout(DROPOUT_RATE))
            current_dim = next_dim
        
        # 输出层
        layers.append(nn.Linear(current_dim, OUTPUT_DIM))
        self.mlp = nn.Sequential(*layers)
    
    def _build_residual_model(self):
        """构建残差 MLP"""
        blocks = []
        num_blocks = max(1, NUM_LAYERS // RESIDUAL_BLOCK_SIZE)
        layers_per_block = max(2, NUM_LAYERS // num_blocks)
        
        current_dim = INPUT_DIM
        for i in range(num_blocks):
            out_dim = HIDDEN_DIM
            blocks.append(ResidualBlock(current_dim, out_dim, HIDDEN_DIM, layers_per_block))
            blocks.append(get_activation())
            if DROPOUT_RATE > 0:
                blocks.append(nn.Dropout(DROPOUT_RATE))
            current_dim = out_dim
        
        # 输出层
        blocks.append(nn.Linear(current_dim, OUTPUT_DIM))
        self.mlp = nn.Sequential(*blocks)
    
    def build_optimizer(self):
        """
        构建优化器（由 prepare.py 的 run_training 调用）。
        AI Agent 可修改此方法来尝试不同的优化器配置。
        """
        if OPTIMIZER_TYPE == 'adam':
            optimizer = optim.Adam(
                self.parameters(), lr=LEARNING_RATE,
                betas=BETAS, weight_decay=WEIGHT_DECAY
            )
        elif OPTIMIZER_TYPE == 'adamw':
            optimizer = optim.AdamW(
                self.parameters(), lr=LEARNING_RATE,
                betas=BETAS, weight_decay=WEIGHT_DECAY
            )
        elif OPTIMIZER_TYPE == 'sgd':
            optimizer = optim.SGD(
                self.parameters(), lr=LEARNING_RATE,
                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
            )
        else:
            optimizer = optim.Adam(
                self.parameters(), lr=LEARNING_RATE,
                betas=BETAS, weight_decay=WEIGHT_DECAY
            )
        
        return optimizer
    
    def forward(self, input):
        X = self.flatten(input)
        logits = self.classify(self.mlp(X))
        return logits


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = get_device()
    print(f"设备: {device}")
    
    # 构建模型
    model = MLP_AutoResearch().to(device)
    
    # 初始化权重
    init_weights(model)
    
    # 打印超参数摘要
    print(f"\n=== 超参数配置 ===")
    print(f"层数:         {NUM_LAYERS}")
    print(f"隐藏维度:     {HIDDEN_DIM}")
    print(f"BatchNorm:    {USE_BATCHNORM}")
    print(f"残差连接:     {USE_RESIDUAL}")
    print(f"激活函数:     {ACTIVATION}")
    print(f"初始化:       {INIT_METHOD}")
    print(f"Dropout:      {DROPOUT_RATE}")
    print(f"优化器:       {OPTIMIZER_TYPE}")
    print(f"学习率:       {LEARNING_RATE}")
    print(f"权重衰减:     {WEIGHT_DECAY}")
    print(f"LR调度:       {LR_SCHEDULE}")
    print()
    
    # 构建学习率调度器
    lr_scheduler = None
    if LR_SCHEDULE != 'none':
        optimizer_for_scheduler = model.build_optimizer()
        if LR_SCHEDULE == 'cosine':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer_for_scheduler, T_max=NUM_EPOCHS
            )
        elif LR_SCHEDULE == 'step':
            lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer_for_scheduler, step_size=LR_STEP_SIZE, gamma=LR_GAMMA
            )
    
    # 运行训练
    results = run_training(model, device, lr_scheduler=lr_scheduler)
