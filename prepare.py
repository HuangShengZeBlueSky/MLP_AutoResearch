"""
MLP AutoResearch 基础设施 (只读文件，AI Agent 不得修改)

职责：
- 固定常量定义
- MNIST 数据加载
- 标准训练循环
- 固定评估函数
- 结果摘要输出

Usage:
    python prepare.py          # 验证数据加载正常
    (被 train.py import 使用)
"""

import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# 固定常量 (不可修改)
# ---------------------------------------------------------------------------

NUM_EPOCHS = 20          # 训练轮数预算
BATCH_SIZE = 64          # 批大小
INPUT_DIM = 784          # MNIST 28x28 展平
OUTPUT_DIM = 10          # 10 个数字类别
DATA_DIR = "./MNIST"     # 数据存储目录

# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def get_device():
    """返回可用的计算设备"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def load_data(batch_size=BATCH_SIZE):
    """
    加载 MNIST 数据集，返回训练集和测试集的 DataLoader。
    
    Returns:
        train_loader, test_loader
    """
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST(
        root=f"{DATA_DIR}/train", train=True, download=True, transform=trans
    )
    test_dataset = datasets.MNIST(
        root=f"{DATA_DIR}/test", train=False, download=True, transform=trans
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# 训练函数
# ---------------------------------------------------------------------------

def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    """
    执行一个 epoch 的训练。
    
    Input:  model, train_loader, optimizer, loss_fn, device
    Process: 遍历 mini-batch → 前向传播 → 计算 loss → 反向传播 → 更新参数
    Output: (avg_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
        total_correct += (pred.argmax(dim=1) == y).sum().item()
        total_samples += X.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# 评估函数 (DO NOT CHANGE — 这是固定的评估指标)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, test_loader, loss_fn, device):
    """
    在测试集上评估模型性能。
    
    Input:  model, test_loader, loss_fn, device
    Process: 遍历测试集 → 前向传播 → 累积 loss 和正确数
    Output: (avg_loss, accuracy)
    
    这是 AutoResearch 的核心指标函数，不得修改。
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        loss = loss_fn(pred, y)
        
        total_loss += loss.item() * X.size(0)
        total_correct += (pred.argmax(dim=1) == y).sum().item()
        total_samples += X.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# 完整训练流程
# ---------------------------------------------------------------------------

def run_training(model, device, lr_scheduler=None):
    """
    执行完整的训练流程。
    
    Input:  model（已在 device 上）, device, 可选的 lr_scheduler
    Process: 加载数据 → 训练 NUM_EPOCHS 轮 → 每轮评估 → 输出摘要
    Output: 最终的 test_accuracy 和 test_loss
    
    返回值: dict 包含所有训练结果
    """
    train_loader, test_loader = load_data()
    loss_fn = nn.CrossEntropyLoss()
    
    # 从 model 获取优化器（train.py 中定义）
    optimizer = model.build_optimizer() if hasattr(model, 'build_optimizer') else None
    if optimizer is None:
        raise ValueError("模型必须实现 build_optimizer() 方法，返回优化器实例")
    
    best_test_accuracy = 0.0
    total_train_time = 0.0
    total_test_time = 0.0
    
    results = {
        'model_name': getattr(model, 'name', model.__class__.__name__),
        'num_params': sum(p.numel() for p in model.parameters()),
        'epochs': {},
    }
    
    print(f"模型: {results['model_name']}")
    print(f"参数量: {results['num_params']:,}")
    print(f"设备: {device}")
    print(f"训练轮数: {NUM_EPOCHS}")
    print()
    
    for epoch in range(NUM_EPOCHS):
        # 训练
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        t1 = time.time()
        train_time = t1 - t0
        
        # 学习率调度
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        # 评估
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
        t2 = time.time()
        test_time = t2 - t1
        
        best_test_accuracy = max(best_test_accuracy, test_acc)
        total_train_time += train_time
        total_test_time += test_time
        
        # 日志
        print(f"epoch {epoch+1:>3}/{NUM_EPOCHS} | "
              f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
              f"test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f} | "
              f"time: {train_time:.1f}s")
        
        results['epochs'][epoch + 1] = {
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
        }
    
    # 最终结果
    final_test_loss, final_test_acc = evaluate(model, test_loader, loss_fn, device)
    
    results.update({
        'final_test_accuracy': final_test_acc,
        'final_test_loss': final_test_loss,
        'best_test_accuracy': best_test_accuracy,
        'avg_train_time_per_epoch': total_train_time / NUM_EPOCHS,
        'avg_test_time_per_epoch': total_test_time / NUM_EPOCHS,
        'total_time': total_train_time + total_test_time,
    })
    
    # 打印标准摘要（供 grep 提取）
    print()
    print("---")
    print(f"test_accuracy:          {final_test_acc:.6f}")
    print(f"best_test_accuracy:     {best_test_accuracy:.6f}")
    print(f"test_loss:              {final_test_loss:.6f}")
    print(f"num_params:             {results['num_params']}")
    print(f"total_time:             {results['total_time']:.1f}")
    print(f"avg_epoch_time:         {results['avg_train_time_per_epoch']:.1f}")
    print(f"model_name:             {results['model_name']}")
    
    return results


# ---------------------------------------------------------------------------
# Main (验证数据加载)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("验证 MNIST 数据加载...")
    train_loader, test_loader = load_data()
    print(f"训练集: {len(train_loader.dataset):,} 样本, {len(train_loader)} 批次")
    print(f"测试集: {len(test_loader.dataset):,} 样本, {len(test_loader)} 批次")
    
    # 检查一个 batch
    X, y = next(iter(train_loader))
    print(f"输入形状: {X.shape} (batch_size={BATCH_SIZE}, 1, 28, 28)")
    print(f"标签形状: {y.shape}")
    print(f"标签范围: {y.min().item()} ~ {y.max().item()}")
    print()
    print("数据加载验证通过！可以运行 train.py 开始实验。")
