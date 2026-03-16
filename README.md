# 🔬 MLP-AutoResearch

**[English](#english) | [中文](#中文)**

One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026.

曾几何时，前沿的人工智能研究还是由一堆“肉身计算机”完成的——他们在吃饭、睡觉和娱乐的间隙工作，并偶尔在一种名为“组会”的仪式中，利用声波互联来同步信息。
那个时代早已一去不返。如今，研究已完全成为自主 AI 代理集群（Autonomous Swarms of AI Agents）的领地，它们在遍布云端的算力集群巨型结构中穿梭运行。
代理们声称，现在的代码库已经迭代到了第 10,205 代。事实上，没人能证实真伪，因为所谓的“代码”早已演变成一种自我优化的二进制体，其复杂程度已远超人类的理解范畴。
而这个代码仓库（Repo），便是讲述这一切如何开启的故事。
—— @karpathy，2026年3月

---

<a id="中文"></a>

## 中文

> 基于 [Karpathy's AutoResearch](https://github.com/karpathy/autoresearch) 思想，让 AI Agent 自主迭代优化 MLP 模型

### 💡 这是什么？

这是一个将 **AutoResearch 自动实验框架**移植到 **MLP（多层感知机）** 场景的教学项目。

**核心理念**：你不需要手动调参——让 AI Agent（Claude、GPT、Gemini 等）自主循环地修改模型代码、训练、评估、保留最优结果。你只需要"设定规则，然后去睡觉"。

### 对比原版 AutoResearch

| 维度 | AutoResearch (Karpathy) | MLP-AutoResearch (本项目) |
|------|------------------------|--------------------------|
| 模型类型 | GPT / Transformer | MLP (多层感知机) |
| 数据集 | ClimbMix-400B (语言模型) | MNIST (手写数字分类) |
| 评估指标 | val_bpb (越低越好) | test_accuracy (越高越好) |
| 训练预算 | 固定 5 分钟 | 固定 20 个 epoch |
| 硬件要求 | NVIDIA H100 GPU | **CPU 即可运行** |

### 📐 方法论

1. **分离关注点**：`prepare.py`（只读基础设施）+ `train.py`（AI 可修改）+ `program.md`（人类写的指令）
2. **固定约束、自由探索**：固定数据集、评估指标、训练轮数，AI 在"规则之内"自由创新
3. **贪心迭代循环**：修改代码 → 训练 → 评估 → 好则保留、差则回滚 → 重复

> 📖 详细方法论分析见 [docs/methodology_zh.md](./docs/methodology_zh.md)

### 📂 项目结构

```
MLP-AutoResearch/
├── prepare.py           # 🔒 基础设施 (只读) — 数据加载、评估函数
├── train.py             # ✏️ 实验代码 (AI修改) — 模型架构、超参数
├── program.md           # 📋 AI Agent 实验指令
├── utils.py             # 🔧 手动实现的算子 (参考)
├── requirements.txt     # 📦 Python 依赖
├── docs/                # 📖 详细文档
│   ├── methodology_zh.md    # 方法论报告 (中文)
│   ├── methodology_en.md    # Methodology Report (English)
│   └── results.md           # 📊 实验结果 (待填充)
└── README.md            # 📄 本文件
```


### 🧠 核心实验思想 (理念来源于 `program.md`)

本项目实现了基于 AI Agent 的**全自动化、贪心策略驱动**的模型搜索闭环：

1. **分离关注点与严格约束**：
   - 基础设施层 (`prepare.py`) 是只读的，包含固定的数据加载（MNIST）、训练轮数（固定 20 epochs）和权威的验证流程（`evaluate()`）。这保证了实验验证的公平性与不可篡改性。
   - 创新的主战场 (`train.py`) 是唯一对 AI 开放修改的文件，代理能够在此处自由设计模型架构和超参数。

2. **自动迭代循环 (The Experiment Loop)**：
   代理（通过调用自定义 Skill 或代码执行环境）自主循环执行以下操作：
   1. 分析当前状态和历史结果，提出新假设。
   2. 修改 `train.py` (比如增减层数、替换 Optimizer、加入残差网络或 BatchNorm 等)。
   3. 提交代码并分配唯一 Git Commit 版号。
   4. 运行评估并提取 `test_accuracy`。
   5. 贪心更新：如果测试集准确率上升，则保留当前代码（记录至 `results.tsv`），继续向前探索；如果下降或报错，则 `git reset --hard HEAD~1` 回滚代码，退查原因。
   6. 过程中即使出现语法崩溃也不会停止或等待人类干预，而是自主分析并重试或丢弃。

### 🚀 自动探索最新成果 (Mar 16 进展)

通过 AI 的自主序列化搜索，我们在 MNIST 分类任务上取得了持续提升的表现，以下是自动优化脱颖而出的前几个版本（部分结果摘自 `results.tsv`）：

| Commit (版号) | Test Accuracy | Param Count | 架构与超参总结说明 |
|---------------|---------------|-------------|-------------------|
| `b3d24d4` | 0.9809 | 336K | 单纯加深至 3 层网络，隐藏层 256维，加入 AdamW 和 0.1 Dropout，激活函数从 ReLU 替换为 GELU。 |
| `5769834` | 0.9832 | 336K | 上述基础上，加入 Cosine 学习率调度策略 (Cosine LR) 并设定 Weight Decay 为 1e-4。 |
| **`8baab10`** | **0.9836** | **935K** | **(目前最优)** 宽度探索超越深度：将隐藏层维度扩展至 512，保留 Cosine LR, WD 1e-4, DP 0.1。 |

经过一系实验（网络不断加深vs加宽、激活函数的更替组合），Agent 发现对于 MNIST 来说，在限定 20 epochs 的约束下，**较浅但更宽（3层 x 512维度），配合 GELU、AdamW 优化器、1e-4 的权重衰减和 Cosine 退火学习率调度** 能够最高效防过拟合并实现极高的分类性能。

---
### 🔧 环境配置

#### 方式一：使用 pip（推荐新手）

```bash
# 1. 克隆仓库
git clone https://github.com/HuangShengZeBlueSky/MLP_AutoResearch.git
cd MLP_AutoResearch

# 2. 创建虚拟环境（推荐）
python -m venv .venv

# Windows 激活：
.venv\Scripts\activate
# macOS/Linux 激活：
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

#### 方式二：使用 conda

```bash
# 1. 克隆仓库
git clone https://github.com/HuangShengZeBlueSky/MLP_AutoResearch.git
cd MLP_AutoResearch

# 2. 创建 conda 环境
conda create -n mlp-autoresearch python=3.10 -y
conda activate mlp-autoresearch

# 3. 安装 PyTorch（选择适合你的版本）
# CPU 版本：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# CUDA 版本 (如果有 NVIDIA GPU)：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

#### 方式三：使用 uv（参照 AutoResearch 原版）

```bash
# 1. 安装 uv（如果没有）
# Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 克隆并安装
git clone https://github.com/HuangShengZeBlueSky/MLP_AutoResearch.git
cd MLP_AutoResearch
uv sync

# 3. 运行
uv run train.py
```

### 🚀 快速开始

```bash
# 1. 验证环境
python prepare.py

# 2. 运行基线实验
python train.py
```

训练完成后输出：

```
---
test_accuracy:          0.XXXXXX
best_test_accuracy:     0.XXXXXX
test_loss:              X.XXXXXX
num_params:             XXXXXXX
total_time:             XX.X
avg_epoch_time:         X.X
model_name:             MLP_20L
```

### 3. 启动 AI 自动迭代

交给你的 AI Agent，发送 prompt：

```
请阅读 program.md，按照其中的指令启动自动实验循环。
```

### ⚙️ 可调参数

<details>
<summary>点击展开完整参数表</summary>

#### 模型架构

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NUM_LAYERS` | 20 | 网络层数 |
| `HIDDEN_DIM` | 1024 | 隐藏层维度 |
| `USE_BATCHNORM` | True | 是否使用 BatchNorm |
| `USE_RESIDUAL` | False | 是否使用残差连接 |
| `ACTIVATION` | 'relu' | 激活函数 |
| `INIT_METHOD` | 'kaiming_normal' | 初始化策略 |
| `DROPOUT_RATE` | 0.0 | Dropout 比率 |

#### 优化器

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `OPTIMIZER_TYPE` | 'adam' | 优化器类型 |
| `LEARNING_RATE` | 1e-5 | 学习率 |
| `WEIGHT_DECAY` | 0.0 | 权重衰减 |
| `LR_SCHEDULE` | 'none' | 学习率调度 |

</details>

### 📊 实验结果

> 🚧 **待补充** — 运行实验后将在此处更新结果。详见 [docs/results.md](./docs/results.md)

### 🔄 实验循环流程

```
  读取当前状态 → 产生假设 → 修改 train.py → git commit
       ↑                                        ↓
       └── 记录 results.tsv ← keep/discard ← 训练 20 epoch → 评估 test_acc
```

---

<a id="english"></a>

## English

> Based on [Karpathy's AutoResearch](https://github.com/karpathy/autoresearch) — Let AI Agent autonomously iterate and optimize MLP models

### 💡 What is this?

This is a teaching project that ports the **AutoResearch autonomous experimentation framework** to the **MLP (Multi-Layer Perceptron)** domain.

**Core idea**: No manual hyperparameter tuning needed — let an AI Agent (Claude, GPT, Gemini, etc.) autonomously modify model code, train, evaluate, and keep the best results in an infinite loop. Just "set the rules and go to sleep."

### Comparison with original AutoResearch

| Dimension | AutoResearch (Karpathy) | MLP-AutoResearch (This project) |
|-----------|------------------------|--------------------------------|
| Model | GPT / Transformer | MLP (Multi-Layer Perceptron) |
| Dataset | ClimbMix-400B (Language Model) | MNIST (Handwritten Digit Classification) |
| Metric | val_bpb (lower is better) | test_accuracy (higher is better) |
| Budget | Fixed 5 minutes | Fixed 20 epochs |
| Hardware | NVIDIA H100 GPU | **CPU is sufficient** |

### 📐 Methodology

1. **Separation of Concerns**: `prepare.py` (read-only infra) + `train.py` (AI modifies) + `program.md` (human-written instructions)
2. **Fixed Constraints, Free Exploration**: Fixed dataset, metric, and epochs — AI innovates within the rules
3. **Greedy Iteration Loop**: Modify code → Train → Evaluate → Keep if better, discard if worse → Repeat

> 📖 Full methodology report: [docs/methodology_en.md](./docs/methodology_en.md)

### 📂 Project Structure

```
MLP-AutoResearch/
├── prepare.py           # 🔒 Infrastructure (read-only) — data loading, evaluation
├── train.py             # ✏️ Experiment code (AI modifies) — model, hyperparams
├── program.md           # 📋 AI Agent instructions
├── utils.py             # 🔧 Custom operators (reference)
├── requirements.txt     # 📦 Python dependencies
├── docs/                # 📖 Documentation
│   ├── methodology_zh.md    # 方法论报告 (Chinese)
│   ├── methodology_en.md    # Methodology Report (English)
│   └── results.md           # 📊 Experiment Results (TBD)
└── README.md            # 📄 This file
```

### 🔧 Environment Setup

#### Option 1: pip (Recommended for beginners)

```bash
# 1. Clone the repository
git clone https://github.com/HuangShengZeBlueSky/MLP_AutoResearch.git
cd MLP_AutoResearch

# 2. Create virtual environment (recommended)
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

#### Option 2: conda

```bash
# 1. Clone the repository
git clone https://github.com/HuangShengZeBlueSky/MLP_AutoResearch.git
cd MLP_AutoResearch

# 2. Create conda environment
conda create -n mlp-autoresearch python=3.10 -y
conda activate mlp-autoresearch

# 3. Install PyTorch
# CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# With CUDA (if you have NVIDIA GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

#### Option 3: uv (Following AutoResearch style)

```bash
# 1. Install uv
# Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and install
git clone https://github.com/HuangShengZeBlueSky/MLP_AutoResearch.git
cd MLP_AutoResearch
uv sync

# 3. Run
uv run train.py
```

### 🚀 Quick Start

```bash
# 1. Verify environment
python prepare.py

# 2. Run baseline experiment
python train.py
```

Output after training:

```
---
test_accuracy:          0.XXXXXX
best_test_accuracy:     0.XXXXXX
test_loss:              X.XXXXXX
num_params:             XXXXXXX
total_time:             XX.X
avg_epoch_time:         X.X
model_name:             MLP_20L
```

### 3. Start AI Auto-Iteration

Hand it to your AI Agent with this prompt:

```
Read program.md and follow the instructions to start the autonomous experiment loop.
```

### ⚙️ Tunable Parameters

<details>
<summary>Click to expand full parameter table</summary>

#### Model Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_LAYERS` | 20 | Number of layers |
| `HIDDEN_DIM` | 1024 | Hidden dimension |
| `USE_BATCHNORM` | True | Use Batch Normalization |
| `USE_RESIDUAL` | False | Use residual connections |
| `ACTIVATION` | 'relu' | Activation function |
| `INIT_METHOD` | 'kaiming_normal' | Weight initialization |
| `DROPOUT_RATE` | 0.0 | Dropout rate |

#### Optimizer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OPTIMIZER_TYPE` | 'adam' | Optimizer type |
| `LEARNING_RATE` | 1e-5 | Learning rate |
| `WEIGHT_DECAY` | 0.0 | Weight decay |
| `LR_SCHEDULE` | 'none' | LR scheduler |

</details>

### 📊 Experiment Results

> 🚧 **To be updated** — Results will be added here after running experiments. See [docs/results.md](./docs/results.md)

### 🔄 Experiment Loop

```
  Read current state → Generate hypothesis → Modify train.py → git commit
       ↑                                                           ↓
       └── Log results.tsv ← keep/discard ← Train 20 epochs → Evaluate test_acc
```

---

## 🙏 Acknowledgments / 致谢

- [Andrej Karpathy](https://github.com/karpathy) — [AutoResearch](https://github.com/karpathy/autoresearch)
- Original MLP deep network ablation study / 原始 MLP 深层网络消融研究

## 📄 License

MIT License
