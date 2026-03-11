# 🔬 MLP-AutoResearch

> 基于 [Karpathy's AutoResearch](https://github.com/karpathy/autoresearch) 思想，让 AI Agent 自主迭代优化 MLP 模型

---

## 💡 这是什么？

这是一个将 **AutoResearch 自动实验框架**移植到 **MLP（多层感知机）** 场景的教学项目。

**核心理念**：你不需要手动调参——让 AI Agent（如 Claude、GPT、Gemini）自主循环地修改模型代码、训练、评估、保留最优结果。你只需要"设定规则，然后去睡觉"。

### AutoResearch 原版 vs 本项目

| 维度 | AutoResearch (Karpathy) | MLP-AutoResearch (本项目) |
|------|------------------------|--------------------------|
| 模型类型 | GPT / Transformer | MLP (多层感知机) |
| 数据集 | ClimbMix-400B (语言模型) | MNIST (手写数字分类) |
| 评估指标 | val_bpb (越低越好) | test_accuracy (越高越好) |
| 训练预算 | 固定 5 分钟 | 固定 20 个 epoch |
| 硬件要求 | NVIDIA H100 GPU | CPU 即可运行 |
| 搜索空间 | Transformer 架构/优化器 | MLP 层数/维度/激活函数/初始化/优化器 |

---

## 📐 方法论：三句话概括

1. **分离关注点**：`prepare.py`（只读基础设施）+ `train.py`（AI 可修改）+ `program.md`（人类写的指令）
2. **固定约束、自由探索**：固定数据集、评估指标、训练轮数，AI 在"规则之内"自由创新
3. **贪心迭代循环**：修改代码 → 训练 → 评估 → 好则保留、差则回滚 → 重复

> 📖 详细方法论分析见 [autoresearch_methodology_report.md](./autoresearch_methodology_report.md)

---

## 📂 项目结构

```
MLP-AutoResearch/
├── prepare.py                        # 🔒 基础设施 (只读，AI 不可修改)
│                                     #    - 固定常量、MNIST 数据加载
│                                     #    - 训练循环、评估函数
│                                     #    - 结果摘要输出
│
├── train.py                          # ✏️ 实验代码 (AI 唯一可修改的文件)
│                                     #    - 模型架构定义
│                                     #    - 所有可调超参数
│                                     #    - 优化器配置
│
├── program.md                        # 📋 AI Agent 实验指令
│                                     #    - 实验规则 (能做/不能做)
│                                     #    - 日志格式
│                                     #    - 无限循环流程
│
├── utils.py                          # 🔧 手动实现的算子 (参考)
│                                     #    - MyLinear, MyReLU, MyBatchNorm1d
│
├── autoresearch_methodology_report.md # 📖 AutoResearch 方法论分析报告
│
└── README.md                         # 📄 你正在看的这个文件
```

### 三个文件的角色

| 文件 | 谁来修改 | 类比 |
|------|---------|------|
| `prepare.py` | ❌ 不可修改 | 实验室的仪器设备（固定的评估标准） |
| `train.py` | 🤖 AI Agent | 研究员的实验方案（自由探索） |
| `program.md` | 👤 人类 | 实验室主任下达的研究方向 |

---

## 🚀 快速开始

### 环境要求

- Python >= 3.8
- PyTorch >= 1.10
- torchvision >= 0.11

```bash
pip install torch torchvision
```

### 1. 验证环境

```bash
python prepare.py
```

输出应显示 MNIST 数据加载成功。

### 2. 运行基线实验

```bash
python train.py
```

这会使用默认参数（20 层 MLP，hidden_dim=1024，lr=1e-5）训练 20 个 epoch，输出类似：

```
---
test_accuracy:          0.968300
best_test_accuracy:     0.968800
test_loss:              0.001234
num_params:             2345678
total_time:             45.2
avg_epoch_time:         2.3
model_name:             MLP_20L
```

### 3. 启动 AI 自动迭代

将项目交给你的 AI Agent（Claude Code、Cursor、Copilot 等），发送以下 prompt：

```
请阅读 program.md，按照其中的指令启动自动实验循环。
先运行 baseline，然后开始自主迭代优化。
```

然后你就可以去忙别的事了 🎉

---

## ⚙️ 可调参数（`train.py` 中）

AI Agent 可以修改以下所有参数：

### 模型架构

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NUM_LAYERS` | 20 | 网络层数 |
| `HIDDEN_DIM` | 1024 | 隐藏层维度 |
| `USE_BATCHNORM` | True | 是否使用 Batch Normalization |
| `USE_RESIDUAL` | False | 是否使用残差连接 |
| `RESIDUAL_BLOCK_SIZE` | 5 | 每个残差块的层数 |
| `ACTIVATION` | 'relu' | 激活函数 (relu/gelu/silu/leaky_relu) |
| `INIT_METHOD` | 'kaiming_normal' | 权重初始化策略 |
| `DROPOUT_RATE` | 0.0 | Dropout 比率 |

### 优化器

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `OPTIMIZER_TYPE` | 'adam' | 优化器 (adam/adamw/sgd) |
| `LEARNING_RATE` | 1e-5 | 学习率 |
| `WEIGHT_DECAY` | 0.0 | 权重衰减 |
| `BETAS` | (0.9, 0.999) | Adam betas |
| `LR_SCHEDULE` | 'none' | 学习率调度 (none/cosine/step) |

---

## 🔄 实验循环流程

```
                    ┌──────────────────┐
                    │  读取当前状态     │
                    │  (代码 + 历史)    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  产生实验假设     │
                    │  (修改 train.py)  │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  git commit      │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  python train.py │
                    │  (训练 20 epoch)  │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  提取 test_acc   │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  更好？           │
                    └───┬─────────┬────┘
                    YES │         │ NO
                    ┌───▼───┐ ┌──▼────┐
                    │ keep  │ │discard│
                    │ 保留   │ │ 回滚  │
                    └───┬───┘ └──┬────┘
                        │        │
                        └───┬────┘
                            │
                    ┌───────▼───────┐
                    │  记录 results  │
                    │  .tsv         │
                    └───────┬───────┘
                            │
                            └──→ 回到顶部 (永不停歇)
```

---

## 📊 实验记录格式

实验结果记录在 `results.tsv`（Tab 分隔）：

```
commit	test_accuracy	best_accuracy	num_params	status	description
a1b2c3d	0.968300	0.968800	2345678	keep	baseline (20L, hidden=1024, lr=1e-5)
b2c3d4e	0.973500	0.974200	2345678	keep	switch to AdamW, add weight_decay=0.01
c3d4e5f	0.965100	0.967000	2345678	discard	switch to GELU activation
```

---

## 📖 方法论报告

本项目附带了一份面向教学和分享的 **AutoResearch 方法论完整分析报告**：

→ [autoresearch_methodology_report.md](./autoresearch_methodology_report.md)

报告涵盖：

- 🏗️ **系统级 IPO**：整体架构的输入-处理-输出定义
- 🔄 **实验循环级 IPO**：单次实验的完整流程
- 🧩 **代码模块级 IPO**：`train.py` 内部各组件的职责
- 🎯 **关键设计哲学**：固定时间预算、简约性原则、完全自主运行
- 📊 **与传统方法对比**：AutoResearch vs 手动调参 vs AutoML

---

## 🎓 教学价值

本项目适合以下场景：

1. **学习 AutoResearch 思想**：无需 H100 GPU，用 CPU 即可体验 AI 自动迭代研究
2. **理解 MLP 调参**：通过 AI 自动探索，观察不同超参数对性能的影响
3. **Prompt Engineering 实践**：`program.md` 是如何给 AI "写指令书"的范例
4. **AI Agent 工作流设计**：如何将开放研究问题转化为 AI 可自主执行的闭环流程

---

## 🙏 致谢

- [Andrej Karpathy](https://github.com/karpathy) 的 [AutoResearch](https://github.com/karpathy/autoresearch) 原始项目
- 原始 MLP 实验项目的深度网络消融研究

---

## 📄 许可证

MIT License
