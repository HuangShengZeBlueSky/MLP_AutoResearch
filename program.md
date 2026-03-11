# MLP AutoResearch

这是一个基于 [Karpathy's AutoResearch](https://github.com/karpathy/autoresearch) 思想的 MLP 自动迭代实验框架。

AI Agent 通过自主循环地修改 `train.py` 中的超参数和模型架构，在 MNIST 数据集上自动寻找最优的 MLP 配置。

## Setup

开始实验前，请完成以下准备：

1. **确认 run tag**：基于当前日期提议一个 tag（如 `mar11`），创建分支 `autoresearch/<tag>`。
2. **创建分支**：`git checkout -b autoresearch/<tag>`
3. **阅读所有文件**，理解上下文：
   - `README.md` — 项目背景（MLP 在 MNIST 上的深层网络实验）
   - `prepare.py` — 固定的基础设施：数据加载、训练循环、评估函数。**不可修改**。
   - `train.py` — 你修改的唯一文件。模型架构、超参数、优化器配置。
   - `utils.py` — 手动实现的算子（MyLinear, MyReLU, MyBatchNorm1d），可参考但不在搜索空间内。
4. **验证数据**：运行 `python prepare.py`，确认 MNIST 数据可正常加载。
5. **初始化 results.tsv**：创建 `results.tsv`，只写表头。
6. **确认后开始实验**。

## Experimentation

每次实验运行 `python train.py`，固定训练 **20 个 epoch**。

### 你能做什么：

- 修改 `train.py` — 这是你唯一编辑的文件。可以修改：
  - **模型架构**：层数、隐藏维度、激活函数、是否使用 BatchNorm/Residual/Dropout
  - **优化器**：Adam/AdamW/SGD、学习率、weight decay、betas、momentum
  - **初始化策略**：Kaiming/Xavier/Random
  - **学习率调度**：cosine/step/none
  - **模型结构代码**：可以完全重写 `MLP_AutoResearch` 类的内部实现
  - **新增更复杂的架构变体**：如 Pre-activation 残差块、多尺度特征融合等

### 你不能做什么：

- 修改 `prepare.py`。它是只读的。包含固定的评估函数、数据加载和训练常量。
- 修改 `utils.py`（手动实现的算子，只作为参考）。
- 安装新的包或添加依赖。只能使用已有的 PyTorch + torchvision。
- 修改评估流程。`evaluate()` 函数在 `prepare.py` 中，是权威的评估标准。

### 目标指标

**`test_accuracy`（测试集准确率）— 越高越好。**

由于训练轮数固定为 20 epoch，你不需要担心训练时间，一切可修改的维度都是公平比较的。

### 简约性原则

- 同等效果下，**更简洁的代码更好**
- 增加很多复杂度但只提升 0.1% 的准确率？**可能不值得**
- 删除代码但效果相同或更好？**非常好，保留**
- 微小提升但代码大幅膨胀？**慎重考虑**

## Output format

训练完成后，脚本会打印标准摘要：

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

你可以用以下命令提取核心指标：

```bash
grep "^test_accuracy:\|^best_test_accuracy:\|^num_params:" run.log
```

## Logging results

每次实验完成后，记录到 `results.tsv`（Tab 分隔）：

```
commit	test_accuracy	best_accuracy	num_params	status	description
```

1. git commit hash（7 位短 hash）
2. 最终 test_accuracy（如 0.968300）— 崩溃时用 0.000000
3. 最佳 test_accuracy（训练过程中的最高值）— 崩溃时用 0.000000
4. 参数量 — 崩溃时用 0
5. 状态：`keep`、`discard`、`crash`
6. 简短描述（本次实验做了什么）

示例：

```
commit	test_accuracy	best_accuracy	num_params	status	description
a1b2c3d	0.968300	0.968800	2345678	keep	baseline (20L, hidden=1024, lr=1e-5)
b2c3d4e	0.973500	0.974200	2345678	keep	switch to AdamW, add weight_decay=0.01
c3d4e5f	0.965100	0.967000	2345678	discard	switch to GELU activation
d4e5f6g	0.000000	0.000000	0	crash	increase hidden_dim to 4096 (OOM)
```

## The experiment loop

实验在独立分支上运行（如 `autoresearch/mar11`）。

**LOOP FOREVER:**

1. 查看当前 git 状态（当前分支/commit）
2. 产生实验假设，修改 `train.py` 中的超参数或模型架构
3. `git commit -am "描述修改内容"`
4. 运行实验：`python train.py > run.log 2>&1`（重定向输出，不要让输出淹没你的上下文）
5. 提取结果：`grep "^test_accuracy:\|^best_test_accuracy:\|^num_params:" run.log`
6. 如果 grep 输出为空，说明运行崩溃了。用 `tail -n 50 run.log` 查看错误栈
7. 记录结果到 `results.tsv`（**不要** git commit 这个文件）
8. 如果 test_accuracy 提升（更高）→ 保留 commit，分支向前推进
9. 如果 test_accuracy 持平或下降 → `git reset --hard HEAD~1` 回滚

### 实验方向建议

以下是一些值得探索的方向（不限于此）：

1. **学习率**：当前 1e-5，可以尝试不同量级
2. **优化器**：Adam vs AdamW vs SGD with momentum
3. **层数**：5L vs 10L vs 20L vs 更深
4. **残差连接**：开启残差后的效果
5. **激活函数**：ReLU vs GELU vs SiLU
6. **BatchNorm 开/关**
7. **Dropout**：加入正则化
8. **隐藏维度**：512 vs 1024 vs 2048
9. **学习率调度**：cosine annealing
10. **权重初始化**：Kaiming vs Xavier
11. **架构创新**：如 bottleneck 结构、金字塔结构等

### 重要提示

- **第一次运行**：始终先跑 baseline（不修改任何参数），记录基准结果
- **NEVER STOP**：一旦实验循环开始，不要暂停询问人类。人类可能不在电脑前。完全自主运行，直到被手动中断
- **崩溃处理**：如果是简单 bug（拼写错误、import 遗漏），修复后重跑。如果想法本身有根本问题，放弃并记录 crash
- **超时**：每次实验正常应在 1-3 分钟内完成。如果超过 5 分钟，kill 掉并视为失败
