# Linear Probing 评估指南

本文档说明如何使用 Linear Probing 评估训练好的 encoder 的表征质量。

## 目录
- [什么是 Linear Probing](#什么是-linear-probing)
- [快速开始](#快速开始)
- [评估方式](#评估方式)
- [配置说明](#配置说明)
- [使用示例](#使用示例)
- [结果解读](#结果解读)

---

## 什么是 Linear Probing

Linear Probing 是一种评估自监督学习模型表征质量的常用方法。

### 工作原理

1. **冻结 Encoder**：将训练好的 encoder 参数冻结，不进行更新
2. **训练线性分类器**：在 encoder 输出的特征上训练一个简单的线性分类器
3. **评估性能**：通过分类器的准确率来评估 encoder 学习到的表征质量

### 优势

- **简单高效**：只需要训练一个线性层，计算开销小
- **公平比较**：encoder 不参与训练，纯粹评估表征质量
- **标准做法**：视觉表征学习领域的标准评估方法

### 适用场景

- 评估自监督学习模型的表征质量
- 比较不同 encoder 的性能
- 作为下游任务的基准评估

---

## 快速开始

### 方式 1：使用评估脚本（推荐）

使用独立的评估脚本，先提取特征再训练分类器：

```bash
# 评估训练好的 encoder
python src/eval_linear_probe.py \
    --encoder_checkpoint logs/train/runs/2025-02-07/12-34-56/checkpoints/last.ckpt \
    --data_dir /path/to/imagenet/val \
    --output_dir logs/linear_probe

# 使用预训练的 DINO v2 encoder（baseline）
python src/eval_linear_probe.py \
    --encoder_checkpoint null \
    --encoder_config_path facebook/dinov2-with-registers-base \
    --data_dir /path/to/imagenet/val \
    --output_dir logs/linear_probe/dino_baseline
```

### 方式 2：使用 Hydra 配置训练

使用完整的训练流程（包括特征提取和分类器训练）：

```bash
# 使用默认配置训练
python src/train.py experiment=linear_probe

# 指定 encoder checkpoint
python src/train.py experiment=linear_probe \
    model.encoder_checkpoint=logs/train/runs/XXXX-XX-XX/checkpoints/last.ckpt \
    data.data_dir=/path/to/imagenet_hf

# 测试训练（使用少量样本）
python src/train.py experiment=linear_probe \
    model.encoder_checkpoint=logs/train/runs/XXXX-XX-XX/checkpoints/last.ckpt \
    data.data_dir=data/test_imagenet \
    trainer.max_epochs=5
```

---

## 评估方式

### 评估脚本方式

`src/eval_linear_probe.py` 提供了更灵活的评估方式：

**优点**：
- 先提取所有特征，然后快速训练分类器
- 可以在少量样本上快速测试
- 支持不同的 pooling 策略
- 更快的迭代速度

**适用场景**：
- 快速评估 encoder 质量
- 比较不同 encoder
- 小规模测试

### Hydra 训练方式

使用 `python src/train.py experiment=linear_probe`：

**优点**：
- 完整的训练流程
- 自动日志记录和 checkpoint 保存
- 支持恢复训练
- 更适合长时间训练

**适用场景**：
- 正式的评估实验
- 需要完整训练记录
- 大规模数据集评估

---

## 配置说明

### 模型配置

在 `configs/model/linear_probe.yaml` 中配置模型参数：

```yaml
# Encoder 配置
encoder_cls: Dinov2withNorm
encoder_config_path: facebook/dinov2-with-registers-base
encoder_input_size: 224
encoder_checkpoint: null  # 设置为 RAE 训练好的 encoder 路径

# 分类任务配置
num_classes: 1000  # ImageNet-1K 的类别数

# 训练超参数
lr: 1e-3           # 学习率
weight_decay: 0.0  # 权重衰减
max_epochs: 90     # 训练轮数

# 模型设置
freeze_encoder: true       # 冻结 encoder
normalize_features: true   # 归一化特征
pool_type: avg            # 池化方式：avg/cls/flatten
```

### Pooling 类型

支持三种特征池化方式：

| Pooling 类型 | 说明 | 适用场景 |
|-------------|------|----------|
| `avg` | 平均池化 | 默认推荐，综合所有 patch 信息 |
| `cls` | 使用 CLS token | 类似 BERT，适用于全局语义 |
| `flatten` | 展平所有 patch | 保留空间信息，但特征维度大 |

### 数据配置

在 `configs/data/imagenet.yaml` 中配置数据：

```yaml
data_dir: /path/to/imagenet_hf
use_hf_dataset: true
hf_split: train              # 使用训练集训练分类器
hf_validation_split: validation  # 使用验证集评估
train_split: 1.0             # 使用全部训练数据
image_size: 224
batch_size: 256
num_workers: 8
```

---

## 使用示例

### 示例 1：评估训练好的 RAE encoder

```bash
# 使用评估脚本
python src/eval_linear_probe.py \
    --encoder_checkpoint logs/train/runs/2025-02-07/12-34-56/checkpoints/last.ckpt \
    --encoder_config_path facebook/dinov2-with-registers-base \
    --data_dir /path/to/imagenet/val \
    --batch_size 256 \
    --pool_type avg \
    --lr 1e-3 \
    --epochs 90 \
    --output_dir logs/linear_probe/rae_encoder

# 使用 Hydra 训练
python src/train.py experiment=linear_probe \
    model.encoder_checkpoint=logs/train/runs/2025-02-07/12-34-56/checkpoints/last.ckpt \
    data.data_dir=/path/to/imagenet_hf \
    model.num_classes=1000
```

### 示例 2：快速测试（小规模数据）

```bash
# 使用测试数据集快速验证
python src/eval_linear_probe.py \
    --encoder_checkpoint logs/train/runs/2025-02-07/12-34-56/checkpoints/last.ckpt \
    --data_dir data/test_imagenet/val \
    --num_samples 1000 \
    --epochs 10 \
    --output_dir logs/linear_probe/quick_test

# 使用 Hydra 配置快速测试
python src/train.py experiment=linear_probe \
    model.encoder_checkpoint=logs/train/runs/2025-02-07/12-34-56/checkpoints/last.ckpt \
    data.data_dir=data/test_imagenet \
    trainer.max_epochs=5 \
    data.batch_size=32
```

### 示例 3：比较不同 pooling 策略

```bash
# 比较 avg pooling
python src/eval_linear_probe.py \
    --encoder_checkpoint logs/train/runs/2025-02-07/12-34-56/checkpoints/last.ckpt \
    --data_dir /path/to/imagenet/val \
    --pool_type avg \
    --output_dir logs/linear_probe/pooling_avg

# 比较 cls token
python src/eval_linear_probe.py \
    --encoder_checkpoint logs/train/runs/2025-02-07/12-34-56/checkpoints/last.ckpt \
    --data_dir /path/to/imagenet/val \
    --pool_type cls \
    --output_dir logs/linear_probe/pooling_cls
```

### 示例 4：Baseline 对比

```bash
# 预训练 DINO v2 baseline
python src/eval_linear_probe.py \
    --encoder_checkpoint null \
    --encoder_config_path facebook/dinov2-with-registers-base \
    --data_dir /path/to/imagenet/val \
    --output_dir logs/linear_probe/dino_baseline

# RAE 训练后的 encoder
python src/eval_linear_probe.py \
    --encoder_checkpoint logs/train/runs/2025-02-07/12-34-56/checkpoints/last.ckpt \
    --encoder_config_path facebook/dinov2-with-registers-base \
    --data_dir /path/to/imagenet/val \
    --output_dir logs/linear_probe/rae_encoder
```

---

## 结果解读

### 输出示例

```
Using device: cuda
Loading encoder from logs/train/runs/2025-02-07/12-34-56/checkpoints/last.ckpt
Loading dataset from /path/to/imagenet/val
Number of classes: 1000
Dataset size: 50000
Using 50000 samples
Extracting features: 100%|██████████| 196/196 [02:30<00:00,  1.30it/s]
Features shape: torch.Size([50000, 768])
Labels shape: torch.Size([50000])

Training linear probe...
Best validation accuracy: 0.6543

Final evaluation...

Linear Probe Results:
  Number of classes: 1000
  Number of samples: 50000
  Best validation accuracy: 0.6543
  Test accuracy: 0.6532
```

### 结果文件

如果指定了 `--output_dir`，会生成结果文件：

```
logs/linear_probe/rae_encoder/
└── linear_probe_results.txt
```

文件内容：

```
Linear Probe Results
===================

Encoder checkpoint: logs/train/runs/2025-02-07/12-34-56/checkpoints/last.ckpt
Dataset: /path/to/imagenet/val
Number of classes: 1000
Number of samples: 50000
Pooling type: avg
Learning rate: 0.001
Epochs: 90

Best validation accuracy: 0.6543
Test accuracy: 0.6532
```

### 准确率解读

Linear Probing 的准确率反映了 encoder 表征质量：

| 准确率范围 | 表征质量 | 说明 |
|----------|----------|------|
| 70-80% | 优秀 | 接近或超过预训练模型 |
| 60-70% | 良好 | 有用的表征 |
| 50-60% | 一般 | 可以使用，但有待改进 |
| < 50% | 较差 | 表征质量不佳 |

**注意**：
- Linear Probing 在 ImageNet 上的准确率通常低于完整的端到端训练
- 这是因为 encoder 参数被冻结，只训练一个简单的线性层
- 对于自监督学习，Linear Probing 是公平的评估方法

---

## 参数说明

### 评估脚本参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--encoder_checkpoint` | Encoder checkpoint 路径 | null（使用预训练） |
| `--encoder_config_path` | HuggingFace 模型路径 | facebook/dinov2-with-registers-base |
| `--data_dir` | 数据集目录 | 必需 |
| `--image_size` | 图像尺寸 | 224 |
| `--num_samples` | 使用样本数（0=全部） | 0 |
| `--batch_size` | 批大小 | 256 |
| `--num_workers` | 数据加载进程数 | 8 |
| `--pool_type` | 池化类型（avg/cls/flatten） | avg |
| `--lr` | 学习率 | 1e-3 |
| `--epochs` | 训练轮数 | 90 |
| `--seed` | 随机种子 | 42 |
| `--output_dir` | 输出目录 | null |

---

## 常见问题

### 1. 内存不足 (OOM)

```bash
# 减小 batch size
python src/eval_linear_probe.py \
    --encoder_checkpoint logs/train/runs/XXXX/checkpoints/last.ckpt \
    --data_dir /path/to/imagenet/val \
    --batch_size 64

# 使用 CPU（慢但不需要 GPU）
python src/eval_linear_probe.py \
    --encoder_checkpoint logs/train/runs/XXXX/checkpoints/last.ckpt \
    --data_dir /path/to/imagenet/val \
    --num_samples 1000
```

### 2. 加载 checkpoint 失败

确保 checkpoint 路径正确，并且包含 encoder 权重：

```bash
# 检查 checkpoint 内容
python -c "
import torch
ckpt = torch.load('path/to/checkpoint.ckpt', map_location='cpu')
print('Keys:', list(ckpt.keys()))
"

# 如果 checkpoint 只有 encoder 权重
# 可以直接加载；如果包含完整模型，会自动提取 encoder 部分权重
```

### 3. 准确率很低

可能的原因：
- Encoder 训练不充分
- Pooling 策略不适合当前任务
- 数据集不匹配（encoder 训练和评估使用不同的数据）

建议：
- 检查 encoder 训练的 rFID 分数
- 尝试不同的 pooling 策略
- 确保评估数据集和训练数据集一致

### 4. 与预训练模型对比

```bash
# 预训练 DINO v2（作为 baseline）
python src/eval_linear_probe.py \
    --encoder_checkpoint null \
    --encoder_config_path facebook/dinov2-with-registers-base \
    --data_dir /path/to/imagenet/val \
    --output_dir logs/baseline/dino

# 你的 RAE encoder
python src/eval_linear_probe.py \
    --encoder_checkpoint logs/train/runs/XXXX/checkpoints/last.ckpt \
    --encoder_config_path facebook/dinov2-with-registers-base \
    --data_dir /path/to/imagenet/val \
    --output_dir logs/rae/encoder

# 对比结果
cat logs/baseline/dino/linear_probe_results.txt
cat logs/rae/encoder/linear_probe_results.txt
```

---

## 下一步

### 生成任务（待补充）

计划添加类似于 LLaVA 的生成任务评估：

1. **Image Captioning**：图像描述生成
2. **VQA**：视觉问答
3. **Image-Text Retrieval**：图像-文本检索

这些任务将更好地评估 encoder 的生成和理解能力。

---

## 参考文档

- [Linear Probing 论文](https://arxiv.org/abs/1403.6382)
- [DINO v2 论文](https://arxiv.org/abs/2304.07193)
- [RAE 训练指南](./RAE_TRAINING_GUIDE.md)

---

## 支持

如有问题，请检查：
1. Encoder checkpoint 路径是否正确
2. 数据集路径是否正确
3. 数据集格式是否正确（ImageFolder 或 HuggingFace）
4. GPU 内存是否足够