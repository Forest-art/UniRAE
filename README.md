<div align="center">

# Lightning-Hydra RAE Training Framework

[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

基于 PyTorch Lightning 和 Hydra 的 RAE (Reconstruction Autoencoder) 训练框架

</div>

<br>

## 📌 项目简介

本项目是基于 [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template) 开发的 RAE 训练框架，集成了完整的训练、评估和实验管理功能。

### 核心功能

- ✅ **RAE 训练**: 支持 DINO v2 和 SigLIP2 作为编码器训练自编码器
- ✅ **分布式训练**: 原生支持 DDP (Distributed Data Parallel) 多 GPU 训练
- ✅ **rFID 评测**: 训练过程中自动计算重建 FID 分数
- ✅ **Linear Probing**: 评估编码器的表示学习质量
- ✅ **灵活配置**: 基于 Hydra 的配置系统，支持命令行参数覆盖
- ✅ **多种日志**: 支持 TensorBoard、W&B、MLFlow 等多种日志工具
- ✅ **完整文档**: 详细的训练和评估指南

<br>

## 🚀 快速开始

### 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd lightning-hydra-template

# 使用 conda 安装（推荐）
conda env create -f environment.yaml
conda activate rae-train

# 或使用 pip 安装
pip install -r requirements.txt
```

### 小规模测试

在完整训练之前，建议先运行小规模测试确保配置正确：

```bash
# DINO v2 - 使用测试数据集（少量图片）
python src/train.py experiment=rae_dummy

# DINO v2 - 测试 GPU 训练
python src/train.py experiment=rae_dummy trainer=gpu

# SigLIP2 - 使用测试数据集
python src/train.py experiment=rae_siglip_dummy

# SigLIP2 - 测试 GPU 训练
python src/train.py experiment=rae_siglip_dummy trainer=gpu
```

**编码器差异说明**：
- **DINO v2**: 默认图像尺寸 224×224，patch_size=14
- **SigLIP2**: 默认图像尺寸 378×378，patch_size=14
- 两者使用相同的训练参数和评估标准，仅编码器不同

### RAE 训练

#### DINO v2 训练

```bash
# 单 GPU 训练
python src/train.py experiment=rae_dino

# 多 GPU DDP 训练（8 GPU）
python src/train.py experiment=rae_ddp

# 自定义参数训练
python src/train.py experiment=rae_dino \
    data.data_dir=/path/to/imagenet_hf \
    data.batch_size=128 \
    model.optimizer.lr=2e-4
```

#### SigLIP2 训练

```bash
# 单 GPU 训练
python src/train.py experiment=rae_siglip

# 自定义参数训练
python src/train.py experiment=rae_siglip \
    data.data_dir=/path/to/imagenet_hf \
    data.batch_size=128 \
    model.optimizer.lr=2e-4
```

**重要说明**：
- DINO v2 和 SigLIP2 使用完全相同的训练参数（epochs=16, lr=2e-4, global_batch_size=512）
- 两者仅在编码器类型和默认图像尺寸上不同
- rFID 评估、Linear Probing 评估方法完全一致

### Linear Probing 评估

```bash
# 评估训练好的 encoder
python src/eval_linear_probe.py \
    --encoder_checkpoint logs/train/runs/XXXX/checkpoints/last.ckpt \
    --data_dir /path/to/imagenet/val \
    --output_dir logs/linear_probe

# 使用 Hydra 配置训练
python src/train.py experiment=linear_probe \
    model.encoder_checkpoint=logs/train/runs/XXXX/checkpoints/last.ckpt
```

<br>

## 📚 文档

### RAE 训练指南
[**RAE_TRAINING_GUIDE.md**](docs/RAE_TRAINING_GUIDE.md) - 完整的 RAE 训练指南
- 环境配置
- 数据准备（ImageNet 转 HuggingFace 格式）
- 训练命令（单/多 GPU）
- rFID 评测（边训边评）
- 配置说明和常见问题

### Linear Probing 评估指南
[**LINEAR_PROBING_GUIDE.md**](docs/LINEAR_PROBING_GUIDE.md) - Linear Probing 评估指南
- 什么是 Linear Probing
- 评估方法（脚本方式 vs Hydra 训练方式）
- 配置说明（pooling 类型、学习率等）
- 使用示例和结果解读

### 迁移总结
[**MIGRATION_SUMMARY.md**](docs/MIGRATION_SUMMARY.md) - 从原始 RAE 代码迁移到 Lightning-Hydra 框架的总结

<br>

## 🏗️ 项目结构

```
lightning-hydra-template/
├── configs/                   # Hydra 配置文件
│   ├── callbacks/            # Callback 配置
│   │   ├── rae.yaml         # RAE 训练 callbacks
│   │   └── rfid.yaml        # rFID 评估 callback
│   ├── data/                # 数据配置
│   │   └── imagenet.yaml    # ImageNet 数据集配置
│   ├── experiment/          # 实验配置
│   │   ├── rae_dino.yaml        # DINO 训练配置（单 GPU）
│   │   ├── rae_ddp.yaml         # DDP 训练配置（多 GPU）
│   │   ├── rae_dummy.yaml       # DINO dummy 测试配置
│   │   ├── rae_siglip.yaml      # SigLIP 训练配置
│   │   ├── rae_siglip_dummy.yaml # SigLIP dummy 测试配置
│   │   └── linear_probe.yaml    # Linear Probing 配置
│   ├── model/               # 模型配置
│   │   ├── rae.yaml         # RAE 模型配置（DINO v2）
│   │   ├── rae_siglip.yaml  # RAE 模型配置（SigLIP2）
│   │   └── linear_probe.yaml # Linear Probe 配置
│   ├── trainer/             # 训练器配置
│   │   ├── default.yaml     # 默认训练器
│   │   └── ddp.yaml         # DDP 训练器
│   └── ...                  # 其他配置
│
├── src/                      # 源代码
│   ├── models/              # 模型实现
│   │   ├── rae_module.py    # RAE Lightning Module
│   │   ├── linear_probe.py  # Linear Probing 模型
│   │   └── stage1/          # RAE 模型组件
│   │       └── rae.py       # RAE 核心模型
│   ├── data/                # 数据模块
│   │   └── image_folder_datamodule.py
│   ├── callbacks/           # 自定义 callbacks
│   │   └── rfid_callback.py # rFID 评估 callback
│   ├── disc/                # 判别器实现
│   ├── train.py             # 训练脚本
│   ├── eval_rfid.py         # rFID 评估脚本
│   └── eval_linear_probe.py # Linear Probing 评估脚本
│
├── scripts/                  # 工具脚本
│   ├── create_dummy_data.py # 创建测试数据集
│   └── create_hf_test_data.py # 转换为 HuggingFace 格式
│
├── docs/                     # 文档
│   ├── RAE_TRAINING_GUIDE.md
│   ├── LINEAR_PROBING_GUIDE.md
│   └── MIGRATION_SUMMARY.md
│
├── data/                     # 数据目录
│   └── test_hf/             # 测试数据集
│
├── logs/                     # 训练日志（自动生成）
├── environment.yaml          # Conda 环境配置
├── requirements.txt          # Python 依赖
└── README.md                 # 本文件
```

<br>

## ⚙️ 配置说明

### 基本配置覆盖

Hydra 支持通过命令行覆盖任何配置参数：

```bash
# 修改数据路径
python src/train.py experiment=rae_dino data.data_dir=/path/to/imagenet_hf

# 修改图像尺寸
python src/train.py experiment=rae_dino data.image_size=256

# 修改 batch size
python src/train.py experiment=rae_dino data.batch_size=32

# 修改学习率
python src/train.py experiment=rae_dino model.optimizer.lr=1e-4

# 修改训练 epoch 数
python src/train.py experiment=rae_dino trainer.max_epochs=100
```

### 分布式训练

```bash
# 使用 8 个 GPU（默认配置）
python src/train.py experiment=rae_ddp

# 使用 4 个 GPU
python src/train.py experiment=rae_ddp trainer.devices=4 data.batch_size=128

# 使用 torchrun 启动
torchrun --nproc_per_node=8 src/train.py experiment=rae_ddp
```

**batch_size 调整说明**：
- 原始 RAE 配置：global_batch_size=512
- 8 个 GPU：`batch_size=64` (512/8)
- 4 个 GPU：`batch_size=128` (512/4)
- 2 个 GPU：`batch_size=256` (512/2)

### rFID 评测配置

```bash
# 默认每 1000 步和每个 epoch 结束时评测
python src/train.py experiment=rae_dino

# 自定义评测频率
python src/train.py experiment=rae_dino \
    callbacks.rfid.rfid_every_n_steps=500 \
    callbacks.rfid.rfid_every_epoch=true

# 仅在 epoch 结束时评测
python src/train.py experiment=rae_dino \
    callbacks.rfid.rfid_every_n_steps=0

# 修改评测样本数
python src/train.py experiment=rae_dino \
    callbacks.rfid.rfid_num_samples=500
```

<br>

## 📊 监控训练

### TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir logs/runs

# 然后在浏览器中打开 http://localhost:6006
```

查看的指标包括：
- 损失曲线（train/loss, val/loss）
- rFID 分数（rfid/score）
- 学习率
- 生成样本图像（如果启用了样本保存）

### 日志目录

```
logs/
├── runs/
│   └── YYYY-MM-DD/
│       └── HH-MM-SS/
│           ├── .hydra/
│           │   └── config.yaml       # 完整的配置
│           ├── checkpoints/          # 模型检查点
│           └── events.out.tfevents.* # TensorBoard 日志
```

<br>

## 🔧 数据准备

### ImageNet 数据集

1. 下载 ImageNet 数据集（假设数据已下载到 `/path/to/imagenet`）

2. **重要**: ImageNet 数据需要转换为 HuggingFace Dataset 格式

```bash
# 将 ImageNet 转换为 HuggingFace 格式
python scripts/create_hf_test_data.py \
    --input_dir /path/to/imagenet/train \
    --output_dir /path/to/imagenet_hf
```

3. 配置数据路径

```yaml
# configs/data/imagenet.yaml
data_dir: /path/to/imagenet_hf
use_hf_dataset: true
hf_split: train
hf_validation_split: validation
train_split: 1.0  # 使用全部 train 数据训练
```

### 数据集配置说明

数据模块支持两种配置方式：

**方式 1：数据集有独立的 train/validation split（推荐）**
- 训练：使用 train split 的全部数据（train_split=1.0）
- rFID 评测：使用 validation split
- 这是最佳配置，可以准确评估模型的泛化能力

**方式 2：数据集只有 train split（从训练集分割）**
- 训练：使用 train split 的 80%（train_split=0.8）
- rFID 评测：使用 train split 的 20%
- 适用于测试和小规模实验

<br>

## 🧪 调试技巧

### 快速调试

```bash
# 运行 1 个 epoch
python src/train.py experiment=rae_dino debug=default

# 使用少量数据（1 batch）
python src/train.py experiment=rae_dino debug=fdr

# 尝试过拟合到 1 batch
python src/train.py experiment=rae_dino debug=overfit

# 打印执行时间分析
python src/train.py experiment=rae_dino debug=profiler
```

### 恢复训练

```bash
# 从最新的检查点恢复
python src/train.py experiment=rae_dino ckpt_path="last"

# 从指定路径恢复
python src/train.py experiment=rae_dino ckpt_path="/path/to/checkpoint.ckpt"
```

### 常见问题

**内存不足 (OOM)**：
```bash
# 减小 batch size
python src/train.py experiment=rae_dino data.batch_size=16

# 减小图像尺寸
python src/train.py experiment=rae_dino data.image_size=128

# 使用梯度累积
python src/train.py experiment=rae_dino trainer.accumulate_grad_batches=4
```

**数据加载慢**：
```bash
# 增加 num_workers
python src/train.py experiment=rae_dino data.num_workers=8
```

<br>

## 📖 核心技术

### PyTorch Lightning
- 高性能 PyTorch 训练框架
- 自动管理训练循环、混合精度、分布式训练等

### Hydra
- 优雅的配置管理系统
- 支持配置组合和命令行覆盖
- 自动管理实验日志和输出目录

### RAE 模型
- 使用 DINO v2、MAE 或 SigLIP2 作为编码器
- Transformer decoder 重建图像
- 对抗判别器提升重建质量
- EMA 更新机制

### 评估指标
- **rFID**: 重建 FID 分数，衡量重建质量（越低越好）
- **Linear Probing**: 线性分类准确率，衡量表示质量（越高越好）

<br>

## 🎯 训练参数说明

### 通用训练参数（DINO v2 和 SigLIP2 均适用）

| 参数 | 原始值 | 说明 |
|------|--------|------|
| epochs | 16 | 训练轮数 |
| global_batch_size | 512 | 总 batch size |
| num_workers | 8 | 数据加载进程数 |
| lr | 2e-4 | 学习率 |
| betas | [0.9, 0.95] | Adam 优化器参数 |
| ema_decay | 0.9978 | EMA 衰减率 |
| disc_weight | 0.75 | GAN 损失权重 |
| disc_start_epoch | 8 | 开始使用判别器的 epoch |
| sample_every | 2500 | 采样间隔（步数） |

### 编码器差异

| 编码器 | 模型名称 | 图像尺寸 | Patch Size | 配置文件 |
|--------|----------|----------|-----------|----------|
| DINO v2 | facebook/dinov2-with-registers-base | 224 | 16 | configs/model/rae.yaml |
| SigLIP2 | google/siglip-so400m-patch14-384 | 378 | 14 | configs/model/rae_siglip.yaml |

**注意**：DINO v2 和 SigLIP2 使用完全相同的训练参数，仅在编码器类型和图像尺寸上有差异。

<br>

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

在提交之前，请确保：
- 问题在当前 `main` 分支上仍然存在
- Python 依赖已更新到最新版本

<br>

## 📄 许可证

本项目基于 MIT 许可证开源。

<br>

## 🔗 相关链接

- [原始 RAE 仓库](../RAE)
- [Lightning 官方文档](https://pytorch-lightning.readthedocs.io/)
- [Hydra 官方文档](https://hydra.cc/)
- [DINO v2 论文](https://arxiv.org/abs/2304.07193)
- [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template)

<br>

## 💡 使用建议

1. **先测试再训练**: 使用 `experiment=rae_dummy` 和小数据集验证配置
2. **监控 rFID**: 关注 rFID 分数的变化，判断训练是否收敛
3. **调整 batch size**: 根据显存大小调整 batch size，保持 global batch size = batch_size × num_gpus
4. **使用混合精度**: `precision: 16` 可以显著减少显存使用
5. **定期保存样本**: 使用 rFID callback 保存重建样本，直观评估重建质量

<br>

---

<div align="center">

如有问题，请查看详细文档或提交 Issue

</div>