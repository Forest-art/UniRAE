# DiT (Diffusion Transformer) 训练指南

本指南介绍如何使用 Lightning-Hydra 框架训练 DiT (Diffusion Transformer) 模型，这是 RAE 项目的第二阶段（Stage 2）。

## 📋 目录

1. [概述](#概述)
2. [准备工作](#准备工作)
3. [快速开始](#快速开始)
4. [训练 DiT 模型](#训练-dit-模型)
5. [采样生成](#采样生成)
6. [配置说明](#配置说明)
7. [常见问题](#常见问题)

---

## 概述

### 什么是 DiT？

DiT (Diffusion Transformer) 是一种基于 Transformer 的扩散模型，用于高质量图像生成。在本框架中，DiT 模型在 RAE (Reconstruction Autoencoder) 的潜空间中进行训练，实现高效的高分辨率图像生成。

### 训练流程

```
Stage 1 (RAE): 训练自编码器
    → 图像 → RAE Encoder → 潜在表示 (Latent)
    → 潜在表示 → RAE Decoder → 重建图像

Stage 2 (DiT): 训练扩散模型
    → 随机噪声 → DiT 模型 → 潜在表示
    → 潜在表示 → RAE Decoder → 生成图像
```

### 关键特性

- ✅ **潜空间训练**: 在 RAE 的潜空间中训练，显著降低计算成本
- ✅ **分类器无关引导 (CFG)**: 支持条件生成和高质量的图像生成
- ✅ **EMA 更新**: 使用指数移动平均提高生成质量
- ✅ **灵活配置**: 基于 Hydra 的配置系统，易于实验
- ✅ **分布式训练**: 支持 DDP 多 GPU 训练

---

## 准备工作

### 1. 先决条件

在训练 DiT 模型之前，需要：

- ✅ 完成 RAE (Stage 1) 训练
- ✅ 获得 RAE 检查点文件
- ✅ 准备好训练数据集（ImageNet 或其他）

### 2. 安装依赖

```bash
# 确保已安装所有依赖
pip install -r requirements.txt

# 或使用 conda
conda env create -f environment.yaml
conda activate rae-train
```

### 3. 准备 RAE 检查点

确保你已经训练了 RAE 模型并获得检查点：

```bash
# RAE 检查点通常位于
logs/train/runs/YYYY-MM-DD/HH-MM-SS/checkpoints/
├── last.ckpt
├── epoch=X-step=Y.ckpt
└── ...
```

---

## 快速开始

### 小规模测试

在完整训练之前，建议先运行小规模测试：

```bash
# CPU 测试（验证代码）
python src/train.py experiment=dit_dummy

# GPU 测试（验证 GPU 支持）
python src/train.py experiment=dit_dummy trainer=gpu
```

### 完整训练

```bash
# 使用预训练的 RAE 检查点训练 DiT
python src/train.py experiment=dit_dino \
    model.rae.encoder_checkpoint_path=/path/to/rae_checkpoint.ckpt

# 自定义训练参数
python src/train.py experiment=dit_dino \
    model.rae.encoder_checkpoint_path=/path/to/rae_checkpoint.ckpt \
    data.data_dir=/path/to/imagenet_hf \
    data.batch_size=32 \
    model.dit_module.learning_rate=1e-4
```

---

## 训练 DiT 模型

### 单 GPU 训练

```bash
# 基础训练命令
python src/train.py experiment=dit_dino \
    model.rae.encoder_checkpoint_path=/path/to/rae_checkpoint.ckpt

# 使用较小的 batch size
python src/train.py experiment=dit_dino \
    model.rae.encoder_checkpoint_path=/path/to/rae_checkpoint.ckpt \
    data.batch_size=16
```

### 多 GPU DDP 训练

```bash
# 使用 8 个 GPU（推荐配置）
python src/train.py experiment=dit_dino \
    model.rae.encoder_checkpoint_path=/path/to/rae_checkpoint.ckpt \
    trainer=ddp

# 使用 4 个 GPU
python src/train.py experiment=dit_dino \
    model.rae.encoder_checkpoint_path=/path/to/rae_checkpoint.ckpt \
    trainer=ddp \
    trainer.devices=4 \
    data.batch_size=64

# 使用 torchrun 启动
torchrun --nproc_per_node=8 src/train.py experiment=dit_dino \
    model.rae.encoder_checkpoint_path=/path/to/rae_checkpoint.ckpt
```

### 恢复训练

```bash
# 从最新检查点恢复
python src/train.py experiment=dit_dino \
    model.rae.encoder_checkpoint_path=/path/to/rae_checkpoint.ckpt \
    ckpt_path="last"

# 从指定检查点恢复
python src/train.py experiment=dit_dino \
    model.rae.encoder_checkpoint_path=/path/to/rae_checkpoint.ckpt \
    ckpt_path="/path/to/dit_checkpoint.ckpt"
```

---

## 采样生成

### 基础采样

```bash
# 生成 100 张随机图像
python src/eval_dit.py \
    --checkpoint /path/to/dit_checkpoint.ckpt \
    --num_samples 100 \
    --output_dir outputs/samples

# 使用分类器无关引导 (CFG)
python src/eval_dit.py \
    --checkpoint /path/to/dit_checkpoint.ckpt \
    --num_samples 100 \
    --cfg_scale 2.0 \
    --output_dir outputs/samples_cfg2

# 增加采样步数（提高质量）
python src/eval_dit.py \
    --checkpoint /path/to/dit_checkpoint.ckpt \
    --num_samples 100 \
    --num_steps 200 \
    --output_dir outputs/samples_200steps
```

### 条件生成（指定类别）

```bash
# 生成特定类别的图像
python src/eval_dit.py \
    --checkpoint /path/to/dit_checkpoint.ckpt \
    --labels "0,1,2,3,4" \
    --output_dir outputs/samples_classes

# 使用类别名称列表
python src/eval_dit.py \
    --checkpoint /path/to/dit_checkpoint.ckpt \
    --class_list /path/to/class_names.txt \
    --num_samples 50 \
    --output_dir outputs/samples
```

### 批量采样

```bash
# 生成大量样本用于 FID 评估
python src/eval_dit.py \
    --checkpoint /path/to/dit_checkpoint.ckpt \
    --num_samples 10000 \
    --output_dir outputs/fid_samples \
    --num_steps 50
```

---

## 配置说明

### 模型配置 (`configs/model/dit.yaml`)

```yaml
model:
  # RAE 编码器配置（冻结）
  rae:
    encoder_cls: 'Dinov2withNorm'
    encoder_config_path: 'facebook/dinov2-with-registers-base'
    encoder_input_size: 224
    encoder_checkpoint_path: null  # 训练时指定
  
  # DiT/DDT 模型配置
  dit:
    input_size: 16  # 潜在空间尺寸 (16x16)
    patch_size: 1
    in_channels: 768  # RAE 潜在通道数
    hidden_size: [1152, 2048]  # Encoder/Decoder 隐藏层大小
    depth: [28, 2]  # Encoder/Decoder 层数
    num_heads: [16, 16]  # 注意力头数
    mlp_ratio: 4.0
    class_dropout_prob: 0.1
    num_classes: 1000
  
  # Lightning Module 配置
  dit_module:
    ema_decay: 0.9995  # EMA 衰减率
    learning_rate: 2.0e-4
    warmup_steps: 5000
    max_steps: 100000
    num_classes: 1000
    null_label: 1000
    latent_size: [768, 16, 16]
```

### 训练器配置 (`configs/experiment/dit_dino.yaml`)

```yaml
trainer:
  max_epochs: 1400
  gradient_clip_val: 1.0
  accumulate_grad_batches: 1
  precision: 16  # 混合精度训练
  check_val_every_n_epoch: 1
  log_every_n_steps: 100

data:
  image_size: 256
  batch_size: 32  # 每个 GPU 的 batch size
  num_workers: 8
  train_split: 1.0  # 使用全部训练数据
```

### 常用参数覆盖

```bash
# 修改学习率
python src/train.py experiment=dit_dino \
    model.dit_module.learning_rate=1e-4

# 修改 EMA 衰减率
python src/train.py experiment=dit_dino \
    model.dit_module.ema_decay=0.999

# 修改 batch size
python src/train.py experiment=dit_dino \
    data.batch_size=64

# 修改图像尺寸
python src/train.py experiment=dit_dino \
    data.image_size=512

# 启用模型编译（PyTorch 2.0+）
python src/train.py experiment=dit_dino \
    model.dit_module.compile=true

# 修改训练步数
python src/train.py experiment=dit_dino \
    model.dit_module.max_steps=50000 \
    model.dit_module.warmup_steps=1000
```

---

## 常见问题

### 1. 显存不足 (OOM)

**问题**: CUDA out of memory

**解决方案**:
```bash
# 减小 batch size
python src/train.py experiment=dit_dino data.batch_size=8

# 减小图像尺寸
python src/train.py experiment=dit_dino data.image_size=128

# 使用梯度累积
python src/train.py experiment=dit_dino \
    data.batch_size=8 \
    trainer.accumulate_grad_batches=4

# 使用更小的 DiT 模型
# 修改 configs/experiment/dit_dummy.yaml 中的 DiT 配置
```

### 2. 训练速度慢

**问题**: 训练速度太慢

**解决方案**:
```bash
# 增加数据加载进程数
python src/train.py experiment=dit_dino data.num_workers=16

# 使用混合精度（默认已启用）
python src/train.py experiment=dit_dino trainer.precision=16

# 启用模型编译（PyTorch 2.0+）
python src/train.py experiment=dit_dino model.dit_module.compile=true

# 使用更多 GPU
python src/train.py experiment=dit_dino trainer=ddp trainer.devices=8
```

### 3. 生成质量差

**问题**: 生成的图像质量不好

**解决方案**:
```bash
# 增加采样步数
python src/eval_dit.py \
    --checkpoint /path/to/checkpoint.ckpt \
    --num_steps 200

# 使用分类器无关引导
python src/eval_dit.py \
    --checkpoint /path/to/checkpoint.ckpt \
    --cfg_scale 3.0

# 确保使用 EMA 权重
# eval_dit.py 默认使用 ema_dit 权重

# 训练更多 epochs
python src/train.py experiment=dit_dino trainer.max_epochs=2000
```

### 4. RAE 检查点加载失败

**问题**: 无法加载 RAE 检查点

**解决方案**:
```bash
# 检查检查点文件是否存在
ls -l /path/to/rae_checkpoint.ckpt

# 检查 RAE 配置是否匹配
# 确保 configs/experiment/dit_dino.yaml 中的 RAE 配置与训练时一致

# 尝试加载不同的 state_dict 键
# DiTModule 会自动尝试不同的键名: "state_dict", "model", "ema"
```

### 5. 数据加载错误

**问题**: 数据加载失败

**解决方案**:
```bash
# 检查数据目录
ls -l /path/to/imagenet_hf

# 检查数据格式
# 确保 ImageNet 已转换为 HuggingFace 格式

# 使用本地数据集格式
python src/train.py experiment=dit_dino \
    data.use_hf_dataset=false \
    data.data_dir=/path/to/imagenet/train
```

---

## 高级用法

### 自定义 DiT 架构

```python
# 在 configs/experiment/dit_dino.yaml 中修改 DiT 配置
dit:
  input_size: 16  # 潜在空间尺寸
  hidden_size: [1024, 1536]  # 自定义隐藏层大小
  depth: [20, 4]  # 自定义层数
  num_heads: [12, 12]  # 自定义注意力头数
  use_rope: true  # 启用旋转位置编码
  use_rmsnorm: true  # 启用 RMSNorm
  use_swiglu: true  # 启用 SwiGLU
```

### 使用不同的 RAE 编码器

```bash
# 使用 SigLIP 编码器
python src/train.py experiment=dit_dino \
    model.rae.encoder_cls='SigLIPwithNorm' \
    model.rae.encoder_config_path='google/siglip-so400m-patch14-384'

# 使用 MAE 编码器
python src/train.py experiment=dit_dino \
    model.rae.encoder_cls='MAEwithNorm' \
    model.rae.encoder_config_path='facebook/mae-base'
```

### 分布式训练优化

```bash
# 使用 NCCL 后端（多机）
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=29500 \
    src/train.py experiment=dit_dino

# 使用 FSDP（全分片数据并行）
python src/train.py experiment=dit_dino \
    trainer.strategy=fsdp
```

---

## 监控训练

### TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir logs/runs

# 在浏览器中打开 http://localhost:6006
```

查看的指标：
- `train/loss`: 训练损失
- `val/loss`: 验证损失
- `train/lr`: 学习率

### 日志目录

```
logs/
├── runs/
│   └── YYYY-MM-DD/
│       └── HH-MM-SS/
│           ├── .hydra/
│           │   └── config.yaml       # 完整配置
│           ├── checkpoints/          # 模型检查点
│           │   ├── last.ckpt         # 最新检查点
│           │   └── epoch=*.ckpt      # 最佳检查点
│           └── events.out.tfevents.* # TensorBoard 日志
```

---

## 性能基准

### 训练配置建议

| 场景 | GPU 数量 | Batch Size | 图像尺寸 | 训练时间 (预估) |
|------|----------|------------|----------|----------------|
| 测试 | 1 | 4 | 128 | 10 分钟 |
| 小规模 | 1 | 8 | 256 | 2-3 小时 |
| 中等 | 4 | 32 | 256 | 6-8 小时 |
| 完整 | 8 | 64 | 256 | 3-4 天 |

### 生成质量建议

| 采样步数 | CFG Scale | 质量 | 速度 |
|---------|-----------|------|------|
| 50 | 1.0 | 基础 | 快 |
| 100 | 2.0 | 良好 | 中等 |
| 200 | 3.0 | 优秀 | 慢 |
| 250 | 4.0+ | 最佳 | 很慢 |

---

## 相关资源

- [原始 RAE 仓库](../RAE)
- [RAE 训练指南](RAE_TRAINING_GUIDE.md)
- [Linear Probing 指南](LINEAR_PROBING_GUIDE.md)
- [DiT 论文](https://arxiv.org/abs/2212.09748)
- [Lightning 文档](https://pytorch-lightning.readthedocs.io/)

---

## 贡献

欢迎提交 Issue 和 Pull Request！

如有问题，请查看详细文档或提交 Issue。