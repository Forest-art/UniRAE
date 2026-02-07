# RAE DINO 训练指南

本文档说明如何使用 lightning-hydra-template 框架训练 RAE (Reconstruction Autoencoder) 模型，使用 DINO v2 作为编码器。

## 目录
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [训练命令](#训练命令)
- [配置说明](#配置说明)
- [实验配置](#实验配置)

---

## 环境配置

### 1. 安装依赖

```bash
cd lightning-hydra-template

# 使用 conda
conda env create -f environment.yaml
conda activate rae-train

# 或使用 pip
pip install -r requirements.txt
```

### 2. 验证安装

```bash
python -c "import torch; import pytorch_lightning; import hydra; print('所有依赖已安装')"
```

---

## 数据准备

### ImageNet 数据集

1. 下载 ImageNet 数据集（假设数据已下载到 `/path/to/imagenet`）

2. **重要**: ImageNet 数据需要转换为 HuggingFace Dataset 格式

```bash
# 将 ImageNet 转换为 HuggingFace 格式
python scripts/create_hf_test_data.py \
    --input_dir /path/to/imagenet/train \
    --output_dir /path/to/imagenet_hf

# 或者使用提供的脚本（如果有预处理的示例数据）
python scripts/create_hf_test_data.py
```

### 数据目录结构

转换后的数据应具有以下结构：

```
/path/to/imagenet_hf/
├── dataset_info.json
├── data-00000-of-00001.arrow
└── ...
```

---

## 训练命令

### 单 GPU 训练

```bash
python src/train.py experiment=rae_dino
```

### 多 GPU DDP 训练

使用 4 个 GPU 进行 DDP 训练：

```bash
# 使用 torchrun (PyTorch >= 1.10)
python -m torch.distributed.run --nproc_per_node=4 src/train.py experiment=rae_ddp

# 或使用 CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 src/train.py experiment=rae_ddp
```

### 指定使用的 GPU

```bash
# 使用 GPU 0 和 1
CUDA_VISIBLE_DEVICES=0,1 python src/train.py experiment=rae_dino trainer.devices=2
```

---

## 配置说明

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

### 组合配置覆盖

```bash
# 同时修改多个参数
python src/train.py experiment=rae_dino \
    data.data_dir=/path/to/imagenet_hf \
    data.batch_size=64 \
    data.image_size=224 \
    model.optimizer.lr=2e-4 \
    trainer.max_epochs=200 \
    trainer.devices=4
```

---

## 实验配置

### 实验配置文件说明

项目提供了两个主要的实验配置：

#### 1. `configs/experiment/rae_dino.yaml` (单 GPU)

适用于单 GPU 训练的基本配置：

```yaml
defaults:
  - model: rae
  - data: imagenet
  - trainer: default
  - logger: tensorboard
  - callbacks: default

data:
  batch_size: 32
  num_workers: 4
  image_size: 224

model:
  encoder: dinov2
  decoder: ViTXL
  image_size: 224
```

#### 2. `configs/experiment/rae_ddp.yaml` (多 GPU)

适用于 DDP 多 GPU 训练的配置：

```yaml
defaults:
  - model: rae
  - data: imagenet
  - trainer: ddp
  - logger: tensorboard
  - callbacks: none

data:
  batch_size: 32
  num_workers: 8
  image_size: 224

model:
  encoder: dinov2
  decoder: ViTXL
  image_size: 224
```

### 使用不同的编码器

```bash
# 使用 DINOv2-B (默认)
python src/train.py experiment=rae_dino model.encoder=dinov2

# 使用 MAE
python src/train.py experiment=rae_dino model.encoder=mae

# 使用 SigLIP2
python src/train.py experiment=rae_dino model.encoder=siglip2
```

### 使用不同的解码器

```bash
# 使用 ViT-B decoder
python src/train.py experiment=rae_dino model.decoder=ViTB

# 使用 ViT-L decoder
python src/train.py experiment=rae_dino model.decoder=ViTL

# 使用 ViT-XL decoder (默认)
python src/train.py experiment=rae_dino model.decoder=ViTXL
```

---

## 模型配置详情

### 编码器配置

在 `configs/model/rae.yaml` 中配置编码器：

```yaml
model:
  encoder: dinov2          # 编码器类型: dinov2, mae, siglip2
  encoder_name: DINOv2-B    # 具体模型名称
  pretrained: true         # 是否使用预训练权重
  freeze_encoder: false    # 是否冻结编码器参数
```

### 解码器配置

```yaml
model:
  decoder: ViTXL           # 解码器类型: ViTB, ViTL, ViTXL
  decoder_config_path: ${oc.decode:configs/decoder/ViTXL/config.json}
  latent_dim: 1024         # 潜在空间维度
```

### 训练配置

```yaml
model:
  optimizer:
    name: adamw            # 优化器: adamw, adam, sgd
    lr: 1e-4               # 学习率
    weight_decay: 0.05     # 权重衰减
  
  scheduler:
    name: cosine           # 学习率调度器
    warmup_epochs: 5       # 预热 epoch 数
  
  loss_weights:
    reconstruction: 1.0    # 重建损失权重
    perceptual: 0.1        # 感知损失权重
    adversarial: 0.001     # 对抗损失权重
```

---

## 训练输出

### 日志目录

训练日志和模型检查点会保存在以下目录：

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

### 监控训练

使用 TensorBoard 查看训练进度：

```bash
tensorboard --logdir logs/runs
```

然后在浏览器中打开 `http://localhost:6006`

---

## 常见问题

### 1. 内存不足 (OOM)

如果遇到 CUDA OOM 错误，尝试以下方法：

```bash
# 减小 batch size
python src/train.py experiment=rae_dino data.batch_size=16

# 减小图像尺寸
python src/train.py experiment=rae_dino data.image_size=128

# 使用梯度累积
python src/train.py experiment=rae_dino trainer.accumulate_grad_batches=4
```

### 2. 数据加载慢

增加 `num_workers` 以加速数据加载：

```bash
python src/train.py experiment=rae_dino data.num_workers=8
```

### 3. 恢复训练

从检查点恢复训练：

```bash
# 从最新的检查点恢复
python src/train.py experiment=rae_dino ckpt_path="last"

# 从指定路径恢复
python src/train.py experiment=rae_dino ckpt_path="/path/to/checkpoint.ckpt"
```

### 4. 仅评估

加载模型进行评估：

```bash
python src/eval.py experiment=rae_dino ckpt_path="/path/to/checkpoint.ckpt"
```

---

## 配置文件位置速查

| 配置类型 | 路径 |
|---------|------|
| 实验配置 | `configs/experiment/rae_dino.yaml` |
| 模型配置 | `configs/model/rae.yaml` |
| 数据配置 | `configs/data/imagenet.yaml` |
| 训练器配置 | `configs/trainer/default.yaml` / `configs/trainer/ddp.yaml` |
| Logger 配置 | `configs/logger/tensorboard.yaml` |
| 回调配置 | `configs/callbacks/default.yaml` |

---

## 参考文档

- [Lightning 官方文档](https://pytorch-lightning.readthedocs.io/)
- [Hydra 官方文档](https://hydra.cc/)
- [RAE 原始仓库](../RAE)

---

## 支持

如有问题，请检查：
1. 数据路径是否正确
2. 依赖是否完整安装
3. GPU 驱动和 CUDA 版本是否兼容
4. 配置参数是否合理