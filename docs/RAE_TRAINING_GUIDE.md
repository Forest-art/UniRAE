# RAE 训练指南（支持 DINO v2 和 SigLIP2）

本文档说明如何使用 lightning-hydra-template 框架训练 RAE (Reconstruction Autoencoder) 模型，支持使用 DINO v2 或 SigLIP2 作为编码器。

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

### 数据集配置说明

数据模块支持两种配置方式，根据数据集是否有独立的验证 split 选择：

**方式 1：数据集有独立的 train/validation split（推荐）**

如果数据集包含独立的 `train` 和 `validation` split（如官方 ImageNet 数据集），配置如下：

```yaml
# configs/data/my_dataset.yaml
data_dir: /path/to/dataset_with_splits  # 包含 train 和 validation split
use_hf_dataset: true
hf_split: train
hf_validation_split: validation  # 独立的验证集
train_split: 1.0  # 使用全部 train 数据训练
```

**行为说明**：
- 训练：使用 train split 的全部数据（train_split=1.0）
- rFID 评测：使用 validation split（推荐，与训练数据独立）
- 这是最佳配置，可以准确评估模型的泛化能力

**方式 2：数据集只有 train split（从训练集分割）**

如果数据集只有 `train` split（如测试数据集 `data/test_hf`），配置如下：

```yaml
# configs/data/my_dataset.yaml
data_dir: /path/to/dataset_train_only  # 只有 train split
use_hf_dataset: true
hf_split: train
hf_validation_split: validation  # 这个 split 不存在也没关系
train_split: 0.8  # 80% 训练，20% 验证
```

**行为说明**：
- 训练：使用 train split 的 80%（train_split=0.8）
- rFID 评测：使用 train split 的 20%
- 系统会自动检测 validation split 不存在，改用 train_split 参数分割
- 适用于测试和小规模实验，但不推荐用于生产环境

**重要提示**：
- rFID 评测需要验证集来计算与真实数据的 FID 距离
- 如果数据集有独立的 validation split，强烈推荐使用方式 1
- 对于生产环境的完整训练，应该使用有独立验证集的数据集
- 测试数据集 `data/test_hf` 只有 20 张图片，使用方式 2 配置 train_split=0.8 会得到 16 张训练图片和 4 张验证图片

---

## 训练命令

### 小规模测试命令

在完整训练之前，建议先运行小规模测试确保配置正确：

#### DINO v2 测试

```bash
# 使用 dummy 数据集（推荐）
python src/train.py experiment=rae_dummy

# 测试 GPU 训练
python src/train.py experiment=rae_dummy trainer=gpu
```

#### SigLIP2 测试

```bash
# 使用 dummy 数据集（推荐）
python src/train.py experiment=rae_siglip_dummy

# 测试 GPU 训练
python src/train.py experiment=rae_siglip_dummy trainer=gpu
```

**编码器差异说明**：
- **DINO v2**: 默认图像尺寸 224×224，patch_size=16
- **SigLIP2**: 默认图像尺寸 378×378，patch_size=14
- 两者使用相同的训练参数和评估标准，仅编码器不同
- 测试时建议使用 dummy 数据集，已预处理好测试数据

**注意**：
- `rae_dummy` 和 `rae_siglip_dummy` 使用 `data/dummy_dataset_hf` 数据集
- 运行测试时，显存占用较小，适合快速验证配置
- 完整训练时需要指向真实的 ImageNet 数据集路径

### 单 GPU 训练

#### DINO v2 单 GPU

```bash
# 默认单 GPU 训练
python src/train.py experiment=rae_dino

# 使用指定 GPU
CUDA_VISIBLE_DEVICES=0 python src/train.py experiment=rae_dino
```

#### SigLIP2 单 GPU

```bash
# 默认单 GPU 训练
python src/train.py experiment=rae_siglip

# 使用指定 GPU
CUDA_VISIBLE_DEVICES=0 python src/train.py experiment=rae_siglip
```

### 多 GPU DDP 训练

#### DINO v2 多 GPU

```bash
# 使用 8 个 GPU（默认配置）
python src/train.py experiment=rae_ddp

# 使用 4 个 GPU（调整 batch_size）
python src/train.py experiment=rae_ddp data.batch_size=128

# 使用 2 个 GPU（调整 batch_size）
python src/train.py experiment=rae_ddp data.batch_size=256

# 使用 torchrun 启动
torchrun --nproc_per_node=8 src/train.py experiment=rae_ddp
```

#### SigLIP2 多 GPU

```bash
# 使用 8 个 GPU
python src/train.py experiment=rae_siglip data.batch_size=64 trainer.strategy=ddp

# 使用 4 个 GPU（调整 batch_size）
python src/train.py experiment=rae_siglip data.batch_size=128 trainer.strategy=ddp

# 使用 torchrun 启动
torchrun --nproc_per_node=8 src/train.py experiment=rae_siglip data.batch_size=64
```

**batch_size 调整说明**：
- 原始 RAE 配置：global_batch_size=512
- 8 个 GPU：`batch_size=64` (512/8)
- 4 个 GPU：`batch_size=128` (512/4)
- 2 个 GPU：`batch_size=256` (512/2)
- 1 个 GPU：`batch_size=512`（但显存可能不足）

**重要**：DINO v2 和 SigLIP2 使用完全相同的训练参数，仅在编码器类型和图像尺寸上有差异。

---

## 配置说明

### 基本配置覆盖

Hydra 支持通过命令行覆盖任何配置参数：

```bash
# DINO v2 - 修改数据路径
python src/train.py experiment=rae_dino data.data_dir=/path/to/imagenet_hf

# DINO v2 - 修改图像尺寸
python src/train.py experiment=rae_dino data.image_size=256

# DINO v2 - 修改 batch size
python src/train.py experiment=rae_dino data.batch_size=32

# DINO v2 - 修改学习率
python src/train.py experiment=rae_dino model.optimizer.lr=1e-4

# DINO v2 - 修改训练 epoch 数
python src/train.py experiment=rae_dino trainer.max_epochs=100

# SigLIP2 - 修改数据路径
python src/train.py experiment=rae_siglip data.data_dir=/path/to/imagenet_hf

# SigLIP2 - 修改图像尺寸（注意：SigLIP 默认 378，可适当调整）
python src/train.py experiment=rae_siglip data.image_size=378

# SigLIP2 - 修改 batch size
python src/train.py experiment=rae_siglip data.batch_size=32
```

### 组合配置覆盖

```bash
# DINO v2 - 同时修改多个参数
python src/train.py experiment=rae_dino \
    data.data_dir=/path/to/imagenet_hf \
    data.batch_size=64 \
    data.image_size=224 \
    model.optimizer.lr=2e-4 \
    trainer.max_epochs=200

# SigLIP2 - 同时修改多个参数
python src/train.py experiment=rae_siglip \
    data.data_dir=/path/to/imagenet_hf \
    data.batch_size=64 \
    data.image_size=378 \
    model.optimizer.lr=2e-4 \
    trainer.max_epochs=200
```

---

## 实验配置

### 实验配置文件说明

项目提供了四个主要的实验配置（DINO v2 和 SigLIP2 各两个）：

#### 1. `configs/experiment/rae_dino.yaml` (DINO 单 GPU)

适用于 DINO v2 单 GPU 训练的基本配置：

```yaml
defaults:
  - model: rae
  - data: imagenet
  - trainer: default
  - logger: tensorboard
  - callbacks: default

data:
  batch_size: 128  # 单 GPU 的 batch size
  num_workers: 8
  image_size: 256

trainer:
  max_epochs: 16
  precision: 16  # 使用 fp16 混合精度
```

#### 2. `configs/experiment/rae_ddp.yaml` (DINO 多 GPU)

适用于 DINO v2 DDP 多 GPU 训练的配置：

```yaml
defaults:
  - model: rae
  - data: imagenet
  - trainer: ddp
  - logger: tensorboard
  - callbacks: default

data:
  batch_size: 64  # 8 GPU 的每 GPU batch size
  num_workers: 8
  image_size: 256

trainer:
  max_epochs: 16
  precision: 16
  strategy: ddp  # 使用 DDP 策略
```

#### 3. `configs/experiment/rae_siglip.yaml` (SigLIP 单 GPU)

适用于 SigLIP2 单 GPU 训练的基本配置：

```yaml
defaults:
  - model: rae_siglip
  - data: imagenet
  - trainer: default
  - logger: tensorboard
  - callbacks: default

data:
  batch_size: 128  # 单 GPU 的 batch size
  num_workers: 8
  image_size: 378  # SigLIP 默认图像尺寸

trainer:
  max_epochs: 16
  precision: 16
```

#### 4. `configs/experiment/rae_siglip_dummy.yaml` (SigLIP 测试)

适用于 SigLIP2 快速测试的配置：

```yaml
defaults:
  - model: rae_siglip
  - data: imagenet
  - trainer: default
  - logger: tensorboard
  - callbacks: no_checkpoint

data:
  data_dir: data/dummy_dataset_hf
  batch_size: 8
  num_workers: 2
  image_size: 378

trainer:
  max_epochs: 2
  precision: 16
  limit_train_batches: 0.1  # 使用 10% 数据
  limit_val_batches: 0.5
```

---

## 模型配置详情

### 编码器配置

#### DINO v2 配置 (`configs/model/rae.yaml`)

```yaml
model:
  encoder_cls: Dinov2withNorm
  encoder_config_path: facebook/dinov2-with-registers-base
  encoder_input_size: 224
  encoder_params:
    dinov2_path: facebook/dinov2-with-registers-base
    normalize: true
```

#### SigLIP2 配置 (`configs/model/rae_siglip.yaml`)

```yaml
model:
  encoder_cls: SigLIP2wNorm
  encoder_config_path: google/siglip-so400m-patch14-384
  encoder_input_size: 378
  encoder_params:
    model_name: google/siglip-so400m-patch14-384
    num_tokens: 256
```

### 编码器对比

| 特性 | DINO v2 | SigLIP2 |
|------|---------|---------|
| 模型 | facebook/dinov2-with-registers-base | google/siglip-so400m-patch14-384 |
| 输入尺寸 | 224 | 378 |
| Patch Size | 16 | 14 |
| 特征维度 | 768 | 1152 |
| 配置文件 | configs/model/rae.yaml | configs/model/rae_siglip.yaml |

### 解码器配置

```yaml
model:
  decoder_config_path: configs/decoder/ViTXL/config.json
  decoder_patch_size: 16
  pretrained_decoder_path: null
```

### 训练配置

```yaml
model:
  # 优化器配置
  optimizer:
    lr: 2e-4
    betas: [0.9, 0.95]
    weight_decay: 0.0
  
  # 学习率调度器
  scheduler:
    type: cosine
    warmup_epochs: 1
    decay_end_epoch: 16
    base_lr: 2e-4
    final_lr: 2e-5
    warmup_from_zero: true
  
  # EMA 配置
  ema_decay: 0.9978
  
  # GAN 损失权重
  disc_weight: 0.75
  perceptual_weight: 1.0
  disc_start_epoch: 8
  disc_upd_start_epoch: 6
  lpips_start_epoch: 0
  
  # 采样配置
  sample_every: 2500  # 每 2500 步生成一次样本
```

### 判别器配置

```yaml
model:
  disc_arch:
    arch:
      dino_ckpt_path: /home/project/models/discs/dino_vit_small_patch8_224.pth
      ks: 9
      norm_type: bn
      using_spec_norm: true
      recipe: S_8
    augment:
      prob: 1.0
      cutout: 0.0
  
  disc_optimizer:
    lr: 2e-4
    betas: [0.9, 0.95]
    weight_decay: 0.0
```

---

## rFID 评测

### 边训边评 (推荐)

RAE 训练配置已集成 rFID 评测 Callback，可以在训练过程中自动评测 rFID：

```bash
# 使用默认配置训练（每 1000 步和每个 epoch 结束时评测 rFID）
python src/train.py experiment=rae_dino

# 自定义 rFID 评测频率
python src/train.py experiment=rae_dino \
    callbacks.rfid.rfid_every_n_steps=500 \
    callbacks.rfid.rfid_every_epoch=true

# 仅在 epoch 结束时评测
python src/train.py experiment=rae_dino \
    callbacks.rfid.rfid_every_n_steps=0 \
    callbacks.rfid.rfid_every_epoch=true

# 修改评测样本数和设备
python src/train.py experiment=rae_dino \
    callbacks.rfid.rfid_num_samples=500 \
    callbacks.rfid.rfid_device=cuda \
    callbacks.rfid.rfid_save_samples=true \
    callbacks.rfid.save_samples_count=32
```

### rFID Callback 配置参数

在 `configs/callbacks/rae.yaml` 中可以调整以下参数：

```yaml
rfid:
  _target_: src.callbacks.rfid_callback.RFIDCallback
  rfid_every_n_steps: 1000      # 每 N 步评测一次（0 表示禁用）
  rfid_every_epoch: true          # 每个 epoch 结束时评测
  rfid_num_samples: 1000         # 评测使用的样本数
  rfid_batch_size: 64            # FID 计算的批大小
  rfid_device: cuda              # 计算设备
  rfid_output_dir: logs/rfid_samples  # 保存样本的目录
  rfid_save_samples: true        # 是否保存样本图像
  save_samples_count: 64         # 保存的样本对数量
  use_train_dataloader: true      # 没有验证集时使用训练集评测（RAE默认没有验证集）
```

**注意**: RAE 训练默认不使用验证集，因此 `use_train_dataloader` 默认为 `true`，会在训练集上进行 rFID 评测。这可以实时监控重建质量，但要注意 rFID 分数可能与在独立测试集上的结果略有不同。

### rFID 结果查看

- **TensorBoard**: rFID 分数会自动记录到 TensorBoard，可以通过 `rfid/score` 查看
- **样本图像**: 如果启用了 `rfid_save_samples`，原始图像和重建图像会保存在 `logs/rfid_samples/step_XXX/` 或 `logs/rfid_samples/epoch_XXX/` 目录下
- **训练日志**: rFID 评测的详细信息会打印在训练日志中

### 什么是 rFID

rFID (reconstruction FID) 是衡量自编码器重建质量的指标，通过计算原始图像和重建图像之间的 FID 分数来评估。分数越低表示重建质量越好。

### 评测命令

```bash
# 使用训练好的模型评测 rFID
python src/eval_rfid.py \
    --checkpoint logs/train/runs/XXXX-XX-XX/checkpoints/best.ckpt \
    --data_dir /path/to/imagenet_hf \
    --num_samples 10000 \
    --batch_size 64 \
    --device cuda \
    --output_dir logs/rfid_samples

# 使用 CPU 评测（如果没有 GPU）
python src/eval_rfid.py \
    --checkpoint logs/train/runs/XXXX-XX-XX/checkpoints/best.ckpt \
    --data_dir /path/to/imagenet_hf \
    --device cpu
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --checkpoint | 模型检查点路径 | 必需 |
| --data_dir | 数据集目录 | 必需 |
| --num_samples | 评测样本数量 | 10000 |
| --batch_size | 批量大小 | 64 |
| --image-size | 图像尺寸 | 256 |
| --device | 计算设备 (cuda/cpu) | cuda |
| --output-dir | 保存样本图像的目录 | None |
| --num-workers | 数据加载进程数 | 4 |

### 输出说明

评测脚本会输出：
1. rFID 分数（越低越好）
2. 如果指定了 `--output_dir`，会保存原始图像和重建图像对比样本

```bash
rFID Score: 12.3456

Sample images saved to: logs/rfid_samples
```

### 评测时机建议

- 在训练结束后使用最佳模型评测
- 可以在验证集上评测，避免过拟合
- 建议使用至少 10000 张图像以获得稳定的结果

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

### 4. DDP 训练问题

确保：
1. 所有 GPU 可见：`nvidia-smi`
2. 检查 NCCL 版本兼容性
3. 正确设置 batch_size（总 batch size = batch_size × num_gpus）

---

## 配置文件位置速查

| 配置类型 | DINO v2 | SigLIP2 |
|---------|---------|----------|
| 单 GPU 实验 | `configs/experiment/rae_dino.yaml` | `configs/experiment/rae_siglip.yaml` |
| 多 GPU 实验 | `configs/experiment/rae_ddp.yaml` | 使用 `rae_siglip.yaml` + `trainer.strategy=ddp` |
| 测试配置 | `configs/experiment/rae_dummy.yaml` | `configs/experiment/rae_siglip_dummy.yaml` |
| 模型配置 | `configs/model/rae.yaml` | `configs/model/rae_siglip.yaml` |

| 配置类型 | 路径 |
|---------|------|
| 数据配置 | `configs/data/imagenet.yaml` |
| 训练器配置 | `configs/trainer/default.yaml` |
| DDP 训练器 | `configs/trainer/ddp.yaml` |
| TensorBoard | `configs/logger/tensorboard.yaml` |
| RAE 回调 | `configs/callbacks/rae.yaml` |
| rFID 回调 | `configs/callbacks/rfid.yaml` |

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
5. 判别器 checkpoint 路径是否正确

## 训练参数说明

### 通用训练参数（DINO v2 和 SigLIP2 均适用）

基于原始 RAE 配置文件，DINO v2 和 SigLIP2 使用完全相同的训练参数：

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

**重要**：DINO v2 和 SigLIP2 使用完全相同的训练参数，仅在编码器类型和图像尺寸上有差异。训练逻辑、评估方法和 rFID 评测完全一致。
