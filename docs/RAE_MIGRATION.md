# RAE Training Migration Guide

本文档说明如何使用 lightning-hydra-template 训练 RAE 模型。

## 迁移概述

已将以下组件从 `/home/project/RAE` 迁移到 `/home/project/lightning-hydra-template`：

- ✅ **核心模型** (`src/models/stage1/`): RAE 编码器和解码器
- ✅ **判别器** (`src/models/components/disc/`): GAN 判别器和损失函数
- ✅ **工具函数** (`src/utils/`): 优化器、训练工具等
- ✅ **Lightning Module** (`src/models/rae_module.py`): RAE Lightning 包装器
- ✅ **Data Module** (`src/data/image_folder_datamodule.py`): ImageFolder 数据模块
- ✅ **配置文件**: 模型、数据和实验配置

## 目录结构

```
lightning-hydra-template/
├── src/
│   ├── models/
│   │   ├── stage1/          # RAE 编码器和解码器
│   │   ├── components/
│   │   │   └── disc/        # GAN 判别器
│   │   └── rae_module.py     # RAE Lightning Module
│   ├── data/
│   │   └── image_folder_datamodule.py  # 数据模块
│   └── utils/                # 工具函数
├── configs/
│   ├── decoder/             # 解码器配置 (从 RAE 迁移)
│   ├── model/
│   │   └── rae.yaml         # RAE 模型配置
│   ├── data/
│   │   └── imagenet.yaml    # 数据集配置
│   └── experiment/
│       └── rae_dino.yaml     # 实验配置
└── requirements.txt          # 更新的依赖
```

## 安装依赖

```bash
cd /home/project/lightning-hydra-template
pip install -r requirements.txt
```

## 准备数据集

### 方式一：使用 HuggingFace 数据集（推荐）

如果您的数据集已经使用 `save_from_disk()` 保存为 HuggingFace 格式：

```python
from datasets import load_dataset

# 加载并保存数据集
dataset = load_dataset("path/to/your/dataset")
dataset.save_to_disk("/path/to/imagenet_hf")
```

训练时设置数据集路径：

```bash
python src/train.py experiment=rae_dino data.data_dir=/path/to/imagenet_hf
```

### 方式二：使用 ImageFolder 格式

将数据集组织为 ImageFolder 格式：

```
data_dir/
├── class_0/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── class_1/
│   ├── image_001.jpg
│   └── ...
└── ...
```

训练时需要设置 `use_hf_dataset=false`：

```bash
python src/train.py experiment=rae_dino \
    data.data_dir=/path/to/imagenet/train \
    data.use_hf_dataset=false
```

## 训练命令

### 基础训练

```bash
python src/train.py experiment=rae_dino data.data_dir=/path/to/your/dataset
```

### 自定义参数

```bash
# 使用 HuggingFace 数据集（默认）
python src/train.py experiment=rae_dino \
    data.data_dir=/path/to/imagenet_hf

# 使用 ImageFolder 数据集
python src/train.py experiment=rae_dino \
    data.data_dir=/path/to/imagenet/train \
    data.use_hf_dataset=false

# 使用 HuggingFace 数据集的验证集
python src/train.py experiment=rae_dino \
    data.data_dir=/path/to/imagenet_hf \
    data.hf_split=validation

# 更改批量大小和学习率
python src/train.py experiment=rae_dino \
    data.batch_size=32 \
    model.optimizer.lr=1e-4 \
    data.data_dir=/path/to/your/dataset

# 更改模型配置
python src/train.py experiment=rae_dino \
    model.decoder_config_path=configs/decoder/ViTL/decoder_config.json \
    model.encoder_cls='MaeEncoder' \
    data.data_dir=/path/to/your/dataset

# 使用 GPU 和多卡训练
python src.train.py experiment=rae_dino \
    trainer=gpu \
    trainer.devices=4 \
    trainer.accelerator="gpu" \
    trainer.strategy="ddp" \
    data.data_dir=/path/to/your/dataset
```

### 混合精度训练

实验配置中已启用混合精度：

```bash
python src/train.py experiment=rae_dino \
    data.data_dir=/path/to/your/dataset
```

### 恢复训练

```bash
python src/train.py experiment=rae_dino \
    ckpt_path=/path/to/checkpoint.ckpt \
    data.data_dir=/path/to/your/dataset
```

## 配置说明

### 模型配置 (`configs/model/rae.yaml`)

主要参数：

- `encoder_cls`: 编码器类型 ('Dinov2withNorm', 'MaeEncoder', 'SigLIP2')
- `encoder_config_path`: 编码器模型路径
- `decoder_config_path`: 解码器配置文件路径
- `noise_tau`: 噪声注入参数
- `disc_weight`: GAN 损失权重
- `perceptual_weight`: LPIPS 损失权重
- `disc_start_epoch`: 开始使用 GAN 的 epoch
- `optimizer`: 优化器配置
- `scheduler`: 学习率调度器配置

### 数据配置 (`configs/data/imagenet.yaml`)

主要参数：

- `data_dir`: 数据集路径
  - HuggingFace 格式：`save_from_disk()` 保存的路径
  - ImageFolder 格式：包含 class 子目录的路径
- `image_size`: 图像大小
- `batch_size`: 批量大小
- `num_workers`: 数据加载器工作进程数
- `use_hf_dataset`: 是否使用 HuggingFace 数据集格式（true）或 ImageFolder（false）
- `hf_split`: HuggingFace 数据集使用的分割（'train' 或 'validation'）

### 实验配置 (`configs/experiment/rae_dino.yaml`)

组合模型、数据和训练器配置：

- 默认训练 16 epochs
- 使用混合精度 (fp16)
- 每 epoch 保存检查点

## 差异说明

### 原始训练 vs Lightning-Hydra

| 特性 | 原始 RAE | Lightning-Hydra |
|------|---------|-----------------|
| 框架 | 原生 PyTorch | PyTorch Lightning |
| 配置 | JSON/YAML | Hydra YAML |
| 日志 | W&B | W&B/TensorBoard 等 |
| 检查点 | 手动管理 | 自动管理 |
| 多卡 | DDP 手动 | DDP 原生支持 |
| 实验追踪 | 手动 | Hydra 自动化 |

### 训练流程

原始训练逻辑已完整迁移到 `RAELitModule` 中：

1. **编码器冻结**：只训练解码器
2. **损失函数**：L1 + LPIPS + GAN
3. **EMA 更新**：每个 batch 更新 EMA 模型
4. **判别器训练**：在指定 epoch 后开始训练
5. **学习率调度**：Cosine 退火 + warmup

## 下一步

- [ ] 添加验证指标计算 (FID, LPIPS)
- [ ] 实现采样和可视化回调
- [ ] 添加 Stage-2 训练支持
- [ ] 创建单元测试
- [ ] 性能基准测试

## 故障排除

### 导入错误

如果遇到模块导入错误，确保：

```bash
export PYTHONPATH="${PYTHONPATH}:/home/project/lightning-hydra-template"
```

### CUDA 内存不足

减少批量大小或使用梯度累积：

```bash
python src/train.py experiment=rae_dino \
    data.batch_size=8 \
    trainer.accumulate_grad_batches=2 \
    data.data_dir=/path/to/your/dataset
```

### HuggingFace 数据集准备

如果您的原始数据集是 ImageFolder 格式，可以转换为 HuggingFace 格式：

```python
from datasets import load_dataset, DatasetDict
from pathlib import Path
from PIL import Image

# 将 ImageFolder 转换为 HuggingFace 格式
def convert_imagefolder_to_hf(imagefolder_path, output_path):
    """Convert ImageFolder to HuggingFace format."""
    imagefolder_path = Path(imagefolder_path)
    
    images = []
    labels = []
    class_names = sorted([d.name for d in imagefolder_path.iterdir() if d.is_dir()])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    for class_name in class_names:
        class_path = imagefolder_path / class_name
        class_idx = class_to_idx[class_name]
        
        for img_path in class_path.glob("*.jpg"):
            images.append(str(img_path))
            labels.append(class_idx)
    
    # Create dataset
    dataset = Dataset.from_dict({"image": images, "label": labels})
    
    # Load images
    def load_image(path):
        return Image.open(path).convert("RGB")
    
    dataset = dataset.map(lambda x: {"image": load_image(x["image"])})
    
    # Save
    dataset.save_to_disk(output_path)
    print(f"Saved dataset to {output_path}")

# 使用
convert_imagefolder_to_hf(
    "/path/to/imagenet/train",
    "/path/to/imagenet_hf"
)
```

## 联系方式

如有问题，请参考原始 RAE 仓库或提交 Issue。