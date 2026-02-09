# UniRAE 配置文件结构说明

## 概述

UniRAE 的配置文件采用分层结构，按照训练阶段（Stage 1 和 Stage 2）组织，同时通过符号链接保持向后兼容。

## 目录结构

```
configs/
├── stage1/                    # Stage 1 (RAE) 配置
│   ├── models/                # RAE 模型配置
│   │   ├── rae.yaml         # DINO v2 RAE 模型
│   │   ├── rae_siglip.yaml  # SigLIP2 RAE 模型
│   │   └── linear_probe.yaml # Linear Probing 模型
│   └── experiments/          # RAE 实验配置
│       ├── rae_dummy.yaml         # DINO 测试配置
│       ├── rae_dino.yaml         # DINO 完整训练
│       ├── rae_ddp.yaml          # DINO DDP 训练
│       ├── rae_siglip.yaml        # SigLIP 完整训练
│       ├── rae_siglip_dummy.yaml  # SigLIP 测试配置
│       └── linear_probe.yaml      # Linear Probing 配置
│
├── stage2/                    # Stage 2 (DiT) 配置
│   ├── models/                # DiT 模型配置
│   │   └── dit.yaml         # DiT 模型
│   └── experiments/          # DiT 实验配置
│       ├── dit_dummy.yaml      # DiT 测试配置
│       ├── dit_dino.yaml      # DiT 完整训练
│       ├── dit_base.yaml      # DiT-Base 配置
│       ├── dit_large.yaml     # DiT-Large 配置
│       ├── dit_xl.yaml       # DiT-XL 配置
│       ├── dit_training.yaml  # DiT 训练配置
│       ├── stage2_dit.yaml   # Stage 2 DiT 配置
│       ├── stage2_dit_512.yaml # Stage 2 DiT 512 配置
│       └── stage2_dit_s.yaml # Stage 2 DiT S 配置
│
├── callbacks/               # Callback 配置
├── data/                    # 数据配置
├── trainer/                 # Trainer 配置
│
├── model/                   # 符号链接（向后兼容）
│   ├── rae.yaml -> ../stage1/models/rae.yaml
│   ├── rae_siglip.yaml -> ../stage1/models/rae_siglip.yaml
│   ├── linear_probe.yaml -> ../stage1/models/linear_probe.yaml
│   └── dit.yaml -> ../stage2/models/dit.yaml
│
└── experiment/              # 符号链接（向后兼容）
    ├── rae_dummy.yaml -> ../stage1/experiments/rae_dummy.yaml
    ├── rae_dino.yaml -> ../stage1/experiments/rae_dino.yaml
    ├── rae_ddp.yaml -> ../stage1/experiments/rae_ddp.yaml
    ├── rae_siglip.yaml -> ../stage1/experiments/rae_siglip.yaml
    ├── rae_siglip_dummy.yaml -> ../stage1/experiments/rae_siglip_dummy.yaml
    ├── linear_probe.yaml -> ../stage1/experiments/linear_probe.yaml
    ├── dit_dummy.yaml -> ../stage2/experiments/dit_dummy.yaml
    ├── dit_dino.yaml -> ../stage2/experiments/dit_dino.yaml
    ├── dit_base.yaml -> ../stage2/experiments/dit_base.yaml
    ├── dit_large.yaml -> ../stage2/experiments/dit_large.yaml
    ├── dit_xl.yaml -> ../stage2/experiments/dit_xl.yaml
    ├── dit_training.yaml -> ../stage2/experiments/dit_training.yaml
    ├── stage2_dit.yaml -> ../stage2/experiments/stage2_dit.yaml
    ├── stage2_dit_512.yaml -> ../stage2/experiments/stage2_dit_512.yaml
    └── stage2_dit_s.yaml -> ../stage2/experiments/stage2_dit_s.yaml
```

## 设计原则

### 1. 分层组织
- **Stage 1**: RAE (Reconstruction Autoencoder) 训练
- **Stage 2**: DiT (Diffusion Transformer) 训练
- 每个阶段包含 `models/` 和 `experiments/` 子目录

### 2. 向后兼容
- 使用符号链接将配置文件映射到旧路径
- 用户可以继续使用 `experiment=rae_dino` 等旧命令
- 不需要修改现有脚本或文档

### 3. 清晰的命名
- 配置文件命名清晰反映其用途
- 例如：`rae_dummy.yaml` 表示 RAE 的测试配置
- 例如：`dit_dino.yaml` 表示 DiT 使用 DINO 编码器的配置

## 使用方法

### Stage 1: RAE 训练

```bash
# DINO v2 训练
python src/train.py experiment=rae_dino

# DINO v2 测试
python src/train.py experiment=rae_dummy

# SigLIP2 训练
python src/train.py experiment=rae_siglip

# DDP 训练（8 GPU）
python src/train.py experiment=rae_ddp

# Linear Probing
python src/train.py experiment=linear_probe \
    model.encoder_checkpoint=logs/runs/XXXX/checkpoints/last.ckpt
```

### Stage 2: DiT 训练

```bash
# DiT 测试
python src/train.py experiment=dit_dummy

# DiT 完整训练
python src/train.py experiment=dit_dino

# DiT Base/Large/XL 变体
python src/train.py experiment=dit_base
python src/train.py experiment=dit_large
python src/train.py experiment=dit_xl
```

### DiT 采样

```bash
# 单 GPU 采样
python src/sample_dit.py \
    --config configs/sample_dit.yaml \
    --mode single \
    --class-labels 207 360 \
    --output-path samples.png

# DDP 采样（大规模）
torchrun --nproc_per_node=8 src/sample_dit.py \
    --config configs/sample_dit.yaml \
    --mode ddp \
    --num-fid-samples 50000
```

## 配置继承

配置文件使用 Hydra 的 `defaults` 机制进行继承：

```yaml
# example: configs/experiment/rae_dino.yaml
defaults:
  - override /data: imagenet
  - override /model: rae  # 引用 configs/model/rae.yaml
  - override /callbacks: rae
  - override /trainer: default
```

## 新增配置

要添加新的实验配置：

1. **创建模型配置**（如果需要）：
   ```bash
   # Stage 1
   touch configs/stage1/models/my_rae.yaml
   
   # Stage 2
   touch configs/stage2/models/my_dit.yaml
   ```

2. **创建实验配置**：
   ```bash
   # Stage 1
   touch configs/stage1/experiments/my_experiment.yaml
   
   # Stage 2
   touch configs/stage2/experiments/my_experiment.yaml
   ```

3. **创建符号链接**（可选，为了向后兼容）：
   ```bash
   # Stage 1
   ln -s ../stage1/experiments/my_experiment.yaml configs/experiment/my_experiment.yaml
   
   # Stage 2
   ln -s ../stage2/experiments/my_experiment.yaml configs/experiment/my_experiment.yaml
   ```

4. **使用新配置**：
   ```bash
   python src/train.py experiment=my_experiment
   ```

## 配置文件模板

### RAE 模型配置模板

```yaml
# configs/stage1/models/rae.yaml
_target_: src.models.rae_module.RAELitModule

encoder_cls: Dinov2withNorm
encoder_config_path: facebook/dinov2-with-registers-base
encoder_input_size: 224
encoder_params:
  dinov2_path: facebook/dinov2-with-registers-base
  normalize: true

decoder_config_path: ${paths.root_dir}/configs/decoder/ViTB/config.json
pretrained_decoder_path: null
decoder_patch_size: 14
image_size: 224

optimizer:
  lr: 2e-4
  betas: [0.9, 0.95]
  weight_decay: 0.0

# ... 更多配置
```

### DiT 模型配置模板

```yaml
# configs/stage2/models/dit.yaml
_target_: src.models.dit_module.DiTModule

rae:
  _target_: src.models.stage1.RAE
  # ... RAE 配置

dit:
  _target_: src.models.stage2.DiTwDDTHead
  # ... DiT 配置

ema_decay: 0.9995
learning_rate: 2e-4
# ... 更多配置
```

## 常见问题

### Q: 为什么使用符号链接而不是直接复制文件？
A: 符号链接确保配置文件只有一个真实位置，避免重复和同步问题。修改源文件会自动反映在所有引用处。

### Q: 可以直接引用 stage1/experiments/ 配置吗？
A: 可以，但不推荐。使用符号链接的路径（如 `experiment=rae_dino`）更加简洁且兼容旧文档。

### Q: 如何知道一个配置文件是哪个阶段的？
A: 查看文件位置：
- `configs/stage1/` → Stage 1 (RAE)
- `configs/stage2/` → Stage 2 (DiT)

### Q: 可以混合使用两个阶段的配置吗？
A: 不推荐。每个阶段应该独立运行，Stage 2 依赖 Stage 1 的输出（预训练的 RAE 模型）。

## 维护指南

### 添加新功能
1. 在对应的 `stage1/` 或 `stage2/` 目录下添加配置
2. 如需向后兼容，在 `configs/experiment/` 和 `configs/model/` 创建符号链接

### 修改现有配置
1. 编辑源文件（如 `configs/stage1/experiments/rae_dino.yaml`）
2. 符号链接会自动反映修改

### 迁移旧配置
1. 将旧配置移动到适当的 `stage1/` 或 `stage2/` 目录
2. 创建符号链接保持兼容性
3. 更新相关文档

## 总结

这个配置结构：
- ✅ 清晰的组织：按训练阶段分离
- ✅ 向后兼容：通过符号链接保持旧接口
- ✅ 易于扩展：添加新配置简单直观
- ✅ 易于维护：避免重复，单一真相源
- ✅ 清晰命名：文件名反映用途

如需更多帮助，请参考：
- [RAE 训练指南](RAE_TRAINING_GUIDE.md)
- [DiT 训练指南](DIT_TRAINING_GUIDE.md)
- [迁移总结](MIGRATION_SUMMARY.md)