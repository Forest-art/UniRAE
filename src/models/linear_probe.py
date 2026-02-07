"""Linear Probing model for evaluating encoder representations."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy
from einops import rearrange

from src.models.stage1.encoders.dinov2 import Dinov2withNorm


class LinearProbeModel(LightningModule):
    """
    Linear Probing model for evaluating encoder representations.
    
    Freezes the encoder and trains only a linear classifier on top of the features.
    This is a common way to evaluate the quality of learned representations.
    """
    
    def __init__(
        self,
        encoder_cls: str,
        encoder_config_path: str,
        encoder_input_size: int,
        encoder_checkpoint: str,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        max_epochs: int = 100,
        freeze_encoder: bool = True,
        normalize_features: bool = True,
        pool_type: str = "avg",
    ):
        """
        Args:
            encoder_cls: Encoder class name (e.g., 'Dinov2withNorm')
            encoder_config_path: HuggingFace model path or config
            encoder_input_size: Input image size for encoder
            encoder_checkpoint: Path to pretrained encoder checkpoint
            num_classes: Number of classes for classification
            lr: Learning rate for linear classifier
            weight_decay: Weight decay for optimizer
            max_epochs: Maximum number of epochs
            freeze_encoder: Whether to freeze encoder weights
            normalize_features: Whether to normalize features before classifier
            pool_type: Pooling type ('avg', 'cls', or 'flatten')
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Load encoder
        self.encoder = Dinov2withNorm(
            dinov2_path=encoder_config_path,
            normalize=False,  # We'll normalize features separately if needed
        )
        
        # Load encoder weights
        if encoder_checkpoint is not None:
            print(f"[LinearProbe] Loading encoder from {encoder_checkpoint}")
            state_dict = torch.load(encoder_checkpoint, map_location="cpu")
            if "encoder_state_dict" in state_dict:
                encoder_state = state_dict["encoder_state_dict"]
            else:
                # Try to extract encoder from full checkpoint
                encoder_state = {}
                for k, v in state_dict.items():
                    if k.startswith("encoder."):
                        encoder_state[k[8:]] = v
            
            # Load with strict=False to allow partial loading
            missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
            if missing:
                print(f"[LinearProbe] Missing keys: {missing}")
            if unexpected:
                print(f"[LinearProbe] Unexpected keys: {unexpected}")
        else:
            print("[LinearProbe] Using pretrained encoder from HuggingFace")
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("[LinearProbe] Encoder frozen")
        
        # Linear classifier
        # Determine feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, encoder_input_size, encoder_input_size)
            dummy_feat = self.encoder(dummy)
            if pool_type == "cls":
                feat_dim = dummy_feat.shape[-1]  # CLS token
            elif pool_type == "avg":
                feat_dim = dummy_feat.shape[-1]  # Average pooled
            else:  # flatten
                feat_dim = dummy_feat.view(1, -1).shape[-1]
        
        self.feature_dim = feat_dim
        self.classifier = nn.Linear(feat_dim, num_classes)
        
        # Normalization
        self.normalize_features = normalize_features
        self.pool_type = pool_type
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Extract features
        features = self.encoder(x)  # (B, num_patches, D) or (B, D)
        
        # Pool features
        if self.pool_type == "cls":
            # Use CLS token
            features = features[:, 0]  # (B, D)
        elif self.pool_type == "avg":
            # Average over patches
            features = features.mean(dim=1)  # (B, D)
        else:  # flatten
            features = features.view(x.shape[0], -1)  # (B, D*P)
        
        # Normalize if specified
        if self.normalize_features:
            features = F.normalize(features, dim=-1)
        
        # Classification
        logits = self.classifier(features)
        return logits
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Log metrics
        preds = logits.argmax(dim=1)
        self.train_acc(preds, y)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/acc", self.train_acc, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Log metrics
        preds = logits.argmax(dim=1)
        self.val_acc(preds, y)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Log metrics
        preds = logits.argmax(dim=1)
        self.test_acc(preds, y)
        self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        # Only train the classifier (encoder is frozen)
        optimizer = AdamW(
            self.classifier.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        
        # Cosine annealing scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }


if __name__ == "__main__":
    _ = LinearProbeModel()