"""
DiT Lightning Module for Stage 2 training.
Integrates RAE encoder and DiT/DDT model for latent diffusion training.
"""

import math
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
import lightning as L
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from src.models.stage1.rae import RAE
from src.models.stage2 import DiTwDDTHead, LightningDiT
from src.models.stage1 import RAE


def update_ema(
    ema_model: nn.Module, model: nn.Module, decay: float = 0.9999, sharded: bool = False
):
    """
    Step the EMA model towards the current model.
    """
    if sharded:
        # Handle sharded parameters if needed
        pass
    else:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


class DiTModule(L.LightningModule):
    """
    PyTorch Lightning module for DiT/DDT training.
    
    This module trains a diffusion model on RAE latents.
    The RAE encoder is frozen, and the diffusion model is trained.
    """
    
    def __init__(
        self,
        rae: nn.Module,
        dit: nn.Module,
        rae_checkpoint_path: Optional[str] = None,
        ema_decay: float = 0.9999,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        betas: tuple = (0.9, 0.95),
        warmup_steps: int = 5000,
        max_steps: int = 100000,
        num_classes: int = 1000,
        null_label: int = 1000,
        latent_size: tuple = (768, 16, 16),
        compile: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize RAE encoder (frozen)
        self.rae = rae
        self.rae.eval()
        self.rae.requires_grad_(False)
        
        # Load RAE checkpoint if provided
        if rae_checkpoint_path:
            self._load_rae_checkpoint(rae_checkpoint_path)
        
        # Initialize DiT model (trainable)
        self.dit = dit
        
        # Compile model if requested
        if compile:
            try:
                self.rae.encode = torch.compile(self.rae.encode)
                print("RAE encode compiled successfully")
            except Exception as e:
                print(f"Failed to compile RAE encode: {e}")
            
            try:
                self.dit.forward = torch.compile(self.dit.forward)
                print("DiT forward compiled successfully")
            except Exception as e:
                print(f"Failed to compile DiT forward: {e}")
        
        # Initialize EMA model
        self.ema_dit = deepcopy(self.dit)
        self.ema_dit.eval()
        self.ema_dit.requires_grad_(False)
        
        # Hyperparameters
        self.ema_decay = ema_decay
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.num_classes = num_classes
        self.null_label = null_label
        self.latent_size = latent_size
        
        # Transport parameters (will be set in configure_optimizers)
        self.transport = None
        
    def _load_rae_checkpoint(self, checkpoint_path: str):
        """Load RAE checkpoint."""
        print(f"Loading RAE checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Try different checkpoint keys
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "ema" in checkpoint:
            state_dict = checkpoint["ema"]
        else:
            state_dict = checkpoint
        
        # Load state dict
        self.rae.load_state_dict(state_dict, strict=False)
        print("RAE checkpoint loaded successfully")
    
    def forward(self, x, t, y):
        """Forward pass of DiT model."""
        return self.dit(x, t, y)
    
    def forward_with_cfg(self, x, t, y, cfg_scale=1.0):
        """Forward pass with classifier-free guidance."""
        if hasattr(self.dit, "forward_with_cfg"):
            return self.dit.forward_with_cfg(x, t, y, cfg_scale)
        else:
            # Fallback to normal forward
            return self.dit(x, t, y)
    
    def encode(self, images):
        """Encode images to latents using RAE encoder."""
        return self.rae.encode(images)
    
    def decode(self, latents):
        """Decode latents to images using RAE decoder."""
        return self.rae.decode(latents)
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        # Only optimize DiT parameters
        optimizer = AdamW(
            self.dit.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        
        # Warmup + cosine decay scheduler
        def lr_lambda(step):
            if step < self.warmup_steps:
                # Linear warmup
                return float(step) / float(max(1, self.warmup_steps))
            else:
                # Cosine decay
                progress = float(step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        images, labels = batch
        batch_size = images.shape[0]
        
        # Encode images to latents
        with torch.no_grad():
            latents = self.encode(images)  # (B, C, H, W)
        
        # Sample random timesteps
        t = torch.rand(batch_size, device=self.device)  # Uniform in [0, 1)
        
        # Sample random noise
        noise = torch.randn_like(latents)
        
        # Simple diffusion model (predict noise)
        # This is a simplified version - in practice, you'd use a proper transport
        # For now, we'll train to predict the noise
        timesteps = t * 1000  # Scale to [0, 1000]
        
        # Add noise to latents
        noisy_latents = latents + noise * t.view(-1, 1, 1, 1)
        
        # Predict noise
        pred_noise = self.dit(noisy_latents, timesteps, labels)
        
        # Compute loss (MSE between predicted and actual noise)
        loss = nn.functional.mse_loss(pred_noise, noise)
        
        # Log metrics
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"], on_step=True)
        
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA after each batch."""
        update_ema(self.ema_dit, self.dit, self.ema_decay)
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images, labels = batch
        
        # Encode images
        with torch.no_grad():
            latents = self.encode(images)
        
        # Sample random timesteps
        t = torch.rand(images.shape[0], device=self.device)
        noise = torch.randn_like(latents)
        
        # Add noise and predict
        noisy_latents = latents + noise * t.view(-1, 1, 1, 1)
        pred_noise = self.dit(noisy_latents, t * 1000, labels)
        
        # Compute loss
        loss = nn.functional.mse_loss(pred_noise, noise)
        
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def sample(self, num_samples: int, labels: Optional[torch.Tensor] = None, cfg_scale: float = 1.0):
        """Generate samples using the diffusion model."""
        self.eval()
        
        if labels is None:
            labels = torch.randint(0, self.num_classes, (num_samples,), device=self.device)
        
        # Start from random noise
        latents = torch.randn(num_samples, *self.latent_size, device=self.device)
        
        # Simple sampling loop (in practice, use proper ODE/SDE sampler)
        num_steps = 100
        for i in range(num_steps):
            t = torch.linspace(1.0, 0.0, num_steps)[i]
            timesteps = t * 1000
            
            with torch.no_grad():
                # Predict and denoise
                if cfg_scale > 1.0 and hasattr(self.ema_dit, "forward_with_cfg"):
                    # Use classifier-free guidance
                    # Duplicate latents for CFG
                    latents_dup = torch.cat([latents, latents], dim=0)
                    labels_dup = torch.cat([labels, torch.full_like(labels, self.null_label)], dim=0)
                    pred = self.ema_dit.forward_with_cfg(latents_dup, timesteps, labels_dup, cfg_scale)
                    # Take conditional half
                    pred = pred[:num_samples]
                else:
                    pred = self.ema_dit(latents, timesteps, labels)
                
                # Simple denoising step
                latents = latents - pred * (1.0 / num_steps)
        
        # Decode latents to images
        with torch.no_grad():
            images = self.decode(latents)
        
        return images


class DiTModuleWithTransport(DiTModule):
    """
    DiT module with proper transport (e.g., ODE transport) for more advanced training.
    This is a placeholder for when transport is fully implemented.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: Initialize transport here
        self.transport = None
    
    def training_step(self, batch, batch_idx):
        """Training step with transport."""
        images, labels = batch
        batch_size = images.shape[0]
        
        # Encode images
        with torch.no_grad():
            latents = self.encode(images)
        
        # Use transport to sample and compute losses
        # TODO: Implement proper transport training
        if self.transport is not None:
            loss_dict = self.transport.training_losses(self.dit, latents, {"y": labels})
            loss = loss_dict["loss"].mean()
        else:
            # Fallback to simple diffusion
            t = torch.rand(batch_size, device=self.device)
            noise = torch.randn_like(latents)
            noisy_latents = latents + noise * t.view(-1, 1, 1, 1)
            pred_noise = self.dit(noisy_latents, t * 1000, labels)
            loss = nn.functional.mse_loss(pred_noise, noise)
        
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"], on_step=True)
        
        return loss