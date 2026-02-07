"""
Lightning Module wrapper for RAE Stage-1 training with GAN and LPIPS losses.
This module wraps the original training logic while using Lightning's framework.
"""

from typing import Any, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .stage1.rae import RAE
from .disc import (
    DiffAug,
    LPIPS,
    build_discriminator,
    hinge_d_loss,
    vanilla_d_loss,
    vanilla_g_loss,
)


class RAELitModule(LightningModule):
    """Lightning Module for RAE Stage-1 training with GAN and LPIPS losses."""

    def __init__(
        self,
        # Model name (for logging)
        name: Optional[str] = None,
        # RAE configuration
        encoder_cls: str = 'Dinov2withNorm',
        encoder_config_path: str = 'facebook/dinov2-with-registers-base',
        encoder_input_size: int = 224,
        encoder_params: dict = None,
        decoder_config_path: str = 'vit_mae-base',
        decoder_patch_size: int = 16,
        pretrained_decoder_path: Optional[str] = None,
        noise_tau: float = 0.8,
        reshape_to_2d: bool = True,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
        # Training configuration
        ema_decay: float = 0.9978,
        clip_grad: float = 0.0,
        # GAN configuration
        disc_weight: float = 0.75,
        perceptual_weight: float = 1.0,
        disc_start_epoch: int = 8,
        disc_upd_start_epoch: int = 6,
        lpips_start_epoch: int = 0,
        max_d_weight: float = 1e4,
        disc_updates: int = 1,
        disc_loss_type: str = "hinge",
        gen_loss_type: str = "vanilla",
        # Discriminator configuration
        disc_arch: dict = None,
        disc_optimizer: dict = None,
        disc_scheduler: dict = None,
        # Image configuration
        image_size: int = 256,
        # Sampling configuration
        sample_every: int = 2500,
        # Compile model
        compile: bool = False,
        # Optimizer configuration
        optimizer: dict = None,
        scheduler: dict = None,
        # Precision
        precision: str = "fp32",
    ) -> None:
        """Initialize RAELitModule."""
        super().__init__()
        
        # Use manual optimization for multiple optimizers
        self.automatic_optimization = False
        
        # Save hyperparameters
        self.save_hyperparameters(logger=False)
        
        # Initialize RAE
        encoder_params = encoder_params or {}
        self.rae = RAE(
            encoder_cls=encoder_cls,
            encoder_config_path=encoder_config_path,
            encoder_input_size=encoder_input_size,
            encoder_params=encoder_params,
            decoder_config_path=decoder_config_path,
            decoder_patch_size=decoder_patch_size,
            pretrained_decoder_path=pretrained_decoder_path,
            noise_tau=noise_tau,
            reshape_to_2d=reshape_to_2d,
            normalization_stat_path=normalization_stat_path,
            eps=eps,
        )
        
        # Compile if requested
        if compile:
            self.rae.encode = torch.compile(self.rae.encode)
            self.rae.forward = torch.compile(self.rae.forward)
        
        # Freeze encoder, train decoder only
        self.rae.encoder.eval()
        self.rae.decoder.train()
        self.rae.encoder.requires_grad_(False)
        self.rae.decoder.requires_grad_(True)
        
        # Create EMA model
        self.ema_model = self._create_ema_model()
        
        # Initialize discriminator
        disc_arch = disc_arch or {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discriminator, self.disc_aug = build_discriminator(disc_arch, device)
        
        # Initialize LPIPS
        self.lpips = LPIPS()
        self.lpips.eval()
        
        # Training state
        # global_step is managed by LightningModule
        self.disc_train = False
        self.use_gan = False
        self.use_lpips = False
        self.clip_grad = clip_grad
        
        # Metrics storage
        self.epoch_metrics: Dict[str, torch.Tensor] = {}
        
        # Configuration
        self.precision = precision

    def _create_ema_model(self) -> nn.Module:
        """Create EMA model from RAE."""
        from copy import deepcopy
        ema = deepcopy(self.rae).eval()
        ema.requires_grad_(False)
        return ema
    
    def _update_ema(self) -> None:
        """Update EMA model."""
        ema_decay = self.hparams.ema_decay
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.rae.parameters()):
                ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)
    
    def _calculate_adaptive_weight(
        self,
        recon_loss: torch.Tensor,
        gan_loss: torch.Tensor,
        layer: nn.Parameter,
    ) -> torch.Tensor:
        """Calculate adaptive weight for GAN loss."""
        recon_grads = torch.autograd.grad(recon_loss, layer, retain_graph=True)[0]
        gan_grads = torch.autograd.grad(gan_loss, layer, retain_graph=True)[0]
        d_weight = torch.norm(recon_grads) / (torch.norm(gan_grads) + 1e-6)
        d_weight = torch.clamp(d_weight, 0.0, self.hparams.max_d_weight)
        return d_weight.detach()
    
    def _select_gan_losses(self):
        """Select GAN loss functions."""
        disc_loss_type = self.hparams.disc_loss_type
        gen_loss_type = self.hparams.gen_loss_type
        
        if disc_loss_type == "hinge":
            disc_loss_fn = hinge_d_loss
        elif disc_loss_type == "vanilla":
            disc_loss_fn = vanilla_d_loss
        else:
            raise ValueError(f"Unsupported discriminator loss '{disc_loss_type}'")
        
        if gen_loss_type == "vanilla":
            gen_loss_fn = vanilla_g_loss
        else:
            raise ValueError(f"Unsupported generator loss '{gen_loss_type}'")
        
        return disc_loss_fn, gen_loss_fn
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RAE."""
        return self.rae(x)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step."""
        images, _ = batch
        
        # Determine which losses to use based on current step
        steps_per_epoch = len(self.trainer.train_dataloader)
        gan_start_step = self.hparams.disc_start_epoch * steps_per_epoch
        disc_update_step = self.hparams.disc_upd_start_epoch * steps_per_epoch
        lpips_start_step = self.hparams.lpips_start_epoch * steps_per_epoch
        
        current_step = self.trainer.global_step
        self.use_gan = current_step >= gan_start_step and self.hparams.disc_weight > 0.0
        train_disc = current_step >= disc_update_step and self.hparams.disc_weight > 0.0
        self.use_lpips = current_step >= lpips_start_step and self.hparams.perceptual_weight > 0.0
        
        # Move images to device
        device = self.device
        images = images.to(device)
        real_normed = images * 2.0 - 1.0
        
        # Training step for generator
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        
        optimizer.zero_grad(set_to_none=True)
        self.discriminator.eval()
        
        # Generator forward pass
        recon = self.rae(images)
        recon_normed = recon * 2.0 - 1.0
        rec_loss = (recon - images).abs().mean()  # L1
        
        if self.use_lpips:
            lpips_loss = self.lpips(real_normed, recon_normed)
        else:
            lpips_loss = rec_loss.new_zeros(())
        
        recon_total = rec_loss + self.hparams.perceptual_weight * lpips_loss
        
        disc_loss_fn, gen_loss_fn = self._select_gan_losses()
        
        if self.use_gan:
            fake_aug = self.disc_aug.aug(recon_normed)
            logits_fake, _ = self.discriminator(fake_aug, None)
            gan_loss = gen_loss_fn(logits_fake)
        else:
            gan_loss = torch.zeros_like(recon_total)
        
        # Calculate adaptive weight and total loss
        if self.use_gan:
            last_layer = self.rae.decoder.decoder_pred.weight
            adaptive_weight = self._calculate_adaptive_weight(
                recon_total, gan_loss, last_layer, self.hparams.max_d_weight
            )
            total_loss = recon_total + self.hparams.disc_weight * adaptive_weight * gan_loss
        else:
            adaptive_weight = torch.zeros_like(recon_total)
            total_loss = recon_total
        
        total_loss.float()
        
        # Backward pass
        self.manual_backward(total_loss)
        
        if self.clip_grad is not None and self.clip_grad > 0:
            self.clip_gradients(optimizer, gradient_clip_val=self.clip_grad, gradient_clip_algorithm="norm")
        
        optimizer.step()
        
        # Update EMA
        self._update_ema()
        
        # Discriminator training
        disc_metrics = {}
        if train_disc:
            disc_optimizer = self.optimizers()[1]
            
            self.rae.eval()
            self.discriminator.train()
            
            for _ in range(self.hparams.disc_updates):
                disc_optimizer.zero_grad(set_to_none=True)
                
                # Fresh forward pass
                with torch.no_grad():
                    recon_disc = self.rae(images)
                    recon_disc_normed = recon_disc * 2.0 - 1.0
                
                # Discretize
                fake_detached = recon_disc_normed.clamp(-1.0, 1.0)
                fake_detached = torch.round((fake_detached + 1.0) * 127.5) / 127.5 - 1.0
                
                fake_input = self.disc_aug.aug(fake_detached)
                real_input = self.disc_aug.aug(real_normed)
                
                logits_fake, logits_real = self.discriminator(fake_input, real_input)
                d_loss = disc_loss_fn(logits_real, logits_fake)
                accuracy = (logits_real > logits_fake).float().mean()
                
                d_loss.float()
                self.manual_backward(d_loss)
                disc_optimizer.step()
                
                disc_metrics = {
                    "disc_loss": d_loss.detach(),
                    "logits_real": logits_real.detach().mean(),
                    "logits_fake": logits_fake.detach().mean(),
                    "disc_accuracy": accuracy.detach(),
                }
            
            self.discriminator.eval()
            self.rae.train()
        
        # Log metrics
        self.log("train/loss_total", total_loss.detach(), prog_bar=True)
        self.log("train/loss_recon", rec_loss.detach())
        self.log("train/loss_lpips", lpips_loss.detach())
        self.log("train/loss_gan", gan_loss.detach())
        
        if disc_metrics:
            self.log("train/disc_loss", disc_metrics["disc_loss"])
            self.log("train/disc_accuracy", disc_metrics["disc_accuracy"])
        
        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        # Generator optimizer
        gen_optimizer_config = self.hparams.optimizer or {}
        gen_optimizer = torch.optim.AdamW(
            self.rae.decoder.parameters(),
            lr=gen_optimizer_config.get("lr", 2e-4),
            betas=gen_optimizer_config.get("betas", [0.9, 0.95]),
            weight_decay=gen_optimizer_config.get("weight_decay", 0.0),
        )
        
        # Discriminator optimizer
        disc_optimizer_config = self.hparams.disc_optimizer or {}
        disc_optimizer = torch.optim.AdamW(
            [p for p in self.discriminator.parameters() if p.requires_grad],
            lr=disc_optimizer_config.get("lr", 2e-4),
            betas=disc_optimizer_config.get("betas", [0.9, 0.95]),
            weight_decay=disc_optimizer_config.get("weight_decay", 0.0),
        )
        
        # Return both optimizers
        return [gen_optimizer, disc_optimizer]
    
    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        """Called after each training batch."""
        # Log learning rates
        opt = self.optimizers(use_pl_optimizer=False)[0]
        self.log("lr/generator", opt.param_groups[0]["lr"])
        
        if len(self.optimizers(use_pl_optimizer=False)) > 1:
            disc_opt = self.optimizers(use_pl_optimizer=False)[1]
            self.log("lr/discriminator", disc_opt.param_groups[0]["lr"])


if __name__ == "__main__":
    _ = RAELitModule()