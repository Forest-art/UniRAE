"""
DiT/DDT sampling script.

This script generates samples using a trained DiT/DDT model.
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import lightning as L
from omegaconf import DictConfig, OmegaConf
import hydra

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.dit_module import DiTModule
from src.models.stage1.rae import RAE


def load_model_from_checkpoint(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device: str = "cuda",
):
    """
    Load DiT model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file (optional)
        device: Device to load model on
        
    Returns:
        Loaded DiTModule
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Try to extract config from checkpoint
    if "cfg" in checkpoint:
        cfg = checkpoint["cfg"]
        print(f"Loaded config from checkpoint")
    elif config_path:
        with open(config_path, "r") as f:
            cfg = OmegaConf.load(f)
        print(f"Loaded config from {config_path}")
    else:
        raise ValueError("No config found in checkpoint and no config_path provided")
    
    # Instantiate model from config
    model = hydra.utils.instantiate(cfg.model)
    
    # Load state dict
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "ema_dit_state_dict" in checkpoint:
        # Load EMA weights
        model.ema_dit.load_state_dict(checkpoint["ema_dit_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    return model


def sample_dit(
    model: DiTModule,
    num_samples: int,
    labels: Optional[torch.Tensor] = None,
    cfg_scale: float = 1.0,
    num_steps: int = 100,
    device: str = "cuda",
    class_names: Optional[list] = None,
) -> tuple:
    """
    Generate samples using DiT model.
    
    Args:
        model: DiTModule
        num_samples: Number of samples to generate
        labels: Class labels for conditional generation (optional)
        cfg_scale: Classifier-free guidance scale
        num_steps: Number of sampling steps
        device: Device to generate on
        class_names: List of class names for saving
        
    Returns:
        Tuple of (images, labels)
    """
    print(f"Generating {num_samples} samples with CFG scale {cfg_scale}...")
    
    model.eval()
    
    # Generate labels if not provided
    if labels is None:
        labels = torch.randint(0, model.num_classes, (num_samples,), device=device)
    else:
        labels = labels.to(device)
        assert len(labels) == num_samples
    
    # Start from random noise
    latents = torch.randn(num_samples, *model.latent_size, device=device)
    
    # Sampling loop
    with torch.no_grad():
        for i in tqdm(range(num_steps), desc="Sampling"):
            t = 1.0 - (i / num_steps)  # Go from 1.0 to 0.0
            timesteps = torch.full((num_samples,), t * 1000, device=device, dtype=torch.long)
            
            # Predict noise
            if cfg_scale > 1.0 and hasattr(model.ema_dit, "forward_with_cfg"):
                # Classifier-free guidance
                latents_dup = torch.cat([latents, latents], dim=0)
                labels_dup = torch.cat([labels, torch.full_like(labels, model.null_label)], dim=0)
                pred = model.ema_dit.forward_with_cfg(latents_dup, timesteps, labels_dup, cfg_scale)
                pred = pred[:num_samples]  # Take conditional half
            else:
                pred = model.ema_dit(latents, timesteps, labels)
            
            # Simple denoising step (Euler)
            dt = 1.0 / num_steps
            latents = latents - pred * dt
    
    # Decode latents to images
    print("Decoding latents to images...")
    with torch.no_grad():
        images = model.decode(latents)
    
    # Convert to numpy and denormalize
    images = images.cpu().numpy()
    images = np.clip(images * 255, 0, 255).astype(np.uint8)
    
    return images, labels.cpu().numpy()


def save_images(
    images: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    class_names: Optional[list] = None,
    nrow: int = 8,
):
    """
    Save generated images.
    
    Args:
        images: Array of images (N, C, H, W)
        labels: Array of labels (N,)
        output_dir: Directory to save images
        class_names: List of class names (optional)
        nrow: Number of images per row in grid
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual images
    for i, (img, label) in enumerate(zip(images, labels)):
        img_pil = Image.fromarray(img.transpose(1, 2, 0))
        class_name = class_names[label] if class_names and label < len(class_names) else f"class_{label}"
        img_pil.save(output_dir / f"sample_{i:05d}_class_{label:04d}_{class_name}.png")
    
    # Create grid of images
    ncol = nrow
    nrows = (len(images) + ncol - 1) // ncol
    
    grid_width = images.shape[2] * ncol
    grid_height = images.shape[1] * nrows
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    for i, img in enumerate(images):
        row = i // ncol
        col = i % ncol
        y_start = row * images.shape[1]
        x_start = col * images.shape[2]
        grid[y_start:y_start + images.shape[1], x_start:x_start + images.shape[2]] = img.transpose(1, 2, 0)
    
    grid_pil = Image.fromarray(grid)
    grid_pil.save(output_dir / "grid.png")
    
    print(f"Saved {len(images)} images to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="DiT/DDT sampling script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="outputs/dit_samples", help="Output directory")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-free guidance scale")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of sampling steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--labels", type=str, default=None, 
                        help="Comma-separated class labels (e.g., '0,1,2')")
    parser.add_argument("--class_list", type=str, default=None,
                        help="Path to class list file (one class name per line)")
    
    args = parser.parse_args()
    
    # Load class names if provided
    class_names = None
    if args.class_list and os.path.exists(args.class_list):
        with open(args.class_list, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(class_names)} class names from {args.class_list}")
    
    # Parse labels
    labels = None
    if args.labels:
        labels = torch.tensor([int(x) for x in args.labels.split(",")])
        args.num_samples = len(labels)
        print(f"Using labels: {labels.tolist()}")
    
    # Load model
    model = load_model_from_checkpoint(
        args.checkpoint,
        args.config,
        args.device,
    )
    
    # Generate samples
    images, label_array = sample_dit(
        model,
        args.num_samples,
        labels,
        args.cfg_scale,
        args.num_steps,
        args.device,
        class_names,
    )
    
    # Save images
    save_images(images, label_array, args.output_dir, class_names)
    
    print("Sampling complete!")


if __name__ == "__main__":
    main()