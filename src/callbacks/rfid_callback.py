"""
Callback for evaluating rFID during training.
Evaluates rFID at specified intervals (e.g., every N steps or every epoch).
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

from torch_fidelity import calculate_metrics


class ImageListDataset(Dataset):
    """Simple dataset for a list of PIL Images or numpy arrays.

    Note: torch_fidelity requires the dataset to return torch.Tensor with dtype uint8.
    """
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        # Convert to torch.Tensor (C, H, W) format with dtype uint8
        if isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[2] == 3:  # HWC
                img = torch.tensor(img.astype(np.uint8)).permute(2, 0, 1)
            elif img.ndim == 3 and img.shape[0] == 3:  # CHW
                img = torch.tensor(img.astype(np.uint8))
        elif isinstance(img, Image.Image):
            img = np.array(img)
            img = torch.tensor(img.astype(np.uint8)).permute(2, 0, 1)
        return img


def compute_rfid(original_images, reconstructed_images, batch_size=64, device="cuda"):
    """
    Compute rFID (reconstruction FID) between original and reconstructed images.
    
    Args:
        original_images: List or array of original images (PIL or numpy)
        reconstructed_images: List or array of reconstructed images
        batch_size: Batch size for feature extraction
        device: Device to use for computation
        
    Returns:
        rFID score (lower is better)
    """
    # Create datasets
    original_ds = ImageListDataset(original_images)
    recon_ds = ImageListDataset(reconstructed_images)
    
    # Calculate FID using torch_fidelity
    metrics = calculate_metrics(
        input1=original_ds,
        input2=recon_ds,
        batch_size=batch_size,
        fid=True,
        cuda=(device == "cuda"),
    )
    
    return metrics["frechet_inception_distance"]


class RFIDCallback(Callback):
    """
    Callback to evaluate rFID during training.
    
    Evaluates rFID at specified intervals:
    - Every N steps (if rfid_every_n_steps > 0)
    - Every epoch (if rfid_every_epoch = True)
    """
    
    def __init__(
        self,
        rfid_every_n_steps: int = 1000,
        rfid_every_epoch: bool = True,
        rfid_num_samples: Optional[int] = 1000,
        rfid_batch_size: int = 64,
        rfid_device: str = "cuda",
        rfid_output_dir: Optional[str] = None,
        rfid_save_samples: bool = False,
        save_samples_count: int = 64,
        use_train_dataloader: bool = False,
        use_full_validation_set: bool = False,
    ):
        """
        Initialize RFIDEvalCallback.
        
        Args:
            rfid_every_n_steps: Evaluate rFID every N training steps (0 to disable)
            rfid_every_epoch: Evaluate rFID at the end of each epoch
            rfid_num_samples: Number of samples to use for rFID evaluation (None or use_full_validation_set=True to use full dataset)
            rfid_batch_size: Batch size for FID computation
            rfid_device: Device to use for computation
            rfid_output_dir: Directory to save sample images (None to disable)
            rfid_save_samples: Whether to save sample images (for debugging/visualization, does not affect rFID computation)
            save_samples_count: Number of sample pairs to save
            use_train_dataloader: Use train dataloader for rFID if validation dataloader is not available (default False)
            use_full_validation_set: If True, use all samples from validation set regardless of rfid_num_samples; if False, use rfid_num_samples (default False)
        """
        super().__init__()
        self.rfid_every_n_steps = rfid_every_n_steps
        self.rfid_every_epoch = rfid_every_epoch
        self.rfid_num_samples = rfid_num_samples
        self.rfid_batch_size = rfid_batch_size
        self.rfid_device = rfid_device
        self.rfid_output_dir = rfid_output_dir
        self.rfid_save_samples = rfid_save_samples
        self.save_samples_count = save_samples_count
        self.use_train_dataloader = use_train_dataloader
        self.use_full_validation_set = use_full_validation_set
    
    def on_train_start(self, trainer, pl_module):
        """Called when training starts."""
        self.last_eval_step = 0
        print(f"[RFIDCallback] rFID evaluation enabled: every {self.rfid_every_n_steps} steps, every epoch: {self.rfid_every_epoch}")
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called after each training batch."""
        # Skip if not the main process (in DDP)
        if trainer.is_global_zero is False:
            return
        
        # Skip if rfid_every_n_steps is 0
        if self.rfid_every_n_steps <= 0:
            return
        
        # Check if it's time to evaluate
        current_step = trainer.global_step
        if current_step > 0 and current_step % self.rfid_every_n_steps == 0 and current_step != self.last_eval_step:
            self._evaluate_rfid(trainer, pl_module, f"step_{current_step}")
            self.last_eval_step = current_step
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of each training epoch."""
        # Skip if not the main process (in DDP)
        if trainer.is_global_zero is False:
            return
        
        # Skip if rfid_every_epoch is False
        if not self.rfid_every_epoch:
            return
        
        epoch = trainer.current_epoch
        self._evaluate_rfid(trainer, pl_module, f"epoch_{epoch}")
    
    def _evaluate_rfid(self, trainer, pl_module, eval_name: str):
        """Evaluate rFID and log results."""
        print(f"\n[RFIDCallback] Starting rFID evaluation at {eval_name}...")
        
        # Get model
        model = pl_module
        
        # Get EMA model if available
        if hasattr(model, 'ema_model'):
            eval_model = model.ema_model
        else:
            eval_model = model
        
        eval_model.eval()
        
        # Get validation dataloader (required)
        if trainer.val_dataloaders is not None:
            dataloader = trainer.val_dataloaders
            dataloader_type = "validation"
            print(f"[RFIDCallback] Using {dataloader_type} set for rFID evaluation")
        else:
            print("[RFIDCallback] Error: Validation dataloader not found. rFID evaluation requires a validation set.")
            print("[RFIDCallback] Please ensure your datamodule has a validation set configured.")
            return
        
        # Collect images
        original_images = []
        reconstructed_images = []
        
        device = self.rfid_device
        
        # Determine if we should limit samples
        limit_samples = not self.use_full_validation_set and self.rfid_num_samples is not None
        max_samples = self.rfid_num_samples if limit_samples else float('inf')
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Collecting samples for {eval_name} ({dataloader_type} set)", leave=False):
                # Stop if we've collected enough samples (only when limiting)
                if limit_samples and len(original_images) >= max_samples:
                    break
                
                images, _ = batch
                images = images.to(device)
                
                # Forward pass
                recon = eval_model(images)
                
                # Convert to numpy (0-255, uint8)
                orig_np = (images.cpu().numpy() * 255).astype(np.uint8)
                recon_np = (recon.cpu().numpy() * 255).astype(np.uint8)
                
                # Append individual images
                for i in range(images.size(0)):
                    if limit_samples and len(original_images) >= max_samples:
                        break
                    # NCHW -> HWC
                    original_images.append(orig_np[i].transpose(1, 2, 0))
                    reconstructed_images.append(recon_np[i].transpose(1, 2, 0))
        
        # Save sample images if requested
        if self.rfid_save_samples and self.rfid_output_dir is not None:
            output_dir = Path(self.rfid_output_dir) / eval_name
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"[RFIDCallback] Saving sample images to {output_dir}")
            
            # Save first N pairs
            n_samples = min(self.save_samples_count, len(original_images))
            for i in range(n_samples):
                Image.fromarray(original_images[i]).save(output_dir / f"original_{i:04d}.png")
                Image.fromarray(reconstructed_images[i]).save(output_dir / f"recon_{i:04d}.png")
        
        # Compute rFID
        print(f"[RFIDCallback] Computing rFID with {len(original_images)} samples...")
        rfid_score = compute_rfid(
            original_images,
            reconstructed_images,
            batch_size=self.rfid_batch_size,
            device=device,
        )
        
        # Log results
        pl_module.log(f"rfid/score", rfid_score, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        pl_module.log(f"rfid/{eval_name}", rfid_score, prog_bar=False, on_step=False, on_epoch=True, logger=True)
        
        print(f"[RFIDCallback] rFID at {eval_name}: {rfid_score:.4f}\n")