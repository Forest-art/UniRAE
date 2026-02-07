"""
rFID (reconstruction FID) evaluation script for RAE models.
This script computes the FID score between original and reconstructed images.
"""

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from PIL import Image

from torch_fidelity import calculate_metrics
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3


def save_images_as_numpy(images, output_dir, prefix="img", nrow=8):
    """Save images as individual PNG files and return the directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each image
    for i, img in enumerate(images):
        # Tensor to PIL
        if img.dim() == 3:
            img_pil = Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0))
        else:
            img_pil = make_grid(img.cpu(), nrow=nrow, normalize=True)
            img_pil = Image.fromarray((img_pil.numpy() * 255).astype(np.uint8).transpose(1, 2, 0))
        
        img_pil.save(output_dir / f"{prefix}_{i:06d}.png")
    
    return output_dir


class ImageListDataset:
    """Simple dataset for a list of PIL Images or numpy arrays."""
    def __init__(self, images):
        self.images = images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        # Ensure PIL Image in RGB mode
        if isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[2] == 3:  # HWC
                img = Image.fromarray(img.astype(np.uint8))
            elif img.ndim == 3 and img.shape[0] == 3:  # CHW
                img = Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8))
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


@torch.no_grad()
def evaluate_model_rfid(
    model,
    dataloader,
    num_samples=10000,
    batch_size=64,
    device="cuda",
    output_dir=None,
):
    """
    Evaluate rFID for a RAE model.
    
    Args:
        model: RAE model (RAE or RAELitModule)
        dataloader: DataLoader for original images
        num_samples: Number of samples to evaluate
        batch_size: Batch size for FID computation
        device: Device to use
        output_dir: Optional directory to save sample images
        
    Returns:
        rFID score
    """
    model.eval()
    model = model.to(device)
    
    # Get EMA model if it's a Lightning module
    if hasattr(model, 'ema_model'):
        eval_model = model.ema_model
    else:
        eval_model = model
    
    original_images = []
    reconstructed_images = []
    
    print(f"Collecting {num_samples} image pairs...")
    
    for batch in tqdm(dataloader, desc="Processing"):
        if len(original_images) >= num_samples:
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
            if len(original_images) >= num_samples:
                break
            # NCHW -> HWC
            original_images.append(orig_np[i].transpose(1, 2, 0))
            reconstructed_images.append(recon_np[i].transpose(1, 2, 0))
    
    # Save sample images if requested
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving sample images to {output_dir}")
        
        # Save first 64 pairs
        n_samples = min(64, len(original_images))
        for i in range(n_samples):
            Image.fromarray(original_images[i]).save(output_dir / f"original_{i:04d}.png")
            Image.fromarray(reconstructed_images[i]).save(output_dir / f"recon_{i:04d}.png")
    
    # Compute rFID
    print(f"Computing rFID with {len(original_images)} samples...")
    rfid_score = compute_rfid(original_images, reconstructed_images, batch_size=batch_size, device=device)
    
    return rfid_score


def main():
    parser = argparse.ArgumentParser(description="Evaluate rFID for RAE models")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.ckpt file)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--num-samples", type=int, default=10000,
                        help="Number of samples to evaluate (default: 10000)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for FID computation (default: 64)")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Image size (default: 256)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save sample images (optional)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers (default: 4)")
    
    args = parser.parse_args()
    
    # Import lightning components
    from src.models.rae_module import RAELitModule
    from src.data.image_folder_datamodule import ImageFolderDataModule
    import lightning as L
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = RAELitModule.load_from_checkpoint(args.checkpoint)
    model.eval()
    
    # Setup datamodule
    print(f"Loading data from {args.data_dir}")
    datamodule = ImageFolderDataModule(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup(stage="test")
    
    # Create dataloader (use test or validation)
    dataloader = datamodule.test_dataloader()
    if dataloader is None:
        dataloader = datamodule.val_dataloader()
    
    if dataloader is None:
        # Fallback to train
        print("Warning: No test/val dataloader, using train with validation transform")
        from torchvision import transforms
        val_transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: transforms.functional.center_crop(pil_image, (args.image_size, args.image_size))),
            transforms.ToTensor(),
        ])
        dataloader = datamodule.train_dataloader()
        dataloader.dataset.transform = val_transform
    
    # Evaluate rFID
    rfid_score = evaluate_model_rfid(
        model=model,
        dataloader=dataloader,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
    )
    
    print(f"\n{'='*60}")
    print(f"rFID Score: {rfid_score:.4f}")
    print(f"{'='*60}\n")
    
    if args.output_dir:
        print(f"Sample images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()