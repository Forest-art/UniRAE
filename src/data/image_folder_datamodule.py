"""
DataModule for ImageFolder datasets with custom transformations.
Supports both ImageFolder and HuggingFace dataset formats.
"""

from typing import Optional, Callable, Tuple
from pathlib import Path

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from datasets import load_from_disk


class HFImageNetDataset(Dataset):
    """
    Wrapper for HuggingFace Dataset with PyTorch Dataset interface.
    Handles image mode conversion (Grayscale -> RGB) and error skipping.
    """
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            image = item['image']  # Get PIL Image
            label = item['label']
            
            # ImageNet contains some grayscale images, transforms usually need RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            if self.transform:
                image = self.transform(image)
                
            return image, label

        except Exception as e:
            print(f"Error processing sample at index {idx}: {e}")
            # If current sample fails, try next sample recursively
            return self.__getitem__((idx + 1) % len(self.dataset))


class ImageFolderDataModule(LightningDataModule):
    """DataModule for ImageFolder-style datasets.
    Supports both ImageFolder and HuggingFace dataset formats.
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int = 256,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_split: float = 1.0,
        seed: int = 42,
        use_hf_dataset: bool = False,
        hf_split: str = "train",
        hf_validation_split: Optional[str] = None,
    ) -> None:
        """Initialize ImageFolderDataModule.
        
        Args:
            data_dir: Path to dataset directory. Can be either:
                     - ImageFolder format: directory with class subdirectories
                     - HuggingFace format: path to save_from_disk() output
            image_size: Target image size for training.
            batch_size: Batch size for training.
            num_workers: Number of workers for data loading.
            pin_memory: Whether to pin memory for faster GPU transfer.
            train_split: Fraction of data to use for training (1.0 means use all for training).
            seed: Random seed for reproducibility.
            use_hf_dataset: Whether to use HuggingFace dataset format (True) or ImageFolder (False).
            hf_split: Which split to use for HuggingFace dataset ('train' or 'validation').
            hf_validation_split: Validation split for HuggingFace dataset ('train' and 'validation' are separate splits).
        """
        super().__init__()
        
        self.save_hyperparameters(logger=False)
        
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_split = train_split
        self.seed = seed
        self.use_hf_dataset = use_hf_dataset
        self.hf_split = hf_split
        
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        
        # Define transforms
        self.train_transform = self._get_train_transform()
        self.val_transform = self._get_val_transform()
    
    def _get_train_transform(self) -> Callable:
        """Get training transformations."""
        # First crop size for resizing before random crop
        first_crop_size = 384 if self.image_size == 256 else int(self.image_size * 1.5)
        
        return transforms.Compose([
            transforms.Resize(first_crop_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(self.image_size),
            transforms.ToTensor(),
        ])
    
    def _get_val_transform(self) -> Callable:
        """Get validation transformations."""
        def center_crop_arr(pil_image, image_size):
            pil_image = transforms.functional.center_crop(pil_image, (image_size, image_size))
            return pil_image
        
        return transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, self.image_size)),
            transforms.ToTensor(),
        ])
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Load datasets.
        
        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if self.use_hf_dataset:
            # Load HuggingFace dataset
            self._setup_hf_dataset()
        else:
            # Load ImageFolder dataset
            self._setup_imagefolder_dataset()
    
    def _setup_hf_dataset(self) -> None:
        """Setup HuggingFace dataset."""
        # Load HuggingFace dataset from disk
        hf_dataset = load_from_disk(str(self.data_dir))
        
        # Handle both DatasetDict and single Dataset
        from datasets import DatasetDict
        if isinstance(hf_dataset, DatasetDict):
            # DatasetDict has multiple splits
            if self.hf_split not in hf_dataset:
                raise ValueError(
                    f"Split '{self.hf_split}' not found in dataset at {self.data_dir}. "
                    f"Available: {list(hf_dataset.keys())}"
                )
            
            # Check if using separate validation split
            if self.hf_validation_split is not None:
                if self.hf_validation_split not in hf_dataset:
                    raise ValueError(
                        f"Validation split '{self.hf_validation_split}' not found in dataset at {self.data_dir}. "
                        f"Available: {list(hf_dataset.keys())}"
                    )
                # Use separate train and validation splits
                train_hf = hf_dataset[self.hf_split]
                val_hf = hf_dataset[self.hf_validation_split]
                
                self.train_dataset = HFImageNetDataset(train_hf, transform=self.train_transform)
                self.val_dataset = HFImageNetDataset(val_hf, transform=self.val_transform)
                print(f"[ImageFolderDataModule] Using separate splits: train={len(train_hf)}, val={len(val_hf)}")
            else:
                # Use single split and potentially split it
                current_dataset = hf_dataset[self.hf_split]
                
                # Create wrapped dataset
                if self.train_split < 1.0:
                    # Need to split the dataset
                    total_size = len(current_dataset)
                    train_size = int(total_size * self.train_split)
                    val_size = total_size - train_size
                    
                    # Split the HuggingFace dataset
                    train_hf = current_dataset.select(range(train_size))
                    val_hf = current_dataset.select(range(train_size, total_size))
                    
                    # Create wrapped datasets
                    self.train_dataset = HFImageNetDataset(train_hf, transform=self.train_transform)
                    self.val_dataset = HFImageNetDataset(val_hf, transform=self.val_transform)
                else:
                    # Use all data for training
                    self.train_dataset = HFImageNetDataset(current_dataset, transform=self.train_transform)
                    self.val_dataset = None
        else:
            # Single Dataset - use it directly
            current_dataset = hf_dataset
            
            # Create wrapped dataset
            if self.train_split < 1.0:
                # Need to split the dataset
                total_size = len(current_dataset)
                train_size = int(total_size * self.train_split)
                val_size = total_size - train_size
                
                # Split the HuggingFace dataset
                train_hf = current_dataset.select(range(train_size))
                val_hf = current_dataset.select(range(train_size, total_size))
                
                # Create wrapped datasets
                self.train_dataset = HFImageNetDataset(train_hf, transform=self.train_transform)
                self.val_dataset = HFImageNetDataset(val_hf, transform=self.val_transform)
            else:
                # Use all data for training
                self.train_dataset = HFImageNetDataset(current_dataset, transform=self.train_transform)
                self.val_dataset = None
    
    def _setup_imagefolder_dataset(self) -> None:
        """Setup ImageFolder dataset."""
        # Load full dataset
        full_dataset = ImageFolder(
            root=str(self.data_dir),
            transform=self.train_transform,
        )
        
        # Split into train/val if needed
        if self.train_split < 1.0:
            train_size = int(len(full_dataset) * self.train_split)
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed)
            )
            # Update val dataset transform
            self.val_dataset.dataset.transform = self.val_transform
        else:
            # Use all data for training
            self.train_dataset = full_dataset
            self.val_dataset = None
    
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        if self.val_dataset is None:
            return None
        
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Return test dataloader (uses validation set or training set if no validation)."""
        if self.val_dataset is not None:
            return self.val_dataloader()
        # If no validation set, use training data with validation transform for testing
        test_dataset = HFImageNetDataset(
            self.train_dataset.dataset if isinstance(self.train_dataset, HFImageNetDataset) 
            else self.train_dataset.dataset if hasattr(self.train_dataset, 'dataset')
            else self.train_dataset,
            transform=self.val_transform
        )
        return DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )


if __name__ == "__main__":
    _ = ImageFolderDataModule()