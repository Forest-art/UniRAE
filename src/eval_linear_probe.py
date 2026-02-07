"""Evaluate encoder representations using linear probing."""

import argparse
from pathlib import Path
import torch
from lightning import Trainer, seed_everything
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchmetrics import Accuracy
import numpy as np
from tqdm import tqdm

from src.models.linear_probe import LinearProbeModel
from src.models.stage1.encoders.dinov2 import Dinov2withNorm


def create_test_dataset(data_dir: str, image_size: int = 224):
    """Create test dataset for linear probing evaluation."""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = ImageFolder(data_dir, transform=transform)
    return dataset


def extract_features(encoder, dataloader, device, pool_type="avg"):
    """Extract features from encoder."""
    encoder.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            targets = targets.to(device)
            
            # Extract features
            feats = encoder(images)  # (B, num_patches, D) or (B, D)
            
            # Pool features
            if pool_type == "cls":
                feats = feats[:, 0]  # (B, D)
            elif pool_type == "avg":
                feats = feats.mean(dim=1)  # (B, D)
            else:  # flatten
                feats = feats.view(images.shape[0], -1)
            
            # Normalize
            feats = torch.nn.functional.normalize(feats, dim=-1)
            
            features.append(feats.cpu())
            labels.append(targets.cpu())
    
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    
    return features, labels


def train_linear_probe(features, labels, num_classes, lr=1e-3, epochs=90):
    """Train linear probe on extracted features."""
    feature_dim = features.shape[1]
    classifier = torch.nn.Linear(feature_dim, num_classes)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Use small validation split
    num_samples = features.shape[0]
    val_size = num_samples // 10
    train_size = num_samples - val_size
    
    indices = np.random.permutation(num_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    val_features = features[val_indices]
    val_labels = labels[val_indices]
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Train
        classifier.train()
        optimizer.zero_grad()
        logits = classifier(train_features)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()
        
        # Validate
        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_features)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_labels).float().mean().item()
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = classifier.state_dict()
    
    print(f"Best validation accuracy: {best_acc:.4f}")
    
    # Load best model
    classifier.load_state_dict(best_state)
    
    return classifier, best_acc


def main(args):
    """Main evaluation function."""
    seed_everything(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load encoder
    print(f"Loading encoder from {args.encoder_checkpoint}")
    encoder = Dinov2withNorm(
        dinov2_path=args.encoder_config_path,
        normalize=False,
    )
    
    # Load encoder weights
    if args.encoder_checkpoint is not None:
        state_dict = torch.load(args.encoder_checkpoint, map_location="cpu")
        if "encoder_state_dict" in state_dict:
            encoder_state = state_dict["encoder_state_dict"]
        else:
            encoder_state = {}
            for k, v in state_dict.items():
                if k.startswith("encoder."):
                    encoder_state[k[8:]] = v
        
        encoder.load_state_dict(encoder_state, strict=False)
    else:
        print("Using pretrained encoder from HuggingFace")
    
    encoder = encoder.to(device)
    encoder.eval()
    
    # Create dataset
    print(f"Loading dataset from {args.data_dir}")
    dataset = create_test_dataset(args.data_dir, args.image_size)
    num_classes = len(dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Dataset size: {len(dataset)}")
    
    # Limit samples if specified
    if args.num_samples > 0:
        indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)), replace=False)
        dataset = Subset(dataset, indices)
        print(f"Using {len(dataset)} samples")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Extract features
    features, labels = extract_features(encoder, dataloader, device, args.pool_type)
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Train linear probe
    print("\nTraining linear probe...")
    classifier, best_acc = train_linear_probe(
        features, labels, num_classes, args.lr, args.epochs
    )
    
    # Final test
    print("\nFinal evaluation...")
    classifier.eval()
    with torch.no_grad():
        logits = classifier(features)
        preds = logits.argmax(dim=1)
        test_acc = (preds == labels).float().mean().item()
    
    print(f"\nLinear Probe Results:")
    print(f"  Number of classes: {num_classes}")
    print(f"  Number of samples: {len(features)}")
    print(f"  Best validation accuracy: {best_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    
    # Save results
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / "linear_probe_results.txt"
        with open(results_file, "w") as f:
            f.write(f"Linear Probe Results\n")
            f.write(f"===================\n\n")
            f.write(f"Encoder checkpoint: {args.encoder_checkpoint}\n")
            f.write(f"Dataset: {args.data_dir}\n")
            f.write(f"Number of classes: {num_classes}\n")
            f.write(f"Number of samples: {len(features)}\n")
            f.write(f"Pooling type: {args.pool_type}\n")
            f.write(f"Learning rate: {args.lr}\n")
            f.write(f"Epochs: {args.epochs}\n\n")
            f.write(f"Best validation accuracy: {best_acc:.4f}\n")
            f.write(f"Test accuracy: {test_acc:.4f}\n")
        
        print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate encoder with linear probing")
    
    # Encoder
    parser.add_argument(
        "--encoder_checkpoint",
        type=str,
        default=None,
        help="Path to encoder checkpoint (null for pretrained HuggingFace model)",
    )
    parser.add_argument(
        "--encoder_config_path",
        type=str,
        default="facebook/dinov2-with-registers-base",
        help="HuggingFace model path or config",
    )
    
    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to ImageNet validation dataset",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Image size",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="Number of samples to use (0 for all)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of data loading workers",
    )
    
    # Training
    parser.add_argument(
        "--pool_type",
        type=str,
        default="avg",
        choices=["avg", "cls", "flatten"],
        help="Pooling type for features",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for linear probe",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=90,
        help="Number of epochs",
    )
    
    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    main(args)