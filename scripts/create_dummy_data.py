"""
Create complete dummy dataset with train/val/test splits for testing.
This dataset will be used for all algorithm testing and debugging.
"""

from pathlib import Path
import shutil
from datasets import Dataset, DatasetDict, Image, Features, Value
import json
import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm


def create_dummy_images(output_dir: Path, num_classes: int = 10, images_per_class: int = 20, image_size: int = 256):
    """Create dummy images with random colors."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating dummy images...")
    print(f"  - Classes: {num_classes}")
    print(f"  - Images per class: {images_per_class}")
    print(f"  - Total images: {num_classes * images_per_class}")
    
    for class_idx in range(num_classes):
        class_name = f"class_{class_idx:03d}"
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        for img_idx in tqdm(range(images_per_class), desc=f"Class {class_name}"):
            # Create random color image
            # Use deterministic random based on class and image index for reproducibility
            np.random.seed(class_idx * 1000 + img_idx)
            img_array = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
            img = PILImage.fromarray(img_array)
            
            img_path = class_dir / f"img_{img_idx:04d}.jpg"
            img.save(img_path)
    
    print(f"✓ Dummy images created in {output_dir}")


def create_hf_dataset_from_folder(source_dir: Path, output_dir: Path, splits: dict):
    """Convert ImageFolder to HuggingFace dataset with specified splits."""
    
    if not source_dir.exists():
        print(f"Error: {source_dir} not found!")
        return
    
    # Collect all images with labels
    all_images = []
    all_labels = []
    label_map = {}
    
    print(f"\nScanning {source_dir}...")
    
    # Iterate through class directories
    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        label_map[class_name] = len(label_map)
        
        images = list(class_dir.glob("*.jpg"))
        for img_path in images:
            all_images.append(str(img_path))
            all_labels.append(label_map[class_name])
    
    print(f"Found {len(all_images)} images across {len(label_map)} classes")
    
    # Shuffle and split
    indices = np.random.RandomState(42).permutation(len(all_images))
    
    splits_data = {}
    for split_name, split_ratio in splits.items():
        if split_name == "train":
            start_idx = 0
            end_idx = int(len(all_images) * split_ratio)
        elif split_name == "validation":
            start_idx = int(len(all_images) * splits["train"])
            end_idx = start_idx + int(len(all_images) * split_ratio)
        else:  # test
            start_idx = int(len(all_images) * (splits["train"] + splits["validation"]))
            end_idx = len(all_images)
        
        split_indices = indices[start_idx:end_idx]
        splits_data[split_name] = {
            "image": [all_images[i] for i in split_indices],
            "label": [all_labels[i] for i in split_indices],
        }
        
        print(f"  {split_name}: {len(split_indices)} samples")
    
    # Create HuggingFace datasets
    features = Features({
        "image": Image(),
        "label": Value("int32")
    })
    
    datasets = {}
    for split_name, data in splits_data.items():
        datasets[split_name] = Dataset.from_dict(data, features=features)
    
    # Create DatasetDict
    dataset_dict = DatasetDict(datasets)
    
    # Save to disk
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))
    
    # Save label map and metadata
    metadata = {
        "label_map": label_map,
        "splits": splits,
        "num_classes": len(label_map),
        "total_samples": len(all_images),
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Dataset saved to {output_dir}")
    print(f"  - Label map: {label_map}")
    print(f"  - Metadata saved")
    
    return dataset_dict


def main():
    """Main function to create complete dummy dataset."""
    
    # Configuration
    num_classes = 10
    images_per_class = 60  # Total 600 images
    image_size = 256
    
    # Splits: 60% train, 20% validation, 20% test
    splits = {
        "train": 0.6,
        "validation": 0.2,
        "test": 0.2,
    }
    
    print("=" * 60)
    print("Creating Complete Dummy Dataset for Testing")
    print("=" * 60)
    
    # Paths
    raw_dir = Path("data/dummy_images_raw")
    hf_output_dir = Path("data/dummy_dataset_hf")
    
    # Step 1: Create dummy images
    create_dummy_images(raw_dir, num_classes, images_per_class, image_size)
    
    # Step 2: Convert to HuggingFace dataset with splits
    dataset_dict = create_hf_dataset_from_folder(raw_dir, hf_output_dir, splits)
    
    # Step 3: Test loading
    print("\n" + "=" * 60)
    print("Testing Dataset Loading")
    print("=" * 60)
    
    from datasets import load_from_disk
    loaded = load_from_disk(str(hf_output_dir))
    
    for split_name in ["train", "validation", "test"]:
        if split_name in loaded:
            print(f"\n{split_name} split:")
            print(f"  - Samples: {len(loaded[split_name])}")
            print(f"  - First sample: {loaded[split_name][0]}")
            print(f"  - Image shape: {loaded[split_name][0]['image'].size}")
    
    # Step 4: Verify metadata
    with open(hf_output_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    print("\n" + "=" * 60)
    print("Dataset Metadata")
    print("=" * 60)
    print(f"Total samples: {metadata['total_samples']}")
    print(f"Number of classes: {metadata['num_classes']}")
    print(f"Splits: {metadata['splits']}")
    
    # Step 5: Print usage instructions
    print("\n" + "=" * 60)
    print("Usage Instructions")
    print("=" * 60)
    print("\nIn your data config (configs/data/imagenet.yaml):")
    print(f"  data_dir: {hf_output_dir}")
    print(f"  use_hf_dataset: true")
    print(f"  hf_split: train")
    print(f"  hf_validation_split: validation")
    print(f"  hf_test_split: test")
    
    print("\nFor training with dummy data:")
    print("  python src/train.py experiment=rae_dino \\")
    print(f"    data.data_dir={hf_output_dir} \\")
    print("    data.batch_size=8 \\")
    print("    model.num_epochs=2")
    
    print("\n✓ Dummy dataset created successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()