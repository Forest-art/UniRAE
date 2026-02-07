"""
Convert test_imagenet folder to HuggingFace dataset format for testing.
"""

from pathlib import Path
from datasets import Dataset, DatasetDict, Image, Features, Value
import json


def create_hf_dataset():
    """Convert ImageFolder structure to HuggingFace dataset."""
    
    # Paths
    source_dir = Path("test_imagenet")
    output_dir = Path("data/test_hf")
    
    if not source_dir.exists():
        print(f"Error: {source_dir} not found!")
        return
    
    # Collect all images with labels
    data = {"image": [], "label": []}
    label_map = {}
    
    # Iterate through class directories
    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        label_map[class_name] = len(label_map)
        
        for img_path in class_dir.glob("*.jpg"):
            data["image"].append(str(img_path))
            data["label"].append(label_map[class_name])
    
    print(f"Found {len(data['image'])} images across {len(label_map)} classes")
    
    # Create HuggingFace dataset with proper features
    features = Features({
        "image": Image(),
        "label": Value("int32")
    })
    dataset = Dataset.from_dict(data, features=features)
    
    # Convert to DatasetDict format
    dataset_dict = DatasetDict({
        "train": dataset,
    })
    
    # Save to disk
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))
    
    # Save label map
    with open(output_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    
    print(f"Dataset saved to {output_dir}")
    print(f"Label map: {label_map}")
    
    # Test loading
    print("\nTesting load_from_disk...")
    from datasets import load_from_disk
    loaded = load_from_disk(str(output_dir))
    print(f"Successfully loaded: {len(loaded['train'])} samples")
    print(f"First sample: {loaded['train'][0]}")


if __name__ == "__main__":
    create_hf_dataset()
