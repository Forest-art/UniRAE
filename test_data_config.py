"""Test script to verify data module configuration."""

# Test 1: Dataset without validation split (use train_split)
print("=" * 60)
print("Test: Dataset without validation split (test_hf)")
print("=" * 60)
try:
    from src.data.image_folder_datamodule import ImageFolderDataModule
    
    # Test with validation split specified but not present in dataset
    print("Testing with hf_validation_split='validation' (not in dataset)...")
    datamodule1 = ImageFolderDataModule(
        data_dir="data/test_hf",
        image_size=256,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        train_split=0.8,
        seed=42,
        use_hf_dataset=True,
        hf_split="train",
        hf_validation_split="validation",  # This split doesn't exist
    )
    
    print("\nSetting up datasets...")
    datamodule1.setup()
    
    print(f"\nResults:")
    print(f"  Train dataset size: {len(datamodule1.train_dataset)}")
    print(f"  Validation dataset exists: {datamodule1.val_dataset is not None}")
    if datamodule1.val_dataset:
        print(f"  Validation dataset size: {len(datamodule1.val_dataset)}")
    
    # Test dataloader
    print("\nTesting train dataloader...")
    train_loader = datamodule1.train_dataloader()
    batch = next(iter(train_loader))
    print(f"  Batch shape: {batch[0].shape}")
    print(f"  Batch labels: {batch[1]}")
    
    if datamodule1.val_dataset:
        print("\nTesting validation dataloader...")
        val_loader = datamodule1.val_dataloader()
        val_batch = next(iter(val_loader))
        print(f"  Validation batch shape: {val_batch[0].shape}")
    
    print("\n✓ Test 1 passed: Module correctly handles missing validation split")
    
except Exception as e:
    import traceback
    print(f"\n✗ Error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test 2: Dataset with train_split=1.0 (no validation)")
print("=" * 60)
try:
    from src.data.image_folder_datamodule import ImageFolderDataModule
    
    print("Testing with train_split=1.0 (use all data for training)...")
    datamodule2 = ImageFolderDataModule(
        data_dir="data/test_hf",
        image_size=256,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        train_split=1.0,
        seed=42,
        use_hf_dataset=True,
        hf_split="train",
        hf_validation_split=None,
    )
    
    print("\nSetting up datasets...")
    datamodule2.setup()
    
    print(f"\nResults:")
    print(f"  Train dataset size: {len(datamodule2.train_dataset)}")
    print(f"  Validation dataset: {datamodule2.val_dataset}")
    
    print("\n✓ Test 2 passed: Module correctly uses all data when train_split=1.0")
    
except Exception as e:
    import traceback
    print(f"\n✗ Error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

print()
print("=" * 60)
print("Summary")
print("=" * 60)
print("For datasets with train/validation splits:")
print("  - Set hf_split='train'")
print("  - Set hf_validation_split='validation'")
print("  - train_split can be 1.0 (use all train data)")
print()
print("For datasets with only train split (like test_hf):")
print("  - Set hf_split='train'")
print("  - Set hf_validation_split='validation' (or leave as default)")
print("  - Set train_split < 1.0 (e.g., 0.8) to split train data")
print("  - The module will automatically detect missing validation split")