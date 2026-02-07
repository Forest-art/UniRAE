"""
Simple test script to verify RAE imports work correctly.
Run this before starting training to check for any import issues.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all RAE modules can be imported."""
    print("Testing RAE imports...")
    
    try:
        print("  - Importing models...")
        from models.stage1.rae import RAE
        print("    ✓ RAE imported successfully")
        
        from models.stage1.encoders.dinov2 import Dinov2withNorm
        print("    ✓ DINO encoder imported successfully")
        
        from models.stage1.encoders.mae import MaeEncoder
        print("    ✓ MAE encoder imported successfully")
        
        from models.stage1.encoders.siglip2 import SigLIP2
        print("    ✓ SigLIP2 encoder imported successfully")
        
        from models.stage1.decoders.decoder import Decoder
        print("    ✓ Decoder imported successfully")
        
        print("  - Importing discriminator...")
        from models.components.disc import (
            DiffAug,
            LPIPS,
            build_discriminator,
            hinge_d_loss,
            vanilla_d_loss,
            vanilla_g_loss,
        )
        print("    ✓ Discriminator components imported successfully")
        
        print("  - Importing Lightning Module...")
        from models.rae_module import RAELitModule
        print("    ✓ RAELitModule imported successfully")
        
        print("  - Importing Data Module...")
        from data.image_folder_datamodule import ImageFolderDataModule
        print("    ✓ ImageFolderDataModule imported successfully")
        
        print("  - Importing utils...")
        from utils.optim_utils import build_scheduler
        print("    ✓ Utils imported successfully")
        
        print("\n✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)