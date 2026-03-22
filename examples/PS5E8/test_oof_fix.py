"""Quick test to verify OOF saving/loading works correctly"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from tabml import OOFManager

# Create test data
test_series = pd.Series(np.random.rand(100), name='test_pred')
print(f"Original series shape: {test_series.shape}")
print(f"Original series head: {test_series.head()}")

# Save using OOFManager
manager = OOFManager(output_dir="output/test_oof")
manager.save_oof(
    predictions=test_series,
    model_name="test_model",
    cv_score=0.95
)

# Load it back
loaded = manager.load_all_oofs()
print(f"\nLoaded {len(loaded)} files")

for filename, data in loaded.items():
    print(f"\nFile: {filename}")
    print(f"  Predictions type: {type(data['predictions'])}")
    print(f"  Predictions shape: {data['predictions'].shape}")
    print(f"  First values: {data['predictions'].head()}")

# Test combining
combined = manager.combine_oofs(loaded)
print(f"\nCombined shape: {combined.shape}")
print(f"Combined head:\n{combined.head()}")

print("\n✓ Test passed! OOF saving/loading works correctly")