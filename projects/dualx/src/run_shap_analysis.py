"""Run Grad-CAM using the X-FORCE checkout nested in this repository."""

import sys
from pathlib import Path

# X-FORCE is installed from the repository root at ``X-FORCE``. Its package
# source lives under ``X-FORCE/xforce/src``; add that checkout as a fallback so
# this script also works when launched from the dualx project root.
DUALX_ROOT = Path(__file__).resolve().parents[1]
XFORCE_SRC = DUALX_ROOT / "X-FORCE" / "xforce" / "src"
if not (XFORCE_SRC / "xforce").is_dir():
    raise FileNotFoundError(f"Could not find the X-FORCE package source at {XFORCE_SRC}")
sys.path.insert(0, str(XFORCE_SRC))

from xforce.shap import DirectSOAPAnalyzer, perform_direct_shap_analysis

# Resolve data paths from the project root so the script works regardless of
# the working directory used to launch it.
SHAP_DATA = DUALX_ROOT / "X-FORCE" / "xforce" / "data" / "shap"
analyzer = DirectSOAPAnalyzer(
    xyz_file_path=str(SHAP_DATA / "sampled_trajectory_Cr2_temp.xyz"),
    predictions_csv_path=str(SHAP_DATA / "sampled_trajectory_Cr2_temp_predictions_with_binding.csv"),
)

# Load and process data
analyzer.load_data()
datasets = analyzer.create_model_datasets()

# Run SHAP analysis
shap_results = perform_direct_shap_analysis(datasets)

# Access results for a specific model
model_results = shap_results['MACE Foundation']
print(f"Top features: {model_results['feature_names'][:10]}")
