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

from xforce.gradcam import PublicationMACEGradCAM
from ase.io import read

# Resolve input paths from dualx root, independent of the current directory.
DATA_DIR = DUALX_ROOT / "data" / "raw"

# Load your MACE model and structure
gradcam = PublicationMACEGradCAM(
    model_path=str(DATA_DIR / "mace-mpa-0-medium.model"),
    model_name="MPA-0",
    device="cpu"
)

# Load atomic structure
atoms = read(DATA_DIR / "sb2te3_cr_81atom.xyz")

# Compute importance scores
importance_scores, predicted_energy = gradcam.compute_atom_importance(atoms)

# Analyze structure
result = gradcam.analyze_structure(atoms, "my_structure")
print(f"Most important atom: {result['statistics']['most_important_atom']}")
