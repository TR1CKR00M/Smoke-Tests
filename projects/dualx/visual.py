import sys
from pathlib import Path

DUALX_ROOT = Path(__file__).resolve().parent
EXH_ROOT= DUALX_ROOT / "X-FORCE" / "xforce" / "data"
XFORCE_SRC = DUALX_ROOT / "X-FORCE" / "xforce" / "src"
if not (XFORCE_SRC / "xforce").is_dir():
    raise FileNotFoundError(f"Could not find the X-FORCE package source at {XFORCE_SRC}")
sys.path.insert(0, str(XFORCE_SRC))

from ase.io import read
from xforce.gradcam import PublicationMACEGradCAM, PublicationVisualizer

config = {
    "models": [{
        "name": "Foundation",
        "path": str(DUALX_ROOT / "data/raw/mace-mpa-0-medium.model"),
    }],
    "analysis": {
        "device": "cpu",       # 有可用 CUDA 时可改为 "cuda"
        "debug_mode": False,
    },
}

atoms = read(EXH_ROOT /"shap/sampled_trajectory_Cr2_temp.xyz", index=0)

model = PublicationMACEGradCAM(
    model_path=config["models"][0]["path"],
    model_name=config["models"][0]["name"],
    device=config["analysis"]["device"],
    debug_mode=config["analysis"]["debug_mode"],
)
result = model.analyze_structure(atoms, "frame_0")

visualizer = PublicationVisualizer(config)
output_dir = DUALX_ROOT / "results/gradcam"
output_dir.mkdir(parents=True, exist_ok=True)

visualizer.create_2d_continuous_heatmap_pdf(
    result,
    str(output_dir / "frame_0_xy_heatmap.pdf"),
    projection="xy",
)
