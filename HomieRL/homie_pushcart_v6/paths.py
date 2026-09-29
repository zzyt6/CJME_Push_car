"""V3.2 assets stay read-only; V6 owns only its program and outputs."""
from pathlib import Path
from homie_pushcart.paths import (
    ROOT, G1_URDF, G1_GRASP_URDF, CART_URDF, PACKED_USD, ASSET_REPORT,
    INITIAL_POSE_YAML, CART_PARAMS_YAML, HOMIE_JOINTS, PALM_LINKS,
    HOMIE_FROZEN_CHECKPOINT, HOMIE_RUN_DIR,
)

PACKAGE_DIR = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs/rsl_rl/homie_pushcart_v6"
