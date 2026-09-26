"""V5.1 training paths; consume existing V5 assets read-only, write separate logs.

The V5.1 CLI exposes train/play only. Asset generation is intentionally not an
entrypoint: these paths refer to the already calibrated V5 physical model.
"""
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parents[1]

# Read-only inputs owned by HOMIE.
G1_DESCRIPTION_DIR = ROOT / "HomieRL/legged_gym/resources/robots/g1_description"
G1_URDF = G1_DESCRIPTION_DIR / "g1.urdf"
CART_OBJ = ROOT / "HomieRL/legged_gym/resources/object/xiaotuiche.obj"

# Existing V5 assets are read-only inputs to V5.1 training/play.
ASSET_DIR = ROOT / "assets/g1_pushcart_v5"
GENERATED_URDF_DIR = ASSET_DIR / "generated_urdf"
G1_GRASP_URDF = GENERATED_URDF_DIR / "g1/g1_grasp.urdf"
CART_DIR = GENERATED_URDF_DIR / "cart"
CART_URDF = CART_DIR / "cart.urdf"
PACKED_USD = ASSET_DIR / "g1_pushcart.usd"
ASSET_REPORT = ASSET_DIR / "asset_report.json"
LOG_DIR = ROOT / "logs/rsl_rl/homie_pushcart_v51"

# Configuration.
CONFIG_DIR = PACKAGE_DIR / "config"
INITIAL_POSE_YAML = CONFIG_DIR / "initial_pose.yaml"
CART_PARAMS_YAML = CONFIG_DIR / "cart_params.yaml"

# HOMIE training output used for the low-level policy and physics comparison.
HOMIE_RUN_DIR = ROOT / "logs/rsl_rl/homie_g1/2026-09-14_15-10-07_my_policy"
# The frozen low-level policy for the whole stage-one work (decision 2026-09-17: HOMIE iteration 60000).
# Every tool defaults to this file; `stance --apply`, `pack` and `mj build` must be re-run if it changes.
HOMIE_FROZEN_CHECKPOINT = HOMIE_RUN_DIR / "model_60000.pt"

# Joint groups in HOMIE's fixed order (12 legs, waist yaw, 7 + 7 arm joints).
LEG_JOINTS = [f"{side}_{name}_joint" for side in ("left", "right")
              for name in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
WAIST_JOINTS = ["waist_yaw_joint"]
ARM_JOINT_NAMES = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
ARM_JOINTS = [f"{side}_{name}_joint" for side in ("left", "right") for name in ARM_JOINT_NAMES]
HOMIE_JOINTS = LEG_JOINTS + WAIST_JOINTS + ARM_JOINTS
FINGER_JOINT_NAMES = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
FINGER_JOINTS = [f"{side}_hand_{name}_joint" for side in ("left", "right") for name in FINGER_JOINT_NAMES]
PALM_LINKS = {"left": "left_hand_palm_link", "right": "right_hand_palm_link"}
