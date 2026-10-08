"""V7 owns its cart/grasp assets; the robot description and V3.2 assets stay read-only."""
from pathlib import Path
from homie_pushcart.paths import (
    ROOT, G1_URDF, G1_GRASP_URDF, INITIAL_POSE_YAML, CART_PARAMS_YAML, HOMIE_JOINTS, PALM_LINKS,
    HOMIE_FROZEN_CHECKPOINT, HOMIE_RUN_DIR,
)

PACKAGE_DIR = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs/rsl_rl/homie_pushcart_v7"
CART_PARAMS_V7_YAML = PACKAGE_DIR / "config" / "cart_params_v7.yaml"

ASSET_DIR = ROOT / "assets/g1_pushcart_v7"
# Next to V3.2's cart.urdf so the shared OBJ resolves through the same relative path (../xiaotuiche.obj).
CART_URDF = ROOT / "HomieRL/legged_gym/resources/object/cart_v7/cart_v7.urdf"
PACKED_USD = ASSET_DIR / "g1_pushcart_v7.usd"
ASSET_REPORT = ASSET_DIR / "asset_report.json"
MJCF_SCENE = ASSET_DIR / "mujoco" / "g1_pushcart_v7.xml"
MJCF_REPORT = ASSET_DIR / "mujoco" / "mjcf_report.json"
BUILD_MANIFEST = ASSET_DIR / "build_manifest.json"

SIDES = ("left", "right")
SLIDE_JOINTS = {side: f"grasp_{side}_slide_joint" for side in SIDES}
ROLL_JOINTS = {side: f"grasp_{side}_roll_joint" for side in SIDES}
GRASP_LINKS = {side: f"grasp_{side}_link" for side in SIDES}
SLIDER_LINKS = {side: f"grasp_{side}_slider" for side in SIDES}
TILT_JOINTS = {side: f"grasp_{side}_tilt_joint" for side in SIDES}
YAW_JOINTS = {side: f"grasp_{side}_yaw_joint" for side in SIDES}
ROLL_LINKS = {side: f"grasp_{side}_roll_link" for side in SIDES}
TILT_LINKS = {side: f"grasp_{side}_tilt_link" for side in SIDES}
ROTATION_JOINTS = {"roll": ROLL_JOINTS, "tilt": TILT_JOINTS, "yaw": YAW_JOINTS}

# V7 grasp geometry (assets/grasp_fit.py): fitted finger angles + grasp points, and the robot URDF baked with them.
GRASP_POSE_V7_YAML = PACKAGE_DIR / "config" / "grasp_pose_v7.yaml"
G1_GRASP_V7_URDF = G1_GRASP_URDF.with_name("g1_grasp_v7.urdf")
GRASP_SECTION_PNG = ASSET_DIR / "grasp_section_v7.png"
