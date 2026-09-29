"""V6's single physical baseline: the unchanged V3.2 assets and execution.

Legacy MuJoCo already has four contact spheres per foot. Its joints, collision
model, welds, and motor stepping must not pass through the V5 scene converter.
Isaac training continues to use V3.2's mixed leg-effort/upper-position drive;
the original MuJoCo evaluator uses explicit PD motors for all 27 joints.
"""
from __future__ import annotations

import hashlib

from homie_pushcart.paths import (
    CART_PARAMS_YAML, G1_GRASP_URDF, G1_URDF, INITIAL_POSE_YAML, PACKED_USD, ROOT,
)


MJCF_SCENE = ROOT / "assets/g1_pushcart/mujoco/g1_pushcart.xml"
PHYSICS_ID = "v6_v32_legacy_four_spheres"
PHYSICS_DT = 0.005
CONTROL_DECIMATION = 4
FOOT_CONTACT_POINTS = ((-.05, .025, -.03), (-.05, -.025, -.03),
                       (.12, .03, -.03), (.12, -.03, -.03))

# Reviewed V3.2 inputs, including the composed USD's complete 17-file closure.
# Keep these hashes in the version contract, independent of benchmark logs.
_USD_SHA256 = {
    "g1_pushcart.usd": "ac262b2f755052ac57a7761571860d854826584a6d1d0ac11de27094a6c75254",
    "cart/cart/cart.usda": "2a79c65dd3b6840be04604ce22a34d569b26efe4ef7253a90d55d7e3ece8917c",
    "cart/cart/payloads/base.usda": "0fb93424409dc95f191d17234608eb0e2b37614d77f22864ffe8d79d4c2175c0",
    "cart/cart/payloads/robot.usda": "3288cd9cb6c6d67d6b0f57b3430f8cc6ecb062052013dbf8bca17ae5314ef80b",
    "cart/cart/payloads/geometries.usd": "eee8069cfdcb0ef0fb04564634cd83853a1437c966d65f0496eb5b84608d48e1",
    "cart/cart/payloads/instances.usda": "8ad3d38f93d2867c4bdfde768a53736b912838baa5c6fa606c83b59c9e30a6e2",
    "cart/cart/payloads/materials.usda": "e28b351acdcf3b6bc126c34225ccd4aa8aaf706775bd03b7ab6ff8e03bebcaab",
    "cart/cart/payloads/Physics/physics.usda": "be26d0e61ba684540fdaab2ce246918797435e5eb87eafaeb85f05b965c98b31",
    "cart/cart/payloads/Physics/physx.usda": "43a3f84677eb7be5b4fe890747acb935a0cee8e30212c45edaa63a66562042be",
    "g1_grasp/g1_grasp/g1_grasp.usda": "061b637379601647a842c0c50cd034d1f5a9631f2d9d428195517b42f88efbab",
    "g1_grasp/g1_grasp/payloads/base.usda": "7b2311107317c522ee0559934506e6eb551071f004b259fe00ee44477b7169f2",
    "g1_grasp/g1_grasp/payloads/robot.usda": "763d4834c78c15964d3dc3896ade3774c2f838b683ad0c72784ca4314c03defe",
    "g1_grasp/g1_grasp/payloads/geometries.usd": "51328e981aafcf7df3aeba275497186823f7965c338e1c8c77e15ab1943c76fe",
    "g1_grasp/g1_grasp/payloads/instances.usda": "f1eaca7dbb4dba50e311fd015a3d4212e1b50a83ba76fdd1deadd8091988938d",
    "g1_grasp/g1_grasp/payloads/materials.usda": "ce406191672d1b17223ee39e496a06e7955ba6ecc6a1fcc275c8339defd6a601",
    "g1_grasp/g1_grasp/payloads/Physics/physics.usda": "a9be99873c8ed424e6cb162097df0fc78b87f36cbe847c408b57eb0ed5fcfb4f",
    "g1_grasp/g1_grasp/payloads/Physics/physx.usda": "1fe618c8117530490cf4b04e0835d6c344d72bf32fc5b5c5534e1e9fc2c543b1",
}
_INPUT_SHA256 = {
    MJCF_SCENE: "29392f5cce8e850a0cbc58a2abe5e4e85717bccfc8745ef8937bd50a1f8e7b98",
    INITIAL_POSE_YAML: "907fce1062dc4ace7cbca72ab1de158eb625093225e7ad7dce3dfca56a73080b",
    CART_PARAMS_YAML: "97ecfc0a807cfeecc3b33b23098b0de6375ddb3eb917b91b531d76925a008aea",
    G1_URDF: "449b618e3c0e640d600e73e6cc9072d06b62a8c56c393bbb83aa46b32a9e2153",
    G1_GRASP_URDF: "528256bc2676f26d4c250f244575d3205bf0e4d1f338ead860e35b51591560c9",
    **{PACKED_USD.parent / name: digest for name, digest in _USD_SHA256.items()},
}


def validate_assets() -> dict[str, str]:
    """Reject asset drift; return verified repository-relative SHA256 records."""
    verified = {}
    for path, expected in _INPUT_SHA256.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"V6 V3.2 baseline asset changed: {path}; expected {expected}, got {actual}")
        verified[path.relative_to(ROOT).as_posix()] = actual
    return verified


def physics_manifest() -> dict:
    """Serializable physical provenance for training, playback and bench logs."""
    assets = validate_assets()
    return {
        "physics_id": PHYSICS_ID,
        "mjcf_scene": str(MJCF_SCENE),
        "packed_usd": str(PACKED_USD),
        "asset_sha256": assets,
        "physics_dt_s": PHYSICS_DT,
        "control_decimation": CONTROL_DECIMATION,
        "control_dt_s": PHYSICS_DT * CONTROL_DECIMATION,
        "foot_contact": {"spheres_per_foot": 4, "radius_m": .005,
                         "positions_m": [list(p) for p in FOOT_CONTACT_POINTS]},
        "isaac_drive": "V3.2: leg explicit PD effort, upper implicit position drive",
        "mujoco_drive": "V3.2: all-joint explicit PD, recomputed every physics step",
        "isaac_ground": "V3.2 static collision cuboid, no kinematic rigid body",
        "payload": "V3.2: base mass plus payload; proportional base inertia; original COM/reset order",
        "isaac_source": "homie_pushcart.sim_env.PushCartSimEnv",
        "isaac_payload_source": "homie_pushcart.task_env.PushCartEnv._randomize_cart",
        "mujoco_source": "homie_pushcart.mujoco_env.MujocoHomieEnv",
        "mujoco_payload_source": "homie_pushcart.mujoco_hl.MujocoHighLevelRunner.set_payload",
    }


def _validate_mujoco_config(cfg) -> None:
    from homie_isaaclab.config import G1RoughCfg

    baseline = G1RoughCfg()
    for section, fields in {
        "sim": ("dt", "gravity"),
        "control": ("control_type", "decimation", "stiffness", "damping", "action_scale"),
        "init_state": ("default_joint_angles",),
        "asset": ("armature",),
    }.items():
        for field in fields:
            if getattr(getattr(cfg, section), field) != getattr(getattr(baseline, section), field):
                raise ValueError(f"V6 physical baseline forbids overriding {section}.{field}")


def make_mujoco_env(*, cfg=None):
    """Return the original V3.2 environment, without alternate scene/drive flags.

    Payload changes belong to the unchanged V3.2 high-level runner's
    ``set_payload`` method; this factory never changes mass or inertia.
    """
    from homie_isaaclab.config import G1RoughCfg
    from homie_pushcart.mujoco_env import MujocoHomieEnv

    validate_assets()
    cfg = G1RoughCfg() if cfg is None else cfg
    _validate_mujoco_config(cfg)
    return MujocoHomieEnv(with_cart=True, xml_path=MJCF_SCENE,
                          decimation=CONTROL_DECIMATION, cfg=cfg)
