"""CPU checks for the push-cart assets: HOMIE compatibility of g1_grasp.urdf, pose YAML, cart derivation."""
import json
import math
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HomieRL"))

from homie_pushcart.assets.cart_urdf import derive_cart_pose, load_cart_params, write_cart_urdf  # noqa: E402
from homie_pushcart.assets.g1_grasp_urdf import bake_rpy, verify_g1_grasp_urdf  # noqa: E402
from homie_pushcart.paths import (ARM_JOINTS, FINGER_JOINTS, G1_GRASP_URDF, G1_URDF, HOMIE_JOINTS,  # noqa: E402
                                  INITIAL_POSE_YAML, LEG_JOINTS)
from homie_pushcart.pose import Kinematics, foot_tilt, homie_default_joint_angles, load_initial_pose  # noqa: E402


class GraspUrdfTests(unittest.TestCase):
    def test_bake_rpy_about_z(self):
        rpy = bake_rpy(np.zeros(3), np.array([0., 0., 1.]), 0.7)
        np.testing.assert_allclose(rpy, [0., 0., 0.7], atol=1e-12)

    def test_grasp_urdf_keeps_homie_articulation(self):
        self.assertTrue(G1_GRASP_URDF.is_file(), "run `pushcart.py pose import` first")
        verify_g1_grasp_urdf(G1_URDF, G1_GRASP_URDF)
        root = ET.parse(G1_GRASP_URDF).getroot()
        movable = [j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"]
        self.assertEqual(movable, HOMIE_JOINTS)
        types = {j.get("name"): j.get("type") for j in root.findall("joint")}
        for name in FINGER_JOINTS:
            self.assertEqual(types[name], "fixed")
        self.assertEqual(types["waist_roll_joint"], "fixed")
        self.assertEqual(types["waist_pitch_joint"], "fixed")

    def test_finger_rotation_matches_yaml(self):
        pose = load_initial_pose()
        root = ET.parse(G1_GRASP_URDF).getroot()
        joint = next(j for j in root.findall("joint") if j.get("name") == "right_hand_index_0_joint")
        rpy = [float(v) for v in joint.find("origin").get("rpy").split()]
        self.assertAlmostEqual(rpy[2], pose["robot"]["finger_pos"]["right_hand_index_0_joint"], places=6)


class InitialPoseTests(unittest.TestCase):
    def test_yaml_layout(self):
        pose = load_initial_pose()
        self.assertEqual(list(pose["robot"]["joint_pos"]), HOMIE_JOINTS)
        self.assertEqual(sorted(pose["robot"]["finger_pos"]), sorted(FINGER_JOINTS))
        defaults = homie_default_joint_angles()
        self.assertAlmostEqual(pose["robot"]["joint_pos"]["waist_yaw_joint"], defaults["waist_yaw_joint"])
        source = json.loads((ROOT / pose["source"]["json"]).read_text(encoding="utf-8"))
        last = source["solved_cartesian_sequence"][-1]["frames"][-1]["controlled_motor_positions"]
        for name in ARM_JOINTS:
            self.assertAlmostEqual(pose["robot"]["joint_pos"][name], last[name], places=6)

    def test_legs_are_homie_measured_stance(self):
        """Legs come from `stance --apply` (HOMIE's real standing pose), symmetric, close to the measurement."""
        pose = load_initial_pose()
        stance = pose.get("stance")
        self.assertIsNotNone(stance, "run `stance --apply` first")
        jp = pose["robot"]["joint_pos"]
        for kind in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"):
            sign = -1.0 if kind.endswith(("roll", "yaw")) else 1.0
            self.assertAlmostEqual(jp[f"left_{kind}_joint"], sign * jp[f"right_{kind}_joint"], places=6)
        measured = stance["measured_leg_joint_mean"]
        for name in LEG_JOINTS:
            self.assertLess(abs(jp[name] - measured[name]), 0.03, name)   # symmetrisation + ankle leveling only
        # the stance is not HOMIE's default pose: this is the whole point of the calibration
        self.assertGreater(abs(jp["left_knee_joint"] - homie_default_joint_angles()["left_knee_joint"]), 0.1)
        self.assertLess(abs(stance["feet_float_at_reset_m"]), 0.006)
        offset = pose["robot"]["upper_hold_offset"]
        self.assertEqual(sorted(offset), sorted(HOMIE_JOINTS[12:]))
        self.assertLess(max(abs(v) for v in offset.values()), 0.15)
        for name in ARM_JOINTS:
            mirror = name.replace("left", "right") if name.startswith("left") else None
            if mirror:
                sign = -1.0 if any(k in name for k in ("roll", "yaw")) else 1.0
                self.assertAlmostEqual(offset[name], sign * offset[mirror], places=2)

    def test_feet_on_ground_and_mirrored_hands(self):
        pose = load_initial_pose()
        kin = Kinematics(load_meshes=True)
        kin.set_configuration(pose["robot"]["joint_pos"], pose["robot"]["base_pos"], pose["robot"]["base_quat_xyzw"])
        # the lowest vertex of the foot meshes (URDF/MJCF collision geometry) sits at the recorded reset gap
        # above the ground (measured standing height minus FK-on-ground height), and both soles are level
        # in the world even though the pelvis carries HOMIE's measured forward pitch
        graph = kin.urdf.scene.graph
        lowest = min(
            (kin.world_T_base @ graph[node][0] @ np.c_[kin.urdf.scene.geometry[graph[node][1]].vertices,
                                                        np.ones(len(kin.urdf.scene.geometry[graph[node][1]].vertices))].T).T[:, 2].min()
            for node in graph.nodes_geometry if graph.transforms.parents[node].endswith("ankle_roll_link"))
        expected_gap = pose.get("stance", {}).get("feet_float_at_reset_m", 0.0)
        self.assertAlmostEqual(lowest, expected_gap, places=5)
        for side in ("left", "right"):
            pitch, roll = foot_tilt(kin, side)
            self.assertLess(abs(pitch), 1e-5)
            self.assertLess(abs(roll), 1e-5)
        stance = pose.get("stance", {})
        if stance.get("base_pitch_baked_rad") is not None:
            x, y, z, w = pose["robot"]["base_quat_xyzw"]
            self.assertAlmostEqual(2 * math.atan2(y, w), stance["base_pitch_baked_rad"], places=6)
            self.assertAlmostEqual(x, 0.0); self.assertAlmostEqual(z, 0.0)
        left = kin.world_T("left_hand_palm_link")[:3, 3]
        right = kin.world_T("right_hand_palm_link")[:3, 3]
        np.testing.assert_allclose(left * [1, -1, 1], right, atol=2e-5)


class CartTests(unittest.TestCase):
    def test_bar_is_level_and_through_grasp_points(self):
        params, pose = load_cart_params(), load_initial_pose()
        derived = derive_cart_pose(params, pose)
        self.assertLess(derived["bar_tilt_deg"], 0.01)
        gp = derived["grasp_points_world"]
        self.assertAlmostEqual(gp["left"][2], derived["handle_height"], places=6)
        self.assertAlmostEqual(derived["base_pos_world"][0] - derived["bar_center_world"][0],
                               -params["frame"]["x_positions"][0], places=6)
        self.assertEqual(derived["base_pos_world"][2], 0.0)

    def test_cart_urdf_models(self):
        import tempfile
        params = load_cart_params()
        for model, expected_joints in (("caster", 8), ("rigid", 0)):
            with tempfile.TemporaryDirectory() as tmp:
                dest = Path(tmp) / "cart.urdf"
                write_cart_urdf({**params, "model": model}, 0.92, dest)
                root = ET.parse(dest).getroot()
                joints = [j for j in root.findall("joint")]
                self.assertEqual(len(joints), expected_joints)
                self.assertTrue(all(j.get("type") == "continuous" for j in joints))
                base = next(l for l in root.findall("link") if l.get("name") == "base_link")
                names = [c.get("name") for c in base.findall("collision")]
                self.assertIn("deck", names)
                self.assertNotIn("bar_0", names, "handle bar must stay visual only")
                self.assertEqual(float(base.find("inertial/mass").get("value")), params["mass"]["base_kg"])


class ViewerCheckpointDiscoveryTests(unittest.TestCase):
    """`mj view` picks the newest checkpoint of the newest run unless told otherwise (no simulator needed)."""

    def test_find_hl_checkpoint(self):
        import tempfile
        from homie_pushcart.mujoco_viewer import find_hl_checkpoint, list_checkpoints
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp)
            old, new, empty = logs / "2026-09-17_17-57-50_two_stage", logs / "2026-09-17_21-25-55_b4", logs / "2026-09-18_00-00-00_no_ckpt"
            for d in (old, new, empty):
                d.mkdir()
            for name in ("model_0.pt", "model_200.pt", "model_1000.pt"):     # lexical order would put 1000 before 200
                (old / name).write_bytes(b"x")
            for name in ("model_0.pt", "model_800.pt", "model_1800.pt"):
                (new / name).write_bytes(b"x")
            (empty / "params").mkdir()
            self.assertEqual([p.name for p in list_checkpoints(old)], ["model_0.pt", "model_200.pt", "model_1000.pt"])
            self.assertEqual(find_hl_checkpoint(log_dir=logs), new / "model_1800.pt")            # newest run with checkpoints
            self.assertEqual(find_hl_checkpoint(run=old, log_dir=logs), old / "model_1000.pt")     # explicit run, numeric max
            self.assertEqual(find_hl_checkpoint(run=Path(old.name), log_dir=logs), old / "model_1000.pt")   # run by name
            self.assertEqual(find_hl_checkpoint(checkpoint=old / "model_200.pt", log_dir=logs), old / "model_200.pt")
            with self.assertRaises(FileNotFoundError):
                find_hl_checkpoint(run=empty, log_dir=logs)
            with self.assertRaises(FileNotFoundError):
                find_hl_checkpoint(checkpoint=new / "model_999.pt", log_dir=logs)


if __name__ == "__main__":
    unittest.main()
