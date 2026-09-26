"""Batched encoder/FK cart observation for the V5.1 *fixed-grasp* plant.

Only initialization reads the robot URDF and the grasp/cart calibration YAML.
Runtime inputs are absolute encoder angles/rates, HIM base velocity and IMU;
no simulator cart pose, cart velocity or world robot pose is accepted. The
left fixed grasp determines cart pose; the independent right chain checks it.
This is not an observer for slipping, compliant or changing grasps.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .assets.cart_urdf import derive_cart_pose, load_cart_params
from .paths import G1_GRASP_URDF, HOMIE_JOINTS, PALM_LINKS
from .pose import Kinematics, load_initial_pose


def _mv(matrix, vector):
    return (matrix @ vector.unsqueeze(-1)).squeeze(-1)


@dataclass(frozen=True)
class _Joint:
    index: int | None
    kind: str
    position: torch.Tensor
    rotation: torch.Tensor
    axis: torch.Tensor
    skew: torch.Tensor
    outer: torch.Tensor


class BatchedPalmKinematics:
    """URDF tree FK/Jacobians in pelvis axes; cached tensors, no runtime NumPy.

    ``forward(q)`` takes [N,27] absolute angles in ``HOMIE_JOINTS`` order and
    returns ``(positions[N,2,3], rotations[N,2,3,3], jacobian[N,2,6,27])``.
    Jacobian order is linear, angular, at each palm origin, relative to pelvis.
    The two sides are always left, right. Leg columns are correctly zero.
    """

    def __init__(self, urdf_path=G1_GRASP_URDF, *, device="cpu", dtype=torch.float32):
        self.device, self.dtype = torch.device(device), dtype
        root = ET.parse(Path(urdf_path)).getroot()
        parents = {j.find("child").get("link"): j for j in root.findall("joint")}
        roots = {link.get("name") for link in root.findall("link")} - set(parents)
        if len(roots) != 1:
            raise ValueError("URDF must have one root link")
        self.base_link = roots.pop()
        self.joint_names = tuple(HOMIE_JOINTS)
        self.eye = torch.eye(3, device=device, dtype=dtype)
        self.device = self.eye.device
        self.chains = []
        for side in ("left", "right"):
            link, elements, visited = PALM_LINKS[side], [], set()
            while link != self.base_link:
                if link in visited or link not in parents:
                    raise ValueError(f"Invalid chain to {PALM_LINKS[side]}")
                visited.add(link)
                joint = parents[link]
                elements.append(joint)
                link = joint.find("parent").get("link")
            chain = []
            for joint in reversed(elements):
                kind, name = joint.get("type"), joint.get("name")
                if kind not in ("fixed", "revolute", "continuous", "prismatic"):
                    raise ValueError(f"Unsupported joint {name}: {kind}")
                if joint.find("mimic") is not None:
                    raise ValueError("Mimic joints on a palm chain are not supported")
                index = None if kind == "fixed" else self.joint_names.index(name)
                origin, axis_node = joint.find("origin"), joint.find("axis")
                xyz = [float(x) for x in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
                rpy = [float(x) for x in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
                axis = np.array([float(x) for x in (axis_node.get("xyz", "1 0 0") if axis_node is not None else "1 0 0").split()])
                if not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-12:
                    raise ValueError(f"Invalid axis for {name}")
                axis /= np.linalg.norm(axis)
                x, y, z = axis
                skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
                tensor = lambda value: torch.as_tensor(value, device=device, dtype=dtype)
                chain.append(_Joint(index, kind, tensor(xyz), tensor(Rotation.from_euler("xyz", rpy).as_matrix()),
                                    tensor(axis), tensor(skew), tensor(np.outer(axis, axis))))
            self.chains.append(tuple(chain))

    def forward(self, q):
        if q.ndim != 2 or q.shape[1] != 27 or q.device != self.device or q.dtype != self.dtype:
            raise ValueError("q must be [N,27] on the configured device/dtype")
        n = q.shape[0]
        poses, rotations, jacobians = [], [], []
        for chain in self.chains:
            position = torch.zeros(n, 3, device=self.device, dtype=self.dtype)
            rotation = self.eye.expand(n, -1, -1)
            axes = []
            for joint in chain:
                position = position + _mv(rotation, joint.position.expand(n, -1))
                rotation = rotation @ joint.rotation
                if joint.index is None:
                    continue
                axis_b = _mv(rotation, joint.axis.expand(n, -1))
                axes.append((joint.index, joint.kind, position, axis_b))
                angle = q[:, joint.index, None, None]
                if joint.kind == "prismatic":
                    position = position + axis_b * angle[:, :, 0]
                else:
                    c, s = torch.cos(angle), torch.sin(angle)
                    rotation = rotation @ (c * self.eye + s * joint.skew + (1. - c) * joint.outer)
            jacobian = torch.zeros(n, 6, 27, device=self.device, dtype=self.dtype)
            for index, kind, origin, axis in axes:
                if kind == "prismatic":
                    jacobian[:, :3, index] = axis
                else:
                    jacobian[:, :3, index] = torch.linalg.cross(axis, position - origin)
                    jacobian[:, 3:, index] = axis
            poses.append(position)
            rotations.append(rotation)
            jacobians.append(jacobian)
        return torch.stack(poses, 1), torch.stack(rotations, 1), torch.stack(jacobians, 1)


class FixedGraspObserver:
    """Instantaneous fixed-grasp observation with a same-sample input contract.

    All q, dq, HIM ``base_v_b`` and IMU ``gyro_b``/``gravity_b`` must refer to
    the same post-physics sample. q is the absolute measured joint position,
    not q minus the policy standing pose. Velocity is at the pelvis origin.
    IMU angular velocity is the pelvis' world angular velocity in body axes;
    gravity points down. If gravity is omitted, the body is assumed upright.
    No dt/history is used: no finite-difference warm-up or hidden sample lag.
    Optional ``timestamp_s`` [N] (or scalar) is recorded, not used to invent
    synchronization; the caller is responsible for aligning sensor streams.

    Pose outputs are physical pelvis/body axes B. ``cart_orientation_b`` maps
    cart axes to B; ``rel_yaw`` is cart heading minus upright base heading.
    ``cart_linear_velocity_b`` and ``cart_angular_velocity_b`` are *world*
    motion at the cart root origin, expressed in B, including omega x r and
    encoder Jacobian motion. ``cart_twist_b`` selects [v_Bx,v_By,omega_Bz].
    ``cart_twist_base_heading`` selects horizontal [vx,vy,absolute yaw rate]
    in upright base-heading axes; ``cart_twist_heading`` rotates its XY into
    upright cart-heading axes. These refer to cart root origin, NOT cart COM.
    Full 3D velocity/orientation are exposed for shifting to a nominal COM.

    ``valid`` requires finite inputs, non-singular headings, and agreement of
    two cart poses reconstructed from the independently calibrated grasps.
    A right-chain inconsistency invalidates the estimate; it does not alter
    the left-chain pose or silently average incompatible rotations.
    """

    def __init__(self, num_envs, device="cpu", dtype=torch.float32, *, pose=None, cart_params=None,
                 urdf_path=G1_GRASP_URDF, position_tolerance_m=.03, orientation_tolerance_rad=.15):
        self.num_envs, self.device, self.dtype = int(num_envs), torch.device(device), dtype
        if self.num_envs < 1 or position_tolerance_m < 0 or orientation_tolerance_rad < 0:
            raise ValueError("Positive environment count and nonnegative mismatch tolerances required")
        self.position_tolerance_m, self.orientation_tolerance_rad = float(position_tolerance_m), float(orientation_tolerance_rad)
        self.kinematics = BatchedPalmKinematics(urdf_path, device=device, dtype=dtype)
        pose = load_initial_pose() if pose is None else pose
        cart_params = load_cart_params() if cart_params is None else cart_params
        kin = Kinematics(urdf_path=urdf_path, load_meshes=False)
        derived = derive_cart_pose(cart_params, pose, kin)
        # Exactly the T_WC and T_WP used by assets.pack_usd.pack's fixed joints.
        cart_t = np.eye(4)
        cart_t[:3, :3] = Rotation.from_euler("z", derived["base_yaw"]).as_matrix()
        cart_t[:3, 3] = derived["base_pos_world"]
        palm_to_cart = np.stack([np.linalg.inv(kin.world_T(PALM_LINKS[s])) @ cart_t for s in ("left", "right")])
        tensor = lambda value: torch.as_tensor(value, device=device, dtype=dtype)
        self.palm_to_cart = tensor(palm_to_cart)
        self.handle_offset_cart = tensor([cart_params["frame"]["x_positions"][0], 0., derived["handle_height"]])
        self.initial_joint_pos = tensor([pose["robot"]["joint_pos"][name] for name in HOMIE_JOINTS])
        self.calibration = {"urdf": str(Path(urdf_path)), "primary_side": "left", "derived_cart_pose": derived,
                            "palm_to_cart": palm_to_cart.tolist(), "motion_reference": "cart root origin",
                            "timestamp_contract": "q/dq/HIM/IMU from the same post-physics sample"}
        shapes = {"palm_pos_b": (2, 3), "palm_rot_b": (2, 3, 3), "palm_jacobian_b": (2, 6, 27),
                  "cart_origin_pos_b": (3,), "cart_orientation_b": (3, 3), "handle_pos_b": (3,),
                  "cart_linear_velocity_b": (3,), "cart_angular_velocity_b": (3,),
                  "cart_twist_b": (3,), "cart_twist_base_heading": (3,), "cart_twist_heading": (3,),
                  "body_to_heading": (3, 3), "rel_yaw": (), "position_mismatch_m": (),
                  "orientation_mismatch_rad": (), "linear_velocity_mismatch_m_s": (),
                  "angular_velocity_mismatch_rad_s": (), "timestamp_s": ()}
        self._outputs = tuple(shapes)
        for name, shape in shapes.items():
            setattr(self, name, torch.zeros((self.num_envs, *shape), device=device, dtype=dtype))
        self.valid = torch.zeros(self.num_envs, device=device, dtype=torch.bool)
        self.ready = self.valid
        # Optional training acceleration. Eager update remains the general
        # sensor API; only update_fast initializes these CUDA resources.
        self._cuda_graph = None
        self._cuda_graph_inputs = None

    @property
    def cuda_graph_ready(self):
        """Whether update_fast has captured its fixed-shape CUDA computation."""
        return self._cuda_graph is not None

    @torch.no_grad()
    def update_fast(self, q, dq, base_v_b, gyro_b, gravity_b):
        """Replay the same observer math for fixed-shape float32 CUDA inputs.

        The first call warms up and captures ``update`` on a dedicated stream.
        Subsequent calls copy *this sample's* five inputs into stable buffers
        and replay on the caller's current stream. Outputs retain their tensor
        identities, so reset and consumers holding output views stay correct.
        No input history, simulator truth, timestamp, or external validity mask
        is introduced. Use ``update`` for timestamp/mask options. CPU and other
        input types/dtypes use the ordinary eager API without capturing.

        Like update, this method expects inputs produced on the current stream
        (or explicitly synchronized by their producer). One observer cannot be
        updated concurrently from multiple streams.
        """
        values = (q, dq, base_v_b, gyro_b, gravity_b)
        device = self.kinematics.device
        supported = device.type == "cuda" and self.dtype == torch.float32
        supported = supported and all(
            isinstance(value, torch.Tensor) and value.device == device
            and value.dtype == torch.float32 and value.shape == (self.num_envs, width)
            for value, width in zip(values, (27, 27, 3, 3, 3))
        )
        if not supported:
            return self.update(*values)
        if self._cuda_graph is None:
            self._capture_cuda_graph(values)
        else:
            for target, value in zip(self._cuda_graph_inputs, values):
                target.copy_(value)
        self._cuda_graph.replay()
        return self

    def _capture_cuda_graph(self, values):
        # Capture only pure Torch observation math, never simulator operations.
        # Copy live inputs before side-stream warmup; both waits are necessary
        # because warmup writes the same public output tensors as eager update.
        device = self.kinematics.device
        with torch.cuda.device(device):
            current = torch.cuda.current_stream(device)
            # These persistent buffers must also permit later calls outside a
            # rollout's inference_mode scope (e.g. a standalone evaluation).
            with torch.inference_mode(False):
                inputs = tuple(value.clone() for value in values)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.update(*inputs)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                self.update(*inputs)
            current.wait_stream(stream)
        # Keep captured input storage alive for the lifetime of the graph.
        # update uses copy_, so public output buffers are also already stable.
        self._cuda_graph_inputs = inputs
        self._cuda_graph = graph

    def reset(self, env_ids=None):
        """Clear selected output rows; the next valid sample is immediately ready."""
        ids = slice(None) if env_ids is None else env_ids
        for name in self._outputs:
            getattr(self, name)[ids] = 0.
        self.valid[ids] = False

    def _input(self, name, value, width):
        value = torch.as_tensor(value, device=self.device, dtype=self.dtype)
        if value.shape != (self.num_envs, width):
            raise ValueError(f"{name} must be [{self.num_envs},{width}]")
        return value

    def reference_base(self, command_cart_heading):
        """Nominal rigid-motion cart-root reference -> pelvis heading command.

        For the FK-derived root offset r (pelvis -> cart), v_pelvis_ref =
        R_HC v_cart_ref - omega_ref x r_H. No measured joint velocity or cart
        truth enters this reference; RL handles arm motion relative to the
        pelvis. Input/output [N,3] are [vx,vy,yaw_rate] in cart/base heading.
        """
        command = self._input("command_cart_heading", command_cart_heading, 3)
        c, s = torch.cos(self.rel_yaw), torch.sin(self.rel_yaw)
        r = _mv(self.body_to_heading, self.cart_origin_pos_b)
        vx = c * command[:, 0] - s * command[:, 1] + command[:, 2] * r[:, 1]
        vy = s * command[:, 0] + c * command[:, 1] - command[:, 2] * r[:, 0]
        return torch.stack((vx, vy, command[:, 2]), -1)

    @torch.no_grad()
    def update(self, q, dq, base_v_b, gyro_b, gravity_b=None, *, valid=None, timestamp_s=None):
        q, dq = self._input("q", q, 27), self._input("dq", dq, 27)
        base_v_b, gyro_b = self._input("base_v_b", base_v_b, 3), self._input("gyro_b", gyro_b, 3)
        if gravity_b is None:
            gravity_b = torch.zeros_like(base_v_b)
            gravity_b[:, 2] = -1.
        gravity_b = self._input("gravity_b", gravity_b, 3)
        finite = torch.ones_like(self.valid)
        values = [q, dq, base_v_b, gyro_b, gravity_b]
        for value in values:
            finite &= torch.isfinite(value).all(-1)
        q, dq, base_v_b, gyro_b, gravity_b = [torch.nan_to_num(x, nan=0., posinf=0., neginf=0.) for x in values]
        palm_p, palm_r, jac = self.kinematics.forward(q)
        relative = (jac @ dq[:, None, :, None]).squeeze(-1)
        offset = _mv(palm_r, self.palm_to_cart[None, :, :3, 3].expand(self.num_envs, -1, -1))
        positions = palm_p + offset
        rotations = palm_r @ self.palm_to_cart[:, :3, :3]
        relative_cart_v = relative[..., :3] + torch.linalg.cross(relative[..., 3:], offset)
        world_v = base_v_b[:, None] + torch.linalg.cross(gyro_b[:, None].expand(-1, 2, -1), positions) + relative_cart_v
        world_w = gyro_b[:, None] + relative[..., 3:]
        delta_rotation = rotations[:, 0].transpose(-1, -2) @ rotations[:, 1]
        skew = torch.stack((delta_rotation[:, 2, 1] - delta_rotation[:, 1, 2],
                            delta_rotation[:, 0, 2] - delta_rotation[:, 2, 0],
                            delta_rotation[:, 1, 0] - delta_rotation[:, 0, 1]), -1) * .5
        angle = torch.atan2(torch.linalg.vector_norm(skew, dim=-1),
                            (delta_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.) * .5)
        mismatch = torch.linalg.vector_norm(positions[:, 0] - positions[:, 1], dim=-1)
        # R_HB = Ry(pitch) Rx(roll), independent of unobserved world yaw.
        gnorm = torch.linalg.vector_norm(gravity_b, dim=-1)
        gravity = gravity_b / gnorm.clamp_min(1e-9)[:, None]
        roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
        pitch = torch.atan2(gravity[:, 0], torch.linalg.vector_norm(gravity[:, 1:], dim=-1))
        cr, sr, cp, sp = torch.cos(roll), torch.sin(roll), torch.cos(pitch), torch.sin(pitch)
        zero = torch.zeros_like(cr)
        level = torch.stack((cp, sp * sr, sp * cr, zero, cr, -sr, -sp, cp * sr, cp * cr), -1).reshape(-1, 3, 3)
        cart_r = rotations[:, 0]
        x_heading = _mv(level, cart_r[:, :, 0])
        horizontal2 = x_heading[:, :2].square().sum(-1)
        yaw = torch.atan2(x_heading[:, 1], x_heading[:, 0])
        v_heading, w_heading = _mv(level, world_v[:, 0]), _mv(level, world_w[:, 0])
        # Exact yaw derivative of a tilted cart's x axis, not merely omega_z.
        xdot = torch.linalg.cross(w_heading, x_heading)
        yaw_rate = (x_heading[:, 0] * xdot[:, 1] - x_heading[:, 1] * xdot[:, 0]) / horizontal2.clamp_min(1e-9)
        planar = torch.stack((v_heading[:, 0], v_heading[:, 1], yaw_rate), -1)
        c, s = torch.cos(yaw), torch.sin(yaw)
        cart_planar = torch.stack((c * planar[:, 0] + s * planar[:, 1], -s * planar[:, 0] + c * planar[:, 1], yaw_rate), -1)
        time = torch.zeros_like(self.timestamp_s) if timestamp_s is None else torch.as_tensor(timestamp_s, device=self.device, dtype=self.dtype)
        time = torch.broadcast_to(time, (self.num_envs,))
        finite &= torch.isfinite(time)
        good = finite & (gnorm > 1e-6) & (cp.abs() > 1e-4) & (horizontal2 > 1e-8)
        good &= (mismatch <= self.position_tolerance_m) & (angle <= self.orientation_tolerance_rad)
        if valid is not None:
            valid = torch.as_tensor(valid, device=self.device, dtype=torch.bool)
            if valid.shape != (self.num_envs,):
                raise ValueError("valid must be [N]")
            good &= valid
        outputs = {"palm_pos_b": palm_p, "palm_rot_b": palm_r, "palm_jacobian_b": jac,
                   "cart_origin_pos_b": positions[:, 0], "cart_orientation_b": cart_r,
                   "handle_pos_b": positions[:, 0] + _mv(cart_r, self.handle_offset_cart.expand(self.num_envs, -1)),
                   "cart_linear_velocity_b": world_v[:, 0], "cart_angular_velocity_b": world_w[:, 0],
                   "cart_twist_b": torch.stack((world_v[:, 0, 0], world_v[:, 0, 1], world_w[:, 0, 2]), -1),
                   "cart_twist_base_heading": planar, "cart_twist_heading": cart_planar, "body_to_heading": level,
                   "rel_yaw": yaw, "position_mismatch_m": mismatch, "orientation_mismatch_rad": angle,
                   "linear_velocity_mismatch_m_s": torch.linalg.vector_norm(world_v[:, 0] - world_v[:, 1], dim=-1),
                   "angular_velocity_mismatch_rad_s": torch.linalg.vector_norm(world_w[:, 0] - world_w[:, 1], dim=-1),
                   "timestamp_s": torch.nan_to_num(time, nan=0., posinf=0., neginf=0.)}
        for name, value in outputs.items():
            getattr(self, name).copy_(value)
        self.valid.copy_(good)
        return self
