"""V5.3 explicit-PD contract for all 27 drive-torque observation channels."""
from __future__ import annotations

UPPER_DRIVES = ("explicit_pd",)


def validate_upper_drive(mode):
    if mode not in UPPER_DRIVES:
        raise ValueError(f"upper_drive must be one of {UPPER_DRIVES}, got {mode!r}")
    return mode


def pd_effort(q, dq, target, kp, kd, limits, offset=0., target_velocity=0.):
    """FALCON/SDK hybrid PD: q, dq, kp, kd and feed-forward, in SI units.

    NumPy and Torch share this formula. Saturation applies to TOTAL effort;
    a hardware backend must not add a second PD after sending this effort.
    """
    return (kp * (target - q) - kd * (dq - target_velocity) + offset).clip(-limits, limits)


def nominal_configs(checkpoint, upper_drive=None, episode_length_s=20.):
    """Restore the run, then disable stochastic effects for paired diagnostics."""
    from .env_smoke import make_configs
    from homie_pushcart_v52.assets.cart_urdf import load_cart_params
    cfg, task = make_configs(1, False, 17, episode_length_s, checkpoint=checkpoint)
    if upper_drive is not None:
        task.actions.upper_drive = validate_upper_drive(upper_drive)
    task.curriculum.enabled = False
    task.actions.delay_steps_max = 0
    task.observations.joint_torque_delay_steps_max = 0
    task.observations.heading_delay_steps_max = 0
    task.randomization.init_xy_m = 0.
    task.randomization.init_yaw = False
    task.randomization.payload_com_offset_m = 0.
    task.randomization.cart_force_pulse = False
    task.randomization.wheel_stall_prob_per_s = 0.
    task.randomization.robot_push = False
    wheels = load_cart_params()['wheels']
    for key, source in (("wheel_damping_range", "wheel_damping"),
                        ("swivel_damping_range", "swivel_damping"),
                        ("wheel_friction_range", "friction")):
        setattr(task.randomization, key, [wheels[source], wheels[source]])
    return cfg, task


def control_contract(cfg, joint_names, torque_limits, upper_drive):
    """JSON-safe nominal parameters; runtime actuator gains are reported separately."""
    kp, kd = [], []
    for name in joint_names:
        key = next(k for k in cfg.control.stiffness if k in name)
        kp.append(float(cfg.control.stiffness[key]))
        kd.append(float(cfg.control.damping[key]))
    sc = cfg.normalization.obs_scales
    return dict(joint_names=list(joint_names), upper_drive=validate_upper_drive(upper_drive),
                kp=kp, kd=kd, torque_limits=[float(x) for x in torque_limits],
                default_joint_angles=[float(cfg.init_state.default_joint_angles[n]) for n in joint_names],
                action_scale=float(cfg.control.action_scale),
                action_clip=float(cfg.normalization.clip_actions),
                observation_clip=float(cfg.normalization.clip_observations),
                observation_scales={k: float(getattr(sc, k)) for k in ("lin_vel", "ang_vel", "dof_pos", "dof_vel")},
                height_command=float(cfg.rewards.base_height_target),
                physics_dt=float(cfg.sim.dt), decimation=int(cfg.control.decimation),
                control_dt=float(cfg.sim.dt * cfg.control.decimation),
                homie_frame_dim=int(cfg.env.num_one_step_observations),
                homie_history=int(cfg.env.num_actor_history),
                upper_torque_observation=("commanded clipped effort" if upper_drive == "explicit_pd"
                                          else "clipped pre-substep PD approximation"))
