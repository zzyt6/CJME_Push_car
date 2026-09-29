"""V5.3 cart uncertainty, expressed relative to its own nominal plant.

Robot ranges already come from G1RoughCfg (HOMIE). These pure tensor helpers
add cart mass uncertainty and caster asymmetry without editing the V5.2 plant.
"""
from __future__ import annotations

import math
import torch


def profile_overrides(profile):
    """Explicit migration for a resumed run, including measurement augmentation."""
    if profile not in ('legacy', 'homie_relative'):
        raise ValueError('Unknown domain randomization profile')
    from .task_config import PushCartTaskCfg
    cfg = PushCartTaskCfg()
    values = {'randomization.profile': profile}
    if profile == 'homie_relative':
        for key in ('enabled', 'cart_initial_fraction', 'cart_full_level', 'cart_link_mass_scale_range',
                    'wheel_friction_scale_range', 'caster_damping_scale_range', 'payload_com_deck_fraction',
                    'cart_push_delta_v_m_s', 'cart_push_interval_s'):
            values['randomization.'+key] = getattr(cfg.randomization, key)
        for key in ('curriculum_noise', 'joint_position_noise_rad', 'joint_velocity_noise_rad_s',
                    'gyro_noise_rad_s', 'gravity_noise', 'heading_noise_rad', 'heading_bias_rad',
                    'heading_delay_steps_max', 'heading_dropout_probability', 'upper_joint_velocity_noise_rad_s'):
            values['observations.'+key] = getattr(cfg.observations, key)
    else:
        values.update({'randomization.cart_push_interval_s': 5.,
            'observations.curriculum_noise': False, 'observations.joint_position_noise_rad': .001,
            'observations.joint_velocity_noise_rad_s': .02, 'observations.gyro_noise_rad_s': .01,
            'observations.gravity_noise': .005, 'observations.heading_noise_rad': math.radians(2.),
            'observations.heading_bias_rad': 0., 'observations.heading_delay_steps_max': 2,
            'observations.heading_dropout_probability': .05, 'observations.upper_joint_velocity_noise_rad_s': .3})
    return values


def validate_randomization(cfg):
    if cfg.profile not in ('legacy', 'homie_relative'):
        raise ValueError('randomization.profile must be legacy or homie_relative')
    for name in ('cart_link_mass_scale_range', 'wheel_friction_scale_range',
                 'caster_damping_scale_range', 'wheel_damping_range', 'swivel_damping_range'):
        bounds = getattr(cfg, name)
        if len(bounds) != 2 or not all(math.isfinite(v) and v > 0 for v in bounds) or bounds[0] > bounds[1]:
            raise ValueError(f'randomization.{name} requires ordered positive bounds')
    if not 0 <= cfg.cart_initial_fraction <= 1 or cfg.cart_full_level <= 0:
        raise ValueError('cart_initial_fraction must be in [0,1], cart_full_level positive')
    if len(cfg.payload_com_deck_fraction) != 2 or any(not 0 <= v <= .5 for v in cfg.payload_com_deck_fraction):
        raise ValueError('payload COM fractions must fit inside the deck half extents')
    if not math.isfinite(cfg.cart_push_delta_v_m_s) or cfg.cart_push_delta_v_m_s < 0:
        raise ValueError('cart_push_delta_v_m_s must be nonnegative')


def cart_fraction(cfg, level, enabled=True):
    if not enabled:
        return 0.
    if cfg.profile == 'legacy':
        return 1.
    return cfg.cart_initial_fraction + (1-cfg.cart_initial_fraction)*min(1., max(0., level)/cfg.cart_full_level)


def uniform_about(nominal, bounds, fraction, shape, *, device, dtype=torch.float32):
    """Interpolate the support about the nominal value; fraction zero is exact nominal."""
    lo, hi = (nominal + fraction*(float(v)-nominal) for v in bounds)
    return lo + (hi-lo)*torch.rand(shape, device=device, dtype=dtype)


def sample_cart(cfg, params, level, payload_max, n, body_count, *, device, enabled=True, fixed_payload=None):
    """Per-reset operating load plus model uncertainty. No hidden random load in nominal mode."""
    frac = cart_fraction(cfg, level, enabled)
    wheels = params['wheels']
    uniform = lambda nominal, bounds, shape: uniform_about(nominal, bounds, frac, shape, device=device)
    mass_scale = uniform(1., cfg.cart_link_mass_scale_range, (n, body_count))
    payload = torch.rand(n, 1, device=device)*payload_max if enabled else torch.zeros(n, 1, device=device)
    if fixed_payload is not None:
        if not math.isfinite(float(fixed_payload)) or fixed_payload < 0:
            raise ValueError('fixed cart payload must be finite and nonnegative')
        payload.fill_(float(fixed_payload))
    offset = torch.tensor(params['deck']['size'][:2], device=device)*torch.tensor(cfg.payload_com_deck_fraction, device=device)*frac
    com_xy = (2*torch.rand(n, 2, device=device)-1)*offset
    wheel = uniform(wheels['wheel_damping'], cfg.wheel_damping_range, (n, 1))
    swivel = uniform(wheels['swivel_damping'], cfg.swivel_damping_range, (n, 1))
    # Keep the wide existing rolling-resistance operating range; independently
    # perturb each caster by HOMIE's +/-10% damping ratio, never shrink to it.
    wheel = wheel*uniform(1., cfg.caster_damping_scale_range, (n, 4))
    swivel = swivel*uniform(1., cfg.caster_damping_scale_range, (n, 4))
    friction = wheels['friction']*uniform(1., cfg.wheel_friction_scale_range, (n,))
    return dict(fraction=frac, mass_scale=mass_scale, payload=payload, com_xy=com_xy,
                wheel_damping=wheel, swivel_damping=swivel, friction=friction)


def loaded_cart_properties(default_mass, default_inertia, default_com, base_id, mass_scale, payload, com_xy):
    """Scale empty links' mass/inertia together, then add a box payload consistently.

    Inertias are full link-frame tensors about each COM, independent of the
    simulator's principal-axis orientation. Payload offset is relative to base COM.
    Combined COM and full off-diagonal parallel-axis terms are retained.
    """
    n = len(default_mass)
    masses = default_mass*mass_scale
    inertia = default_inertia.reshape(n, -1, 3, 3)*mass_scale[..., None, None]
    coms = default_com.clone()
    empty_base = masses[:, base_id].clone()
    total = empty_base + payload[:, 0]
    d = torch.cat((com_xy, torch.zeros_like(payload)), -1)
    eye = torch.eye(3, device=d.device, dtype=d.dtype)
    parallel = d.square().sum(-1)[:, None, None]*eye-d[:, :, None]*d[:, None, :]
    payload_box = torch.diag(d.new_tensor([.30**2+.20**2, .40**2+.20**2, .40**2+.30**2]))/12
    inertia[:, base_id] += (payload[:, :, None]*payload_box
        + (empty_base*payload[:, 0]/total)[:, None, None]*parallel)
    coms[:, base_id, :2] += payload/total[:, None]*com_xy
    masses[:, base_id] = total
    return masses, inertia.reshape_as(default_inertia), coms


def metadata(cfg, cart_params):
    w = cart_params['wheels']
    if cfg.profile == 'legacy':
        return {'profile': 'legacy', 'enabled': cfg.enabled,
            'robot_reference': 'homie_isaaclab.config.G1RoughCfg.domain_rand',
            'cart_link_mass_scale_range': [1., 1.],
            'cart_friction_full_range': list(cfg.wheel_friction_range),
            'wheel_damping_full_range': list(cfg.wheel_damping_range),
            'swivel_damping_full_range': list(cfg.swivel_damping_range),
            'cart_initial_fraction': 1., 'cart_full_level': 0,
            'cart_force_max_n': cfg.cart_force_max_n, 'grasp_physics_randomized': False,
            'note': 'Saved legacy distribution, including legacy inertia/COM setter order.'}
    return {'profile': cfg.profile, 'enabled': cfg.enabled,
        'robot_reference': 'homie_isaaclab.config.G1RoughCfg.domain_rand',
        'cart_link_mass_scale_range': list(cfg.cart_link_mass_scale_range),
        'cart_friction_full_range': [w['friction']*v for v in cfg.wheel_friction_scale_range],
        'wheel_damping_full_range': [cfg.wheel_damping_range[i]*cfg.caster_damping_scale_range[i] for i in (0,1)],
        'swivel_damping_full_range': [cfg.swivel_damping_range[i]*cfg.caster_damping_scale_range[i] for i in (0,1)],
        'cart_initial_fraction': cfg.cart_initial_fraction, 'cart_full_level': cfg.cart_full_level,
        'cart_push_delta_v_m_s': cfg.cart_push_delta_v_m_s,
        'grasp_physics_randomized': False,
        'note': 'Threshold jitter is not grasp compliance; fixed joints remain unchanged.'}
