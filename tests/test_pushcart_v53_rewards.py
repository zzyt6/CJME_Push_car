"""Motion response and conditional force shaping, without starting a simulator."""
from pathlib import Path
from types import SimpleNamespace
import math
import sys

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'HomieRL'))
from homie_pushcart_v53.cart_quality import CartQualityTracker, apply_quality_profile, rotate_planar
from homie_pushcart_v53.task_config import PushCartTaskCfg, task_cfg_from_run


def tracker(n=1):
    return CartQualityTracker(PushCartTaskCfg().quality, .02, n, dtype=torch.float64)


def sample(q, t, *, command=(.3, 0., 0.), ref=None, motion=True, force=True, disturbance=False,
           wrench_valid=True, context=True):
    n = len(q.ready)
    cmd = torch.tensor([command], dtype=torch.float64).expand(n, -1)
    ref = cmd if ref is None else torch.tensor([ref], dtype=torch.float64).expand(n, -1)
    vel = ref.clone()
    vel[:, 0] += .06 * math.sin(4 * math.pi * t) if motion else 0.
    f = torch.zeros_like(vel)
    f[:, 0] = 40 + (40 * math.sin(4 * math.pi * t) if force else 0.)
    kwargs = dict(command_c=cmd, disturbance_active=torch.full((n,), disturbance),
                  wrench_valid=torch.full((n,), wrench_valid)) if context else {}
    q.update(vel, ref, torch.zeros(n), f, torch.zeros_like(f), **kwargs)


def prime(q):
    for step in range(200):
        sample(q, step * .02)
    assert q.wrench_gate.min() == 1


def test_force_cost_requires_both_motion_ripple_and_force_ripple():
    totals = []
    for motion, force in ((True, True), (False, True), (True, False)):
        q = tracker()
        total = 0.
        for step in range(200):
            sample(q, step*.02, motion=motion, force=force)
            total += q.guarded_wrench_cost.item()
        totals.append(total)
    assert totals[0] > 1
    assert totals[1:] == [0., 0.]  # useful varying force and steady useful force stay free


@pytest.mark.parametrize('change', [dict(command=(0., 0., 0.)), dict(command=(-.3, 0., 0.)),
    dict(command=(.3, 0., .2)), dict(command=(.3, .1, 0.)), dict(command=(.6, 0., 0.)),
    dict(disturbance=True), dict(wrench_valid=False), dict(context=False)])
def test_force_guard_immediately_disables_on_manoeuvres_disturbances_or_unknown_context(change):
    q = tracker()
    prime(q)
    sample(q, 4.04, **change)
    assert q.wrench_gate.item() == 0.
    assert q.guarded_wrench_cost.item() == 0.
    assert q.axis_ripple_cost[:, 0].item() > 0  # protection never turns off primary motion costs


def test_hold_rise_and_transient_force_memory():
    q = tracker()
    prime(q)
    sample(q, 4.04, disturbance=True)
    for step in range(1, 120):
        # Large constant force after a protected pulse must not be a new oscillation.
        ref = torch.tensor([[.3, 0., 0.]], dtype=torch.float64)
        vel = ref + torch.tensor([[.1*math.sin(step), 0., 0.]])
        f = torch.tensor([[300., 0., 0.]], dtype=torch.float64)
        q.update(vel, ref, torch.zeros(1), f, torch.zeros_like(f), command_c=ref,
                 disturbance_active=torch.tensor([step == 1]), wrench_valid=torch.tensor([True]))
        if step <= 40:
            assert q.wrench_gate.item() == 0.
        if step == 50:
            assert 0 < q.wrench_gate.item() < 1
        assert q.guarded_wrench_cost.item() == 0.
    assert q.wrench_gate.item() == 1.


def test_exact_following_is_free_through_start_stop_reverse_and_turn():
    q = tracker()
    ref = torch.zeros(1, 3, dtype=torch.float64)
    yaw = torch.zeros(1, dtype=torch.float64)
    for step in range(500):
        cmd = torch.tensor([((.4, 0., 0.), (.4, 0., .3), (0., 0., 0.), (-.3, 0., -.2),
                             (.3, .1, 0.))[step // 100]], dtype=torch.float64)
        ref += .05*(cmd-ref)
        yaw += .02*ref[:, 2]
        f = torch.full_like(ref, 150*math.sin(step))
        q.update(ref, ref, yaw, f, f, command_c=cmd, disturbance_active=torch.tensor([False]),
                 wrench_valid=torch.tensor([True]))
        assert q.axis_ripple_cost.max() == 0
        assert q.response_cost.item() == 0
        assert q.guarded_wrench_cost.item() == 0


def test_forward_lateral_yaw_are_independent_and_capped():
    for axis in range(3):
        q = tracker()
        totals = torch.zeros(3)
        for step in range(100):
            z = torch.zeros(1, 3, dtype=torch.float64)
            v = z.clone()
            v[:, axis] = 100*math.sin(step)
            q.update(v, z, torch.zeros(1), z, z)
            assert q.axis_ripple_cost.max() <= q.cfg.cost_cap
            totals += q.axis_ripple_cost[0]
        assert totals[axis] > 0
        assert totals.sum() == totals[axis]


def test_partial_reset_and_invalid_sample_close_guard_without_poisoning_motion():
    q = tracker(2)
    prime(q)
    q.reset(torch.tensor([0]))
    assert q.wrench_gate.tolist() == [0., 1.]
    assert q.guard_initialized.tolist() == [False, True]
    ref = torch.tensor([[.3, 0., 0.]]*2, dtype=torch.float64)
    f = torch.zeros_like(ref)
    f[0, 0] = float('nan')
    q.update(ref, ref, torch.zeros(2), f, torch.zeros_like(f), command_c=ref,
             disturbance_active=torch.zeros(2, dtype=torch.bool), wrench_valid=torch.ones(2, dtype=torch.bool))
    assert not q.ready[0] and not q.guard_initialized[0]
    assert q.wrench_gate[0] == 0
    assert torch.isfinite(q.guarded_wrench_cost).all()
    sample(q, 4.2)
    assert torch.isfinite(q.response_cost).all() and q.steps.tolist() == [1, 202]


def test_world_filter_does_not_invent_force_ripple_on_heading_relabelling():
    q = tracker()
    error_w = torch.tensor([[.1, -.2, 0.]], dtype=torch.float64)
    force = torch.tensor([[40., 10., 0.]], dtype=torch.float64)
    zero = torch.zeros_like(force)
    for step in range(150):
        yaw = torch.tensor([step*.1], dtype=torch.float64)
        q.update(rotate_planar(error_w, yaw, inverse=True), zero, yaw, force, zero)
    torch.testing.assert_close(q.velocity_band, zero, atol=1e-12, rtol=0)
    torch.testing.assert_close(q.wrench_band, zero, atol=1e-12, rtol=0)


def test_profile_toggle_and_revision_one_resume(tmp_path):
    from homie_isaaclab.math_utils import class_to_dict
    cfg = apply_quality_profile(PushCartTaskCfg(), 'v53_wrench_guarded')
    assert cfg.rewards.scales.cart_wrench_guarded == -.005
    apply_quality_profile(cfg, 'v53')
    assert cfg.rewards.scales.cart_wrench_guarded == 0.
    assert cfg.rewards.scales.force_rate == cfg.rewards.scales.cart_wrench_osc == 0.
    saved = class_to_dict(cfg)
    saved['implementation_revision'] = 1
    for key in ('cart_vx_ripple', 'cart_vy_ripple', 'cart_wrench_guarded'):
        del saved['rewards']['scales'][key]
    saved['rewards']['scales']['cart_lin_ripple'] = -.1
    saved['quality'].pop('wrench_interval_mean')
    (tmp_path/'params').mkdir()
    (tmp_path/'params/env.yaml').write_text(yaml.safe_dump({'task':saved}), encoding='utf-8')
    old = task_cfg_from_run(tmp_path, verbose=False)
    assert old.rewards.scales.cart_vx_ripple == old.rewards.scales.cart_vy_ripple == 0.
    assert old.rewards.scales.cart_lin_ripple == -.1
    assert not old.quality.wrench_interval_mean
    apply_quality_profile(old, 'v53_wrench_guarded')
    assert old.rewards.scales.cart_lin_ripple == 0. and old.quality.wrench_interval_mean


def test_total_motion_reward_prefers_prompt_tracking_over_slow_or_stationary_cart():
    from homie_pushcart_v53.task_env import PushCartEnv
    cfg = apply_quality_profile(PushCartTaskCfg(), 'v53_wrench_guarded')
    scores = []
    for lag in (0., .4, 100.):
        q = tracker()
        obj = SimpleNamespace(task_cfg=cfg, cart_quality=q, cmd_ref_f=torch.zeros(1, 3),
                              cart_vel_f=torch.zeros(1, 3), _quality_reward_ramp=lambda: 1.)
        score = 0.
        for step in range(300):
            t = step*.02
            ref = torch.tensor([[.3*(1-math.exp(-t/.4)), 0., 0.]])
            vel = torch.tensor([[.3*(1-math.exp(-max(t-lag, 0)/.4)), 0., 0.]])
            # All candidates face the same varying force; it must not make idle win.
            f = torch.tensor([[100*math.sin(t*4*math.pi), 0., 0.]])
            q.update(vel, ref, torch.zeros(1), f, torch.zeros_like(f), command_c=torch.tensor([[.3, 0., 0.]]),
                     disturbance_active=torch.tensor([False]), wrench_valid=torch.tensor([True]))
            obj.cmd_ref_f += .1*(ref-obj.cmd_ref_f)
            obj.cart_vel_f += .1*(vel-obj.cart_vel_f)
            for term in ('track_cart_vx', 'track_cart_vy', 'track_cart_wz', 'track_progress',
                         'cart_vx_ripple', 'cart_vy_ripple', 'cart_yaw_ripple',
                         'cart_response_error', 'cart_wrench_guarded'):
                score += float(getattr(PushCartEnv, '_reward_'+term)(obj))*getattr(cfg.rewards.scales, term)*.02
        scores.append(score)
    assert scores[0] > scores[1] > scores[2]


def test_bad_guard_parameters_fail_fast():
    for name, value in (('wrench_guard_rise_s', 0.), ('wrench_guard_hold_s', float('nan')),
                        ('wrench_guard_ref_accel_max', [0., .1, .1])):
        cfg = PushCartTaskCfg().quality
        setattr(cfg, name, value)
        with pytest.raises(ValueError):
            CartQualityTracker(cfg, .02, 1)
