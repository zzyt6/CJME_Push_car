"""Relative uncertainty, inertial consistency, sensor curriculum and nominal-mode checks."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'HomieRL'))
from homie_pushcart_v53.domain_randomization import (cart_fraction, sample_cart, loaded_cart_properties,
                                                    validate_randomization, profile_overrides, metadata)
from homie_pushcart_v53.task_config import PushCartTaskCfg, task_cfg_from_run
from homie_pushcart_v53.heading_sensor import SimulatedVisualHeading
from homie_pushcart_v53.observation import make_proprioceptive_sample
from homie_isaaclab.config import G1RoughCfg

PARAMS = yaml.safe_load((ROOT/'HomieRL/homie_pushcart_v52/config/cart_params.yaml').read_text(encoding='utf-8'))


def draw(level=17, n=4096, enabled=True, fixed=None):
    return sample_cart(PushCartTaskCfg().randomization, PARAMS, level, 25., n, 9,
                       device='cpu', enabled=enabled, fixed_payload=fixed)


def test_full_ranges_follow_homie_ratios_without_shrinking_old_damping_range():
    cfg, homie = PushCartTaskCfg(), G1RoughCfg()
    assert cfg.randomization.cart_link_mass_scale_range == homie.domain_rand.link_mass_range
    assert cfg.randomization.wheel_friction_scale_range == homie.domain_rand.friction_range
    assert cfg.randomization.caster_damping_scale_range == homie.domain_rand.kd_range
    assert [cfg.observations.joint_position_noise_rad, cfg.observations.joint_velocity_noise_rad_s,
            cfg.observations.gyro_noise_rad_s, cfg.observations.gravity_noise] == [
        homie.noise.noise_scales.dof_pos, homie.noise.noise_scales.dof_vel,
        homie.noise.noise_scales.ang_vel, homie.noise.noise_scales.gravity]
    s = draw()
    assert .8 <= s['mass_scale'].min() < .805 and 1.195 < s['mass_scale'].max() <= 1.2
    assert .08 <= s['friction'].min() < .10 and 2.35 < s['friction'].max() <= 2.4
    assert s['wheel_damping'].min() < .002 and s['wheel_damping'].max() > .02
    assert s['swivel_damping'].min() < .02 and s['swivel_damping'].max() > .15
    assert (s['wheel_damping'].std(dim=1) > 0).all()
    assert (s['swivel_damping'].std(dim=1) > 0).all()
    assert s['com_xy'][:, 0].abs().max() <= .196001
    assert s['com_xy'][:, 1].abs().max() <= .104001


def test_curriculum_support_is_nested_and_reaches_full():
    cfg = PushCartTaskCfg().randomization
    fractions = [cart_fraction(cfg, level) for level in range(18)]
    assert fractions[0] == .25 and fractions[13:] == [1.]*5
    assert sorted(fractions) == fractions
    s = draw(level=0)
    assert s['mass_scale'].min() >= .95 and s['mass_scale'].max() <= 1.05
    assert s['friction'].min() >= .62 and s['friction'].max() <= 1.20


def test_no_randomization_returns_exact_nominal_except_explicit_load():
    for fixed in (None, 12.):
        s = draw(n=8, enabled=False, fixed=fixed)
        assert s['fraction'] == 0 and (s['mass_scale'] == 1).all()
        assert (s['payload'] == (0 if fixed is None else fixed)).all()
        assert not s['com_xy'].any()
        assert (s['friction'] == .8).all()
        assert (s['wheel_damping'] == .005).all()
        assert (s['swivel_damping'] == .05).all()


def test_loaded_cart_conserves_mass_com_and_full_parallel_axis_inertia():
    m = torch.tensor([[12., .4]], dtype=torch.float64)
    inertia = torch.diag(torch.tensor([1., 2., 2.5], dtype=torch.float64)).repeat(1, 2, 1, 1)
    com = torch.zeros(1, 2, 7, dtype=torch.float64)
    com[..., 6] = 1
    scale = torch.tensor([[1.2, .8]], dtype=torch.float64)
    payload = torch.tensor([[10.]], dtype=torch.float64)
    xy = torch.tensor([[.2, -.1]], dtype=torch.float64)
    masses, actual, centers = loaded_cart_properties(m, inertia, com, 0, scale, payload, xy)
    torch.testing.assert_close(masses, torch.tensor([[24.4, .32]], dtype=torch.float64))
    torch.testing.assert_close(centers[0, 0, :2], xy[0]*10/24.4)
    torch.testing.assert_close(actual[0, 1], inertia[0, 1]*.8)
    assert actual[0, 0, 0, 1] == pytest.approx(14.4*10/24.4*.02)
    eig = torch.linalg.eigvalsh(actual)
    assert (eig > 0).all() and (eig[..., 2] <= eig[..., :2].sum(-1)).all()
    # No in-place corruption of defaults across repeated resets.
    assert m[0, 0] == 12. and inertia[0, 0, 0, 0] == 1. and centers[0, 1].equal(com[0, 1])


def test_proprio_noise_bounds_scale_with_curriculum_and_can_be_fully_disabled():
    cfg = PushCartTaskCfg().observations
    data = (torch.zeros(8192, 27), torch.zeros(8192, 27), torch.zeros(8192, 3), torch.zeros(8192, 3))
    for fraction in (0., .25, 1.):
        samples = make_proprioceptive_sample(*data, cfg, noise_fraction=fraction)
        for s, bound in zip(samples, (.02, 2., .5, .05)):
            assert s.abs().max() <= bound*fraction
            if fraction:
                assert s.abs().max() > bound*fraction*.98
    for s in make_proprioceptive_sample(*data, cfg, noise_enabled=False):
        assert not s.any()


def test_vision_bias_is_persistent_delay_is_scaled_and_partial_reset_is_local():
    cfg = PushCartTaskCfg().observations
    sensor = SimulatedVisualHeading(128, .02, cfg)
    sensor.reset(slice(None), True, fraction=1.)
    saved = sensor.bias.clone()
    assert saved.abs().max() <= cfg.heading_bias_rad and saved.std() > 0
    assert sensor.delay.max() == 4
    for _ in range(10):
        sensor.sample(torch.zeros(128))
        assert sensor.bias.equal(saved)
    sensor.reset(torch.arange(64), True, fraction=0.)
    assert not sensor.delay[:64].any() and not sensor.bias[:64].any()
    assert sensor.bias[64:].equal(saved[64:])
    yaw, valid, age = sensor.sample(torch.zeros(128))
    assert not yaw[:64].any() and valid[:64].all() and not age[:64].any()


def test_old_checkpoint_restores_legacy_domain_and_explicit_migration_changes_it(tmp_path):
    from homie_isaaclab.math_utils import class_to_dict
    saved = class_to_dict(PushCartTaskCfg())
    saved['implementation_revision'] = 2
    saved['randomization'].pop('profile')
    saved['observations'].pop('curriculum_noise')
    saved['observations'].pop('heading_bias_rad')
    saved['observations']['joint_velocity_noise_rad_s'] = .02
    (tmp_path/'params').mkdir()
    (tmp_path/'params/env.yaml').write_text(yaml.safe_dump({'task':saved}), encoding='utf-8')
    restored = task_cfg_from_run(tmp_path, verbose=False)
    assert restored.randomization.profile == 'legacy' and not restored.observations.curriculum_noise
    assert restored.observations.heading_bias_rad == 0 and restored.observations.joint_velocity_noise_rad_s == .02
    assert metadata(restored.randomization, PARAMS)['cart_friction_full_range'] == [.4, 1.2]
    for path, value in profile_overrides('homie_relative').items():
        section, key = path.split('.')
        setattr(getattr(restored, section), key, value)
    assert restored.randomization.profile == 'homie_relative'
    assert restored.observations.curriculum_noise and restored.observations.joint_velocity_noise_rad_s == 2.


def test_changed_domain_discards_old_curriculum_statistics_on_resume():
    from homie_pushcart_v53.checkpointing import curriculum_state, restore_curriculum, WINDOW_TENSORS
    from homie_pushcart_v52.curriculum import CurriculumTracker
    def environment():
        cfg = PushCartTaskCfg()
        obj = SimpleNamespace(task_cfg=cfg, curriculum=CurriculumTracker(cfg, 13), dt=.02,
            common_step_counter=100, _next_curriculum_eval=200, episodes_finished=10,
            term_counts=torch.zeros(2), episode_length_buf=torch.ones(2))
        for name in WINDOW_TENSORS:
            setattr(obj, name, torch.ones(3) if name in ('_track_sum', '_gain_vr', '_gain_rr') else torch.tensor(1.))
        obj._apply_level = lambda params: None
        obj.reset = lambda: None
        return obj
    source, target = environment(), environment()
    source.task_cfg.randomization.profile = 'legacy'
    result = restore_curriculum(target, curriculum_state(source))
    assert result['schedule_changed'] and not result['completed_window_restored']
    assert target.curriculum.level == 13 and not target._count.any()


@pytest.mark.parametrize('key,value', [('cart_link_mass_scale_range', [0., 1.]),
    ('wheel_friction_scale_range', [3., .1]), ('cart_full_level', 0),
    ('payload_com_deck_fraction', [.8, .2]), ('profile', 'typo')])
def test_bad_configuration_fails_before_physics(key, value):
    cfg = PushCartTaskCfg().randomization
    setattr(cfg, key, value)
    with pytest.raises(ValueError):
        validate_randomization(cfg)
