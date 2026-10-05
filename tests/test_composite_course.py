"""Geometry and reachability of the 20 m sinusoidal S / raised-detour course."""
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'HomieRL'), str(ROOT / 'tests')]
from pushcart_bench.paths import build_path, composite_course
from pushcart_bench.config_io import load_config
from pushcart_bench.follower import FollowerConfig
from test_pushcart_bench import simulate_unicycle


def test_start_and_finish_are_collinear_and_forward():
    p = build_path('composite_course', load_config()['paths']['composite_course'])
    np.testing.assert_allclose(p.xy[0], [0., 0.], atol=1e-12)
    assert p.goal[0] == pytest.approx(16.12226, abs=1e-3)
    np.testing.assert_allclose(p.goal[1:], [0., 0.], atol=1e-10)
    assert p.yaw[0] == pytest.approx(0.)
    assert np.min(np.diff(p.xy[:, 0])) >= -1e-9  # no hidden U-turn to restore alignment
    assert np.max(p.xy[:, 1]) == pytest.approx(2.7, abs=1e-6)
    assert np.min(p.xy[:, 1]) == pytest.approx(-.7, abs=1e-4)
    assert np.max(abs(np.diff(p.yaw))) < .02
    assert p.length == pytest.approx(20., abs=1e-9)
    assert abs(np.linalg.norm(np.diff(p.xy, axis=0), axis=1).sum() - 20.) < .001
    assert p.min_radius == pytest.approx(1.25, abs=.02)


def test_true_sine_has_positive_and_negative_lobes_and_gentle_radius():
    p = composite_course()
    s = (p.xy[:, 0] > .3) & (p.xy[:, 0] < 8.3)
    assert p.xy[s, 1].max() == pytest.approx(.7, abs=1e-4)
    assert p.xy[s, 1].min() == pytest.approx(-.7, abs=1e-4)
    assert 1. / np.abs(p.curvature[s]).max() == pytest.approx(1.483, abs=.01)
    assert math.radians(34.) < np.abs(p.yaw[s]).max() < math.radians(35.)
    # In the central phase interval this is exactly a sinusoid with constant phase speed.
    x = p.xy[:, 0]
    inner = (x > .3 + .22 * 8) & (x < .3 + .78 * 8)
    phase = ((x[inner] - .3) / 8 - .1) / .8
    np.testing.assert_allclose(p.xy[inner, 1], .7 * np.sin(2 * math.pi * phase), atol=1e-6)
    gap = (x > 8.34) & (x < 10.26)
    np.testing.assert_allclose(p.xy[gap, 1], 0., atol=1e-12)
    np.testing.assert_allclose(p.yaw[gap], 0., atol=1e-8)
    assert gap.sum() > 90


def test_raised_detour_has_four_rounded_quarter_turns():
    p = composite_course()
    top_length = p.goal[0] - (.3 + 8 + 2 + 4 * 1.25 + .3)
    top = (p.xy[:, 0] > 12.85) & (p.xy[:, 0] < 12.75 + top_length)
    up = (abs(p.xy[:, 0] - 11.55) < 1e-8) & (p.xy[:, 1] > 1.28) & (p.xy[:, 1] < 1.42)
    down = (abs(p.xy[:, 0] - (14.05 + top_length)) < 1e-8) & (p.xy[:, 1] > 1.28) & (p.xy[:, 1] < 1.42)
    end = p.xy[:, 0] > p.goal[0] - .2
    np.testing.assert_allclose(p.yaw[top], 0., atol=1e-8)
    np.testing.assert_allclose(p.yaw[up], math.pi / 2, atol=1e-8)
    np.testing.assert_allclose(p.yaw[down], -math.pi / 2, atol=1e-8)
    np.testing.assert_allclose(p.yaw[end], 0., atol=1e-8)
    assert min(up.sum(), down.sum()) >= 5
    turns = (abs(p.curvature) > .7) & (p.xy[:, 0] > 10.3)
    starts = np.flatnonzero(np.diff(np.r_[False, turns].astype(int)) == 1)
    ends = np.flatnonzero(np.diff(np.r_[turns, False].astype(int)) == -1)
    assert len(starts) == 4
    for start, end, sign in zip(starts, ends, [1., -1., -1., 1.]):
        assert np.mean(p.curvature[start + 2:end - 2]) == pytest.approx(sign / 1.25, abs=.01)


def test_same_geometry_supports_world_transform_and_mirror():
    p = composite_course()
    mirror = p.mirrored()
    np.testing.assert_allclose(mirror.xy[:, 1], -p.xy[:, 1])
    theta = .7
    origin = np.array([3., -2.])
    world = p.transformed(origin, theta)
    displacement = world.xy[-1] - world.xy[0]
    np.testing.assert_allclose(displacement, p.goal[0] * np.array([math.cos(theta), math.sin(theta)]), atol=1e-10)
    assert world.goal[2] == pytest.approx(theta)


def test_existing_follower_can_complete_composite_geometry():
    p = composite_course()
    cfg = FollowerConfig.from_dict({**load_config()['follower'], 'timeout_s': 140.})
    trace, follower = simulate_unicycle(p, cfg)
    assert follower.state.reached, follower.info()
    assert max(abs(np.asarray(trace.cross_track))) < .4


@pytest.mark.parametrize('kwargs', [{'height': 2.5}, {'corner_radius': 0.}, {'ds': float('nan')}, {'total_length': 12.}])
def test_degenerate_geometry_rejected(kwargs):
    with pytest.raises(ValueError):
        composite_course(**kwargs)
