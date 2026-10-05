"""Measure frozen HOMIE/HIM body velocity in free and V3.2/V6 cart operation.

No learning or observation substitutions. The primary reference is pelvis-body
COM linear velocity in pelvis axes, matching HOMIE's Isaac training target.
Root-link-origin velocity is also recorded to quantify reference-point effects.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'HomieRL'), str(ROOT / 'scripts')]
LOW = ROOT / 'logs/rsl_rl/homie_g1/2026-09-14_15-10-07_my_policy/model_60000.pt'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def metrics(prediction, label):
    """Physical-unit error statistics shared by the independent HOMIE audits."""
    import numpy as np
    e = np.asarray(prediction, dtype=np.float64) - np.asarray(label, dtype=np.float64)
    if not len(e):
        return {'samples': 0}
    if not np.isfinite(e).all():
        raise FloatingPointError('Nonfinite estimation or tracking error')
    return dict(samples=len(e), mae=np.abs(e).mean(0).tolist(),
                rmse=np.sqrt(np.square(e).mean(0)).tolist(), bias=e.mean(0).tolist(),
                absolute_p95=np.quantile(np.abs(e), .95, axis=0).tolist())


def protocol(t, scenario):
    """Original 24-second HOMIE audit command sequence, independent of high-level versions."""
    import numpy as np
    if t < 2:
        return 'settle', np.zeros(3)
    if t < 6:
        return 'start', np.array([-.2 if scenario == 'reverse' else .3, 0., 0.])
    if t < 12 or 16 <= t < 21:
        vx = .6 if scenario == 'straight' else (-.35 if scenario == 'reverse' else .35)
        wz = .25 if scenario == 'left' else (-.25 if scenario == 'right' else 0.)
        return ('main' if t < 12 else 'restart_turn_reverse'), np.array([vx, 0., wz if t < 12 else -wz])
    return ('brake' if t < 16 else 'final_stop'), np.zeros(3)


def command(t, scenario):
    import numpy as np
    if scenario in ('lateral_left', 'lateral_right'):
        c = protocol(t, 'straight')[1]
        return np.array([0., c[0] * .5 * (1 if scenario == 'lateral_left' else -1), 0.])
    return protocol(t, scenario)[1]


def cases(mode):
    if mode == 'free':
        return [dict(scenario=s, payload_kg=None) for s in ('straight', 'left', 'right', 'reverse', 'lateral_left', 'lateral_right')]
    return [dict(scenario=s, payload_kg=m) for m in (5., 15., 25.) for s in ('straight', 'left', 'right', 'reverse')]


def config_free(count):
    from homie_isaaclab.config import G1RoughCfg
    c = G1RoughCfg()
    c.env.num_envs = count
    c.env.episode_length_s = 30.
    c.env.action_curriculum = False
    c.commands.curriculum = False
    c.commands.resampling_time = 1e6
    for k in dir(c.domain_rand):
        if k.startswith('randomize_') or k in ('use_random', 'push_robots', 'delay', 'upper_body_randomization'):
            setattr(c.domain_rand, k, False)
    c.noise.add_noise = False
    return c


def estimate_velocity(policy, obs):
    """Read the estimator without invoking/recording an extra action."""
    if hasattr(policy, 'estimate_velocity'):
        return policy.estimate_velocity(obs)
    model = policy.model
    return model.estimator(obs[:, :model.actor_proprioceptive_obs_length])[0]


def configure_nominal_cart(task):
    """Disable task-level noise/randomization too, beyond HOMIE's own switch."""
    from homie_pushcart.assets.cart_urdf import load_cart_params
    wheels = load_cart_params()['wheels']
    task.curriculum.enabled = False
    task.actions.delay_steps_max = 0
    for key in ('cart_pose_noise_pos_m', 'cart_pose_noise_yaw_rad', 'cart_pose_delay_steps_max',
                'upper_torque_noise_frac', 'est_vel_noise_m_s'):
        setattr(task.observations, key, 0)
    rz = task.randomization
    rz.init_yaw = False
    rz.robot_push = False
    for key in ('init_xy_m', 'payload_com_offset_m', 'grasp_threshold_jitter',
                'cart_push_vel_max', 'wheel_stall_prob_per_s'):
        setattr(rz, key, 0.)
    for key in ('wheel_friction', 'wheel_damping', 'swivel_damping'):
        nominal = wheels['friction' if key == 'wheel_friction' else key]
        setattr(rz, key + '_range', [nominal, nominal])


def save(args, rows, metadata):
    import numpy as np
    arrays = {k: np.stack(v) for k, v in rows.items()}
    for k, v in arrays.items():
        if v.dtype.kind not in 'US' and not np.isfinite(v).all():
            raise FloatingPointError(k)
    valid = arrays['valid'].astype(bool) & (arrays['time'] >= 2.)
    # Keep complete groups rather than treating correlated timesteps as seeds.
    report = dict(result='PASS_EXECUTION', engine=args.engine, mode=args.mode,
                  low_checkpoint=str(LOW), low_sha256=sha(LOW), high_checkpoint=str(args.checkpoint) if args.mode == 'cart' else None,
                  high_sha256=sha(args.checkpoint) if args.mode == 'cart' else None,
                  cart_version=args.cart_version if args.mode == 'cart' else None,
                  axes=['vx', 'vy', 'vz'], units='m/s', reference='pelvis-body COM velocity in pelvis body axes',
                  warmup_s=2., observation_noise=False, domain_randomization=False,
                  current=metrics(arrays['prediction'][valid], arrays['truth_com'][valid]),
                  origin=metrics(arrays['prediction'][valid], arrays['truth_link'][valid]),
                  reference_point_difference=metrics(arrays['truth_link'][valid], arrays['truth_com'][valid]),
                  action_time_to_next=metrics(arrays['action_time_prediction'][valid], arrays['truth_com'][valid]),
                  cases=cases(args.mode), metadata=metadata,
                  source_sha256=sha(Path(__file__)), run_id=os.environ.get('HOMIE_BODY_AUDIT_CHILD'),
                  valid_samples=int(valid.sum()), falls_or_terminations=int(arrays['done'].sum()))
    if 'cart_prediction' in arrays:
        report['cart_speed'] = metrics(arrays['cart_prediction'][valid], arrays['cart_truth'][valid])
    np.savez_compressed(args.output / 'samples.npz', **arrays)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps({k: report[k] for k in ('result', 'engine', 'mode', 'current', 'action_time_to_next', 'reference_point_difference', 'falls_or_terminations')}, ensure_ascii=False), flush=True)


def run_isaac(args, app):
    import numpy as np
    import torch
    from homie_isaaclab.math_utils import quat_rotate_inverse
    from homie_pushcart.lowlevel import HimLowLevelPolicy
    cs = cases(args.mode)
    torch.set_num_threads(4)
    torch.manual_seed(20260928)
    np.random.seed(20260928)
    if args.mode == 'free':
        from homie_isaaclab.env import HomieIsaacLabEnv
        cfg = config_free(len(cs))
        env = HomieIsaacLabEnv(cfg, ROOT, args.device, True)
        low = HimLowLevelPolicy(LOW, args.device)
        env._resample_commands = lambda ids: None
        obs, _ = env.reset()
        env.commands.zero_(); env.commands[:, 4] = cfg.rewards.base_height_target
        env.obs_buf[:, -env.num_one_step_obs:] = 0.
        env.compute_observations()
        obs = env.obs_buf
        filtered = torch.zeros(len(cs), 3, device=args.device)
    else:
        if args.cart_version == 'v6':
            from homie_pushcart_v6.task_env import PushCartEnv
            from homie_pushcart_v6.hl_policy import HighLevelPolicy
            from homie_pushcart_v6.task_config import task_cfg_from_run
            from homie_pushcart_v6.baseline import apply_config, homie_baseline
        else:
            from homie_pushcart.task_env import PushCartEnv
            from homie_pushcart.hl_policy import HighLevelPolicy
            from homie_pushcart.task_config import task_cfg_from_run
        cfg = config_free(len(cs))
        if args.cart_version == 'v6':
            apply_config(cfg, homie_baseline())
            cfg.env.num_envs = len(cs)
            for key in dir(cfg.domain_rand):
                if key.startswith('randomize_') or key in ('use_random', 'push_robots', 'delay', 'upper_body_randomization'):
                    setattr(cfg.domain_rand, key, False)
            cfg.noise.add_noise = False
        task = task_cfg_from_run(args.checkpoint)
        configure_nominal_cart(task)
        task.env.episode_length_s = max(30., args.steps * cfg.sim.dt * cfg.control.decimation + 1.)
        env = PushCartEnv(cfg, task, ROOT, args.device, True, level=args.level, checkpoint=LOW)
        high = HighLevelPolicy(args.checkpoint, env.num_hl_obs, env.num_hl_critic_obs, 18, args.device)
        low = env.policy
        original = env._randomize_cart
        payloads = torch.tensor([c['payload_kg'] for c in cs], device=args.device)
        def fixed(ids):
            previous = env.fixed_payload_kg
            for payload in (5., 15., 25.):
                subset = ids[payloads[ids.long()] == payload]
                if len(subset):
                    env.fixed_payload_kg = payload
                    original(subset)
            env.fixed_payload_kg = previous
        env._randomize_cart = fixed
        env.fixed_command = [0., 0., 0.]
        env.fixed_caster_angle = 0.
        obs, critic = env.reset()
    assert not cfg.noise.add_noise and not cfg.domain_rand.use_random
    original_act = low.act_and_estimate
    action_prediction = None
    def record_action(obs):
        nonlocal action_prediction
        action, v = original_act(obs)
        action_prediction = v / cfg.normalization.obs_scales.lin_vel
        return action, v
    low.act_and_estimate = record_action
    rows = {}
    with torch.inference_mode():
        for step in range(args.steps):
            if not app.is_running():
                raise RuntimeError('Simulator closed early')
            cmd = torch.as_tensor(np.stack([command(step * env.dt, c['scenario']) for c in cs]), device=args.device, dtype=torch.float32)
            if args.mode == 'free':
                filtered += min(1., env.dt / .4) * (cmd - filtered)
                env.commands[:, :3] = filtered
                env.commands[:, 4] = cfg.rewards.base_height_target
                env.obs_buf[:, -env.num_one_step_obs:-env.num_one_step_obs + 3] = filtered * env.commands_scale
                action, _ = low.act_and_estimate(env.obs_buf)
                obs, _, _, done, _, _, _ = env.step(action)
                prediction = estimate_velocity(low, env.obs_buf) / cfg.normalization.obs_scales.lin_vel
            else:
                env.cmd_task.copy_(cmd); env.cmd_timer.fill_(1000000)
                action = high(obs, env.estimator_obs) if args.cart_version == 'v6' else high(obs, critic)
                obs, critic, _, done, _ = env.step(action)
                prediction = estimate_velocity(low, env.obs_buf) / cfg.normalization.obs_scales.lin_vel
            q = env.base_quat
            truth = env.base_lin_vel
            link = quat_rotate_inverse(q, env.robot.data.root_link_vel_w.torch[:, :3])
            values = dict(time=torch.full((len(cs),), (step + 1) * env.dt, device=args.device),
                          prediction=prediction, action_time_prediction=action_prediction,
                          truth_com=truth, truth_link=link, command=env.commands[:, :3],
                          low_observation=env.obs_buf, q=env.dof_pos, dq=env.dof_vel,
                          gravity=env.projected_gravity, gyro=env.base_ang_vel,
                          episode_step=env.episode_length_buf, done=done,
                          valid=(env.episode_length_buf > 6) & ~done,
                          robot_position=env.root_states[:, :3] - env.env_origins)
            if args.mode == 'cart' and args.cart_version == 'v6':
                values['cart_prediction'] = high.physical_estimates(obs, env.estimator_obs)[0]
                values['cart_truth'] = torch.cat((env.ts.cart_v, env.ts.cart_wz[:, None]), -1)
            for k, v in values.items():
                rows.setdefault(k, []).append(v.detach().cpu().numpy().copy())
            if step % 200 == 0:
                print(f'[HIM body] Isaac {args.mode} step {step}/{args.steps}', flush=True)
    save(args, rows, dict(num_envs=len(cs), control_dt=env.dt, physics_dt=cfg.sim.dt,
                         upper_drive='native HOMIE mixed',
                         label_consistency_max=float((env.base_lin_vel - quat_rotate_inverse(env.base_quat, env.robot.data.root_com_vel_w.torch[:, :3])).abs().max())))


def free_mujoco(args):
    from homie_pushcart.mujoco_env import MujocoHomieEnv
    cfg = config_free(1)
    # The original robot-only asset already has the four-sphere feet and no cart.
    # Use its legacy dynamics directly; no alternative plant or XML mutation.
    env = MujocoHomieEnv(False, cfg=cfg)
    env.upper_hold_actions[:] = 0.
    env.upper_hold_offset[:] = 0.
    env.preset_dof_pos[:] = env.default_dof_pos
    env.preset[env.qpos_adr] = env.default_dof_pos
    env.preset[env.root_qpos:env.root_qpos + 3] = cfg.init_state.pos
    env.preset[env.root_qpos + 3:env.root_qpos + 7] = [1., 0., 0., 0.]
    return env, env.xml_path


def run_mujoco(args):
    import mujoco
    import numpy as np
    import torch
    from homie_pushcart.mujoco_env import quat_rotate_inverse_wxyz
    from homie_pushcart.lowlevel import HimLowLevelPolicy
    torch.set_num_threads(1)
    torch.manual_seed(20260928); np.random.seed(20260928)
    cs, records = cases(args.mode), []
    if args.mode == 'cart':
        if args.cart_version == 'v6':
            from homie_pushcart_v6.mujoco_runner import MujocoV6Runner
            runner = MujocoV6Runner(args.checkpoint, args.level, LOW)
        else:
            from types import SimpleNamespace
            from homie_pushcart.mujoco_hl import make_runner
            runner, _ = make_runner(SimpleNamespace(checkpoint=args.checkpoint, level=args.level, homie_checkpoint=LOW))
        env, low = runner.env, runner.lowlevel
        source = env.xml_path
    else:
        env, source = free_mujoco(args)
        low = HimLowLevelPolicy(LOW, 'cpu')
    scale = env.obs_scales['lin_vel']
    original_act = low.act_and_estimate
    action_prediction = None
    def record_action(obs):
        nonlocal action_prediction
        act, v = original_act(obs)
        action_prediction = v[0].numpy() / scale
        return act, v
    low.act_and_estimate = record_action
    for ci, case in enumerate(cs):
        if args.mode == 'cart':
            runner.set_payload(case['payload_kg']); runner.reset(caster_angle=0.)
        else:
            env.reset(); env.set_command(0., 0., 0.)
        filtered = np.zeros(3)
        rows = {}
        for step in range(args.steps):
            cmd = command(step * env.dt, case['scenario'])
            if args.mode == 'free':
                filtered += min(1., env.dt / .4) * (cmd - filtered)
                env.set_command(*filtered)
                env.obs_hist[-1, :3] = filtered * env.commands_scale
                act, _ = low.act_and_estimate(torch.from_numpy(env.obs_hist.reshape(1, -1)))
                low_obs, fallen = env.step(act[0].numpy())
                with torch.inference_mode():
                    prediction = estimate_velocity(low, torch.from_numpy(low_obs[None]))[0].numpy() / scale
            else:
                _, fallen = runner.step(cmd)
                with torch.inference_mode():
                    prediction = estimate_velocity(low, torch.from_numpy(env.obs_hist.reshape(1, -1)))[0].numpy() / scale
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_BODY, env.pelvis_id, velocity, 0)
            truth = quat_rotate_inverse_wxyz(env.base_quat_wxyz, velocity[3:])
            values = dict(time=(step + 1) * env.dt, prediction=prediction, action_time_prediction=action_prediction,
                          truth_com=truth, truth_link=env.base_lin_vel,
                          command=env.commands[:3], low_observation=env.obs_hist.reshape(-1),
                          q=env.dof_pos, dq=env.dof_vel, gravity=env.projected_gravity, gyro=env.base_ang_vel,
                          episode_step=step + 1, done=fallen, valid=step >= 6 and not fallen,
                          robot_position=env.data.qpos[env.root_qpos:env.root_qpos + 3])
            if args.mode == 'cart' and args.cart_version == 'v6':
                values.update(cart_prediction=runner.pred_cart_velocity,
                              cart_truth=np.r_[runner.ts['cart_v'], runner.ts['cart_wz']])
            for k, v in values.items():
                rows.setdefault(k, []).append(np.asarray(v).copy())
            if fallen:
                # Do not hide failures or substitute a new trial mid-command.
                raise RuntimeError(f'MuJoCo fell in case {ci} at step {step}; no completed report')
        records.append(rows)
        print(f'[HIM body] MuJoCo {args.mode} complete {ci + 1}/{len(cs)}', flush=True)
    rows = {k: [np.stack([records[i][k][step] for i in range(len(cs))]) for step in range(args.steps)] for k in records[0]}
    save(args, rows, dict(num_envs=len(cs), control_dt=env.dt, physics_dt=float(env.model.opt.timestep),
                         scene_source=str(source), scene_sha256=sha(source), mujoco_version=mujoco.__version__,
                         free_model_note='Original legacy four-sphere robot-only asset, default HOMIE pose' if args.mode == 'free' else None,
                         physics_backend='legacy_four_spheres', upper_drive='legacy explicit PD'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', choices=('isaac', 'mujoco'), required=True)
    p.add_argument('--mode', choices=('free', 'cart'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--cart_version', choices=('v3.2', 'v6'), default='v6', help='High-level version for --mode cart only')
    p.add_argument('--checkpoint', type=Path, help='Explicit V3.2/V6 checkpoint; required only for --mode cart')
    p.add_argument('--level', type=int, default=13)
    a = p.parse_args()
    if a.steps < 1:
        p.error('--steps must be positive')
    if a.mode == 'cart' and (a.checkpoint is None or not a.checkpoint.is_file()):
        p.error('--mode cart requires an existing --checkpoint for --cart_version')
    if a.engine == 'isaac' and not os.environ.get('HOMIE_BODY_AUDIT_CHILD'):
        token = uuid.uuid4().hex
        child = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:]],
                               env=dict(os.environ, HOMIE_BODY_AUDIT_CHILD=token))
        path = a.output / 'report.json'
        result = json.loads(path.read_text()) if path.exists() else {}
        passed = child.returncode == 0 and result.get('run_id') == token and result.get('result') == 'PASS_EXECUTION'
        if result.get('run_id') == token:
            result.update(parent_verified=passed, child_exit_code=child.returncode)
            path.write_text(json.dumps(result, indent=2), encoding='utf8')
        print(f'[HIM body parent] exit={child.returncode} verified={passed}', flush=True)
        raise SystemExit(0 if passed else 1)
    a.output.mkdir(parents=True, exist_ok=False)
    checkpoints = [LOW] + ([a.checkpoint] if a.mode == 'cart' else [])
    before = [sha(path) for path in checkpoints]
    if a.engine == 'isaac':
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(headless=True, device=a.device)
        try:
            run_isaac(a, launcher.app)
            assert before == [sha(path) for path in checkpoints]
        except BaseException:
            import traceback
            traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush()
            launcher.app.close(exit_code=1)
            raise
        else:
            launcher.app.close()
    else:
        run_mujoco(a)
        assert before == [sha(path) for path in checkpoints]


if __name__ == '__main__':
    main()
