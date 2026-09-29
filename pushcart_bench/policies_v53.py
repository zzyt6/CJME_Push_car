"""V5.3 MuJoCo control: same observable builder and learned heads as Isaac.

Only the physical plant/instrumentation and read-only diagnostics are reused
from V5.x. FixedGraspObserver is neither constructed nor called.
"""
from pathlib import Path
import numpy as np
import torch

from .policies_v5x import InstrumentedEnv, V5xRunner, V5xPolicy
from .policies import CartPolicyBase
from homie_pushcart_v52.curriculum import level_params
from homie_pushcart_v52.lowlevel import HimLowLevelPolicy, frozen_checkpoint
from homie_pushcart_v53.env_smoke import make_configs
from homie_pushcart_v53.task_config import num_actor_obs, num_critic_obs
from homie_pushcart_v53.observation import FRAME_DIM, HAND_WRENCH_SCALE, RecoveryAuthority, body_to_cart_heading
from homie_pushcart_v53.deploy_observation import ObservableHistory, command_in_base, level_rotation
from homie_pushcart_v53.heading_sensor import SimulatedVisualHeading
from homie_pushcart_v53.cart_quality import CartQualityTracker


class V53Runner(V5xRunner):
    def __init__(self, checkpoint=None, *, level=0, homie_checkpoint=None, actor=None, scene_version='original'):
        cfg, tc = make_configs(1, False, level, checkpoint=checkpoint)
        if tc.actions.upper_drive != 'explicit_pd':
            raise ValueError('V5.3 all-joint torque observations require explicit_pd')
        self.task_cfg, self.package = tc, 'homie_pushcart_v53'
        self.env = env = InstrumentedEnv(True, scene_profile='isaac', cfg=cfg,
                                        decimation=cfg.control.decimation, upper_drive='explicit_pd', inertia_profile='urdf',
                                        scene_version=scene_version)
        self.scene_version = env.scene_version
        from .sim2sim_scene import set_grasp_profile
        set_grasp_profile(env, 'nominal')
        self.homie_checkpoint = Path(homie_checkpoint or getattr(cfg, 'pushcart_saved_checkpoint', None) or frozen_checkpoint())
        self.lowlevel = HimLowLevelPolicy(self.homie_checkpoint, device='cpu')
        from homie_pushcart_v53.hl_policy import HighLevelPolicy, import_rsl_rl5
        import_rsl_rl5()
        if checkpoint is not None:
            self.hl = HighLevelPolicy(Path(checkpoint), num_actor_obs(tc), num_critic_obs(tc), 18, 'cpu')
            self.actor = self.hl.actor
        elif actor is not None:
            self.actor = actor.cpu().eval()
        else:
            raise ValueError('Provide a V5.3 checkpoint, or an explicit actor for runtime tests')
        from tensordict import TensorDict
        self._TensorDict = TensorDict
        self.dt, self.frame_dim, self.history, self.num_actions = env.dt, FRAME_DIM, tc.env.actor_history, 18
        self.residual_scale = np.array(tc.actions.cmd_residual_scale)
        self.cmd_lo, self.cmd_hi = np.array(tc.actions.homie_cmd_limits).T
        self.cmd_alpha = min(1., self.dt/tc.actions.cmd_filter_tau_s) if tc.actions.cmd_filter_tau_s else 1.
        self.ref_alpha = min(1., self.dt/tc.commands.ref_tau_s) if tc.commands.ref_tau_s else 1.
        self.gate_speed = tc.actions.idle_gate_speed
        self.level, self.lp = level, level_params(level, tc)
        self.upper_scale = np.array([self.lp.waist_scale]+[self.lp.arm_scale]*14)
        self.step_lim = tc.actions.upper_rate_limit_rad_s*self.dt
        self.hold_target_q = env.preset_dof_pos[12:]+env.upper_hold_offset
        self.upper_lo = env.default_dof_pos[12:]+env.action_min[12:]*env.action_scale
        self.upper_hi = env.default_dof_pos[12:]+env.action_max[12:]*env.action_scale
        self.commands_scale = env.commands_scale
        self.cart_body, self.pelvis, self.bar_local = env.cart_id, env.pelvis_id, env.bar_local
        self.swivel_qpos = [env.model.joint(f'caster_{i}_swivel_joint').qposadr[0] for i in range(4)]
        self.wheel_body_ids = [env.model.body(f'wheel_{i}').id for i in range(4)]
        self._default_inertia = env.model.body_inertia[self.cart_body].copy()
        self.quality = CartQualityTracker(tc.quality, self.dt, 1, dtype=torch.float64)
        self.recovery = RecoveryAuthority(tc.recovery, self.dt, 1, 'cpu')
        sc = env.obs_scales
        self.observable = ObservableHistory(1, self.history, q_scale=sc['dof_pos'], dq_scale=sc['dof_vel'],
            gyro_scale=sc['ang_vel'], velocity_scale=sc['lin_vel'], torque_scale=tc.observations.torque_scale,
            clip=env.clip_obs, max_heading_age_s=tc.observations.heading_max_age_s)
        self.heading_sensor = SimulatedVisualHeading(1, self.dt, tc.observations)
        self.hand_scale = np.tile(np.array(HAND_WRENCH_SCALE), 2)
        self.reset_buffers()

    def reset_buffers(self):
        self.quality.reset()
        self.cmd_task, self.cmd_ref, self.cmd_homie = np.zeros(3), np.zeros(3), np.zeros(3)
        self.hl_actions, self.upper_delta_q, self.vel_est = np.zeros(18), np.zeros(15), np.zeros(3)
        self.env.upper_delta[:] = 0.
        self.observable.reset(slice(None))
        self.heading_sensor.reset(slice(None), False)
        self.recovery.reset(slice(None))
        self.episode_length = 0

    def _update_hl_obs(self, fresh):
        env = self.env
        f = lambda x: torch.as_tensor(np.asarray(x)[None], dtype=torch.float32)
        yaw, valid, age = self.heading_sensor.sample(torch.tensor([self.ts['rel_yaw']], dtype=torch.float32), False)
        self.hl_obs = self.observable.push(q=f(env.dof_pos), dq=f(env.dof_vel),
            torque=f(env.data.ctrl[env.act_idx]), gyro=f(env.base_ang_vel), gravity=f(env.projected_gravity),
            him_velocity=f(self.vel_est), relative_yaw=yaw, heading_valid=valid, heading_age_s=age,
            default_q=f(env.default_dof_pos), task_reference=f(self.cmd_ref), homie_command=f(self.cmd_homie),
            upper_target=f(self.hold_target_q+self.upper_delta_q), previous_high_action=f(self.hl_actions),
            previous_lower_action=f(env.actions[:12]), fresh=torch.tensor([fresh]))[0].numpy()
        self.heading_valid = bool(self.observable.valid[0])
        to_cart = body_to_cart_heading(level_rotation(f(env.projected_gravity)), self.observable.yaw)
        v = (to_cart@f(self.vel_est)[:, :, None])[:, :, 0]
        w = (to_cart@f(env.base_ang_vel)[:, :, None])[:, :, 0]
        self.recovery.update(torch.cat((v[:, :2], w[:, 2:3]), -1), f(self.cmd_ref),
            f(env.base_ang_vel), f(env.projected_gravity), self.observable.valid & torch.tensor([not fresh]))
        self._predict()

    @torch.inference_mode()
    def _predict(self):
        td = self._TensorDict({'policy':torch.from_numpy(self.hl_obs).unsqueeze(0)}, batch_size=[1])
        self.pred_hands = (self.actor.predict_hand_wrench(td)*self.actor.hand_scale)[0].numpy()
        self.pred_cart_velocity = (self.actor.predict_cart_velocity(td)/self.actor.cart_velocity_scale)[0].numpy()

    def step(self, cmd_task):
        tc, env = self.task_cfg, self.env
        self.cmd_task = np.asarray(cmd_task, dtype=float)
        with torch.inference_mode():
            td = self._TensorDict({'policy':torch.from_numpy(self.hl_obs).unsqueeze(0)}, batch_size=[1])
            self.hl_actions = np.clip(self.actor(td)[0].numpy(), -tc.actions.clip, tc.actions.clip)
        self.cmd_ref += self.ref_alpha*(self.cmd_task-self.cmd_ref)
        gate = min(1., np.linalg.norm(self.cmd_ref)/self.gate_speed) if self.gate_speed else 1.
        gate = max(gate, float(self.recovery.gate[0,0]))*float(self.heading_valid)
        reference = command_in_base(torch.tensor(self.cmd_ref[None],dtype=torch.float32),
                                    self.observable.yaw, self.observable.valid)[0].numpy()
        target = np.clip(reference+gate*self.hl_actions[:3]*self.residual_scale,self.cmd_lo,self.cmd_hi)
        self.cmd_homie += self.cmd_alpha*(target-self.cmd_homie)
        env.set_command(*self.cmd_homie)
        env.obs_hist[-1,:3] = env.commands[:3]*self.commands_scale
        desired = gate*self.hl_actions[3:]*self.upper_scale
        delta = self.upper_delta_q+np.clip(desired-self.upper_delta_q,-self.step_lim,self.step_lim)
        self.upper_delta_q = np.clip(self.hold_target_q+delta,self.upper_lo,self.upper_hi)-self.hold_target_q
        env.upper_delta[:] = self.upper_delta_q/env.action_scale
        low_obs = torch.from_numpy(np.clip(env.obs_hist.reshape(-1),-env.clip_obs,env.clip_obs)).unsqueeze(0)
        legs,_ = self.lowlevel.act_and_estimate(low_obs)
        self._obs,fallen = env.step(legs[0].cpu().numpy())
        self.vel_est = self._estimate_velocity(self._obs)
        self.episode_length += 1
        self.task_state()  # Ground truth remains diagnostic; vision adapter receives only relative yaw.
        self._update_hl_obs(False)
        return self.hl_obs,fallen


def resolve_v53_checkpoint(checkpoint='latest'):
    """Resolve only V5.3 logs; never accidentally select a V3/V4/V5.2 model."""
    if str(checkpoint) != 'latest':
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f'V5.3 checkpoint not found: {path}')
        return path
    root = Path(__file__).resolve().parents[1] / 'logs/rsl_rl/homie_pushcart_v53'
    runs = sorted((d for d in root.iterdir() if d.is_dir() and any(d.glob('model_*.pt'))), key=lambda d: d.name)
    if not runs:
        raise FileNotFoundError(f'No V5.3 checkpoints under {root}')
    return max(runs[-1].glob('model_*.pt'), key=lambda p: int(p.stem.rsplit('_', 1)[1]))


class V53Policy(V5xPolicy):
    """V5.3 adapter for the existing path/step/robust bench and its Viser viewer."""
    def __init__(self, checkpoint='latest', level=13, homie_checkpoint=None, *, scene_version='original', name=None):
        checkpoint = resolve_v53_checkpoint(checkpoint)
        runner = V53Runner(checkpoint, level=level, homie_checkpoint=homie_checkpoint, scene_version=scene_version)
        scene_label = '原始场景' if runner.scene_version == 'original' else '新版四球足底'
        name = name or f'V5.3 {scene_label} · {checkpoint.parent.name}/{checkpoint.stem}'
        self._initialize_runner(runner, checkpoint, level, name)

    def describe(self):
        from .sim2sim_scene import model_report
        out = CartPolicyBase.describe(self)
        out.update(kind='v53', package='homie_pushcart_v53', frame_dim=self.runner.frame_dim,
                   scene_version=self.runner.scene_version, static_prior=True,
                   cart_velocity_source='learned_observable_history', sim2sim=model_report(self.env),
                   observation_model='V5.3 shared observable history; nominal sensors, no observation noise or delay')
        out['cart_quality'].update(moment_origin='handle_center', sampling='physics_substep_interval_mean')
        return out

    def step(self, cmd):
        super().step(cmd)
        # Bench-only diagnostics. No quality/ground-truth value enters inference.
        r = self.runner
        resultant = np.mean(self.env.sub_handle, axis=0)
        tensor = lambda x: torch.as_tensor(np.asarray(x)[None], dtype=torch.float64)
        r.quality.update(tensor(self.velocity()), tensor(r.cmd_ref), torch.tensor([r.ts['cart_yaw']], dtype=torch.float64),
                         tensor(resultant[:3]), tensor(resultant[3:]))

    def telemetry(self):
        out = super().telemetry()
        r = self.runner
        out.update(dict(zip(('cart_est_vx', 'cart_est_vy', 'cart_est_wz'), map(float, r.pred_cart_velocity))))
        for side, hand in zip(('left', 'right'), r.pred_hands.reshape(2, 6)):
            out.update({f'est_{side}_{axis}': float(value) for axis, value in zip(('fx','fy','fz','mx','my','mz'), hand)})
        return out
