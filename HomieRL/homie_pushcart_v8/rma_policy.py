"""V8 final design: an RMA-style teacher whose policy sees the privileged information ONLY through a latent, trained with
Regularized Online Adaptation (ROA) so the latent stays recoverable from the deployable history; the deployable student
is the same base policy with the latent from the adaptation module (docs/push_cart_v8_teacher_student_plan.md, 11).

* RMA (Kumar et al., RSS 2021): environment-factor encoder mu: e -> z (MLP 256-128), base policy pi(x, z); phase 2
  trains an adaptation module phi(history) -> z_hat on the student's own rollouts with pi frozen.
* ROA (Fu, Cheng, Pathak, CoRL 2022 "Deep Whole-Body Control", Algorithm 1 / Appendix B; also Extreme Parkour, Cheng
  et al. 2023): mu, pi and phi train together.
      L = -J_PPO(pi, mu) + lambda * ||z_mu - sg(z_phi)||_2 + ||sg(z_mu) - z_phi||_2
  every H = 20th rollout acts on z_phi and only phi is updated. RMA is the special case lambda = 0. V8 caps lambda at
  0.1 (Extreme Parkour's value): above it the teacher itself degraded (MuJoCo 0.025 -> 0.053 m at lambda 0.2).
* Student RL (asymmetric actor-critic, Pinto et al. 2018): PPO fine-tuning of the deployable network with phi frozen and
  a privileged critic; kept short (~200 updates) -- longer runs over-fit Isaac (MuJoCo 25 kg degraded).

Here: x = the deployable 5-frame policy input (630), e = all 62 privileged values (no raw explicit input), z 20-dim
(ROA's size), phi = a 32-frame 1D-CNN adaptation module (RMA's temporal-convolution design).
"""
from __future__ import annotations

import copy

import torch
from torch import nn
from rsl_rl.algorithms import PPO

from .student_policy import AdaptationModule
from .task_config import ACTION_DIM, FRAME_DIM, LONG_HISTORY, POLICY_HISTORY, PRIVILEGED_DIM
from .teacher_policy import TeacherActor

POLICY_DIM = FRAME_DIM * POLICY_HISTORY
HISTORY_DIM = FRAME_DIM * LONG_HISTORY
RMA_SCHEMA = "v8_rma_policy630_priv62_latent20_history32x126_act30_v1"
# V8.1 velocity head: the adaptation module also outputs these privileged quantities (cart vx, vy, wz, robot vx, vy --
# the teacher's most-used inputs), supervised with the truth; diagnostics only, the policy still reads z alone.
VELOCITY_IDX = (0, 1, 2, 4, 5)


class RMATeacherActor(TeacherActor):
    """pi(normalized policy 630, z); z = mu(normalized privileged 62) or, with ``use_estimate``, phi(history)."""

    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(512, 256, 128), activation="elu",
                 obs_normalization=True, distribution_cfg=None, latent_dim=20, encoder_hidden_dims=(256, 128),
                 leg_gate_init=1.0, leg_init_std=0.1, leg_gate_learnable=False, adaptation_channels=64,
                 deploy=False, velocity_head=False):
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization,
                         distribution_cfg, latent_dim, encoder_hidden_dims, leg_gate_init, leg_init_std,
                         leg_gate_learnable)
        self.velocity_head = bool(velocity_head)
        self.adaptation = AdaptationModule(self.latent_dim, adaptation_channels,
                                           explicit_dim=len(VELOCITY_IDX) if self.velocity_head else 0)
        self.use_estimate = bool(deploy)       # rollout mode: ROA's H-th iterations, phase 2, student RL, deployment
        self.frozen_normalizers = False        # student RL: keep the trained input statistics
        self.adaptation_channels = int(adaptation_channels)

    def _get_obs_dim(self, obs, obs_groups, obs_set):
        if obs_groups[obs_set] != ["policy", "privileged", "history"]:
            raise ValueError("V8 RMA Actor groups must be ['policy', 'privileged', 'history'].")
        if (obs["policy"].shape[-1] != POLICY_DIM or obs["privileged"].shape[-1] != PRIVILEGED_DIM
                or obs["history"].shape[-1] != HISTORY_DIM):
            raise ValueError(f"V8 RMA Actor needs policy {POLICY_DIM}, privileged {PRIVILEGED_DIM}, history {HISTORY_DIM}.")
        return list(obs_groups[obs_set]), POLICY_DIM                     # normalized part: the policy input only

    def _get_latent_dim(self):
        return POLICY_DIM + self.latent_dim

    def z_privileged(self, obs):
        return self.encoder(self.privileged_normalizer(obs["privileged"]))

    def z_estimated(self, obs):
        return self.adaptation(obs["history"])[1]

    def velocity_target(self, obs):
        """The velocity head's target: the privileged velocities in the privileged normaliser's units."""
        p = self.privileged_normalizer(obs["privileged"])
        return p[:, list(VELOCITY_IDX)]

    def get_latent(self, obs, masks=None, hidden_state=None):
        z = self.z_estimated(obs) if self.use_estimate else self.z_privileged(obs)
        return torch.cat((self.obs_normalizer(obs["policy"]), z), -1)

    @torch.no_grad()
    def update_normalization(self, obs):
        if self.frozen_normalizers:
            return
        if self.obs_normalization:
            self.obs_normalizer.update(obs["policy"])
        self.privileged_normalizer.update(obs["privileged"])
        self.adaptation.update_normalization(obs["history"])

    def metadata(self):
        return {"schema": RMA_SCHEMA, "release": "8.0", "policy": POLICY_DIM, "privileged": PRIVILEGED_DIM,
                "history": HISTORY_DIM, "latent": self.latent_dim, "actions": ACTION_DIM}

    def as_jit(self):
        return RMAStudentExport(self) if self.use_estimate else RMATeacherExport(self)

    def as_onnx(self, verbose=False):
        return self.as_jit()


class RMATeacherExport(nn.Module):
    """Deterministic RMA teacher (simulation only): (policy 630, privileged 62) -> 30 actions."""

    def __init__(self, actor: RMATeacherActor):
        super().__init__()
        for name in ("obs_normalizer", "privileged_normalizer", "encoder", "mlp"):
            setattr(self, name, copy.deepcopy(getattr(actor, name)))
        self.deterministic = (actor.distribution.as_deterministic_output_module()
                              if actor.distribution is not None else nn.Identity())

    def forward(self, policy: torch.Tensor, privileged: torch.Tensor):
        z = self.encoder(self.privileged_normalizer(privileged))
        return self.deterministic(self.mlp(torch.cat((self.obs_normalizer(policy), z), -1)))


class RMAStudentExport(nn.Module):
    """Deployable RMA student: (policy 630, history 4032) -> 30 actions; the base policy with z = phi(history)."""

    def __init__(self, actor: RMATeacherActor):
        super().__init__()
        for name in ("obs_normalizer", "adaptation", "mlp", "privileged_normalizer"):
            setattr(self, name, copy.deepcopy(getattr(actor, name)))
        self.deterministic = (actor.distribution.as_deterministic_output_module()
                              if actor.distribution is not None else nn.Identity())
        self.velocity_head = bool(getattr(actor, "velocity_head", False))

    def forward(self, policy: torch.Tensor, history: torch.Tensor):
        z = self.adaptation(history)[1]
        return self.deterministic(self.mlp(torch.cat((self.obs_normalizer(policy), z), -1)))

    @torch.jit.export
    def explicit(self, history: torch.Tensor):
        """The 19 explicit privileged quantities as estimated (privileged units): with the velocity head, cart vx, vy,
        wz and robot vx, vy (indices 0, 1, 2, 4, 5); zeros elsewhere / without it."""
        out = torch.zeros(history.shape[0], 19, device=history.device)
        if self.velocity_head:
            v = self.adaptation(history)[0]
            idx = list(VELOCITY_IDX)
            mean = self.privileged_normalizer._mean.reshape(-1)[idx]
            std = self.privileged_normalizer._std.reshape(-1)[idx]
            out[:, idx] = v * std + mean
        return out


def is_rma_run(checkpoint) -> bool:
    from pathlib import Path
    import yaml
    path = Path(checkpoint)
    agent = (path.parent if path.suffix == ".pt" else path) / "params" / "agent.yaml"
    if not agent.is_file():
        return False
    cfg = yaml.load(agent.read_text(encoding="utf-8"), Loader=yaml.FullLoader) or {}
    return "RMATeacherActor" in str((cfg.get("actor") or {}).get("class_name", ""))


def load_rma(checkpoint, deploy: bool, device="cpu"):
    """RMA run checkpoint -> deterministic teacher (policy, privileged) or deployable student (policy, history)."""
    from pathlib import Path
    import yaml
    from tensordict import TensorDict
    checkpoint = Path(checkpoint)
    acfg = yaml.load((checkpoint.parent / "params/agent.yaml").read_text(encoding="utf-8"),
                     Loader=yaml.FullLoader)["actor"]
    probe = TensorDict({"policy": torch.zeros(2, POLICY_DIM), "privileged": torch.zeros(2, PRIVILEGED_DIM),
                        "history": torch.zeros(2, HISTORY_DIM)}, batch_size=[2])
    kw = {k: v for k, v in acfg.items() if k not in ("class_name", "deploy")}
    actor = RMATeacherActor(probe, {"actor": ["policy", "privileged", "history"]}, "actor", ACTION_DIM, **kw)
    loaded = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    actor.load_state_dict(loaded["actor_state_dict"])
    actor.use_estimate = bool(deploy)
    export = actor.as_jit().to(device).eval()
    export.iteration = loaded.get("iter")
    return export


class V8ROAPPO(PPO):
    """PPO + ROA (see the module docstring), and the two deployable-student stages on the same model:

    * teacher: ``adapt_interval`` = H; ``adapt_every_update`` also fits phi on every privileged rollout (2026-10-12: with
      ROA's schedule phi lagged the encoder, held-out latent R^2 0.61 vs 0.90 when fitted to convergence);
      ``init_checkpoint`` + ``start_update`` continue a run (lambda / H schedules from its global iteration).
    * ``adapt_only``: RMA phase 2 -- every rollout acts on z_phi, only phi learns (base policy, encoder, critic fixed).
    * ``student_rl``: PPO on the deployable network (z from the frozen phi), privileged critic.
    """

    def __init__(self, actor, critic, storage, *, roa_lambda_max: float = 1.0, roa_start: int = 5000,
                 roa_ramp: int = 5000, adapt_interval: int = 20, adapt_lr: float = 5.0e-4, adapt_only: bool = False,
                 init_checkpoint: str = "", adapt_std: float = 0.05, adapt_every_update: bool = False,
                 start_update: int = 0, student_rl: bool = False, student_std: float = 0.2,
                 velocity_loss_coef: float = 1.0, **kwargs):
        if not isinstance(actor, RMATeacherActor):
            raise TypeError("V8 ROA needs the V8 RMATeacherActor")
        super().__init__(actor, critic, storage, **kwargs)
        if init_checkpoint:                              # continuation / phase 2 / student RL: a trained ROA run
            loaded = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
            self.actor.load_state_dict(loaded["actor_state_dict"])
            self.critic.load_state_dict(loaded["critic_state_dict"])
            print(f"[pushcart-v8] ROA: weights from {init_checkpoint} (iter {loaded.get('iter')})", flush=True)
        if adapt_only and adapt_std > 0 and hasattr(self.actor.distribution, "std_param"):
            with torch.no_grad():                        # phase 2 rollouts: near-deterministic base policy
                self.actor.distribution.std_param.fill_(float(adapt_std))
        self.roa_lambda_max, self.roa_start, self.roa_ramp = float(roa_lambda_max), int(roa_start), int(roa_ramp)
        self.adapt_interval, self.adapt_only = max(1, int(adapt_interval)), bool(adapt_only)
        self.adapt_every_update = bool(adapt_every_update)
        self.velocity_loss_coef = float(velocity_loss_coef)
        adapt = list(self.actor.adaptation.parameters())
        ids = {id(p) for p in adapt}
        rest = [p for p in list(self.actor.parameters()) + list(self.critic.parameters()) if id(p) not in ids]
        self.optimizer = torch.optim.Adam(rest, lr=self.learning_rate)     # PPO: pi, mu, critic (adaptive KL)
        self.adapt_opt = torch.optim.Adam(adapt, lr=float(adapt_lr))        # phi: its own fixed rate
        self.n_updates = int(start_update)
        self.student_rl = bool(student_rl)
        if self.student_rl:
            if self.adapt_only:
                raise ValueError("student_rl and adapt_only are exclusive")
            frozen = (list(self.actor.adaptation.parameters()) + list(self.actor.encoder.parameters())
                      + list(self.actor.privileged_normalizer.parameters()))
            ids = {id(q) for q in frozen}
            for q in frozen:
                q.requires_grad_(False)
            self.actor.frozen_normalizers = True
            if student_std > 0 and hasattr(self.actor.distribution, "std_param"):
                with torch.no_grad():
                    self.actor.distribution.std_param.fill_(float(student_std))
            train = [q for q in list(self.actor.parameters()) + list(self.critic.parameters())
                     if id(q) not in ids and q.requires_grad]
            self.optimizer = torch.optim.Adam(train, lr=self.learning_rate)
            self.roa_lambda_max = 0.0
        self.actor.use_estimate = self._estimate_rollout(self.n_updates)

    def _estimate_rollout(self, k: int) -> bool:
        if getattr(self, "student_rl", False):
            return True
        return self.adapt_only or (k % self.adapt_interval == self.adapt_interval - 1)

    def roa_lambda(self) -> float:
        if self.adapt_only or self.roa_lambda_max <= 0:
            return 0.0
        return self.roa_lambda_max * min(max((self.n_updates - self.roa_start) / max(self.roa_ramp, 1), 0.0), 1.0)

    def update(self):
        if self.student_rl:
            out = self._update_ppo()                        # PPO on the deployable network (z from the frozen phi)
        elif self.actor.use_estimate:
            out = self._update_adaptation()
        else:
            out = self._update_ppo()
            if self.adapt_every_update:
                out.update(self._update_adaptation())
        self.storage.clear()
        self.n_updates += 1
        self.actor.use_estimate = self._estimate_rollout(self.n_updates)
        out["roa_lambda"] = self.roa_lambda()
        return out

    def _update_adaptation(self):
        """||sg(z_mu) - z_phi||_2 on the stored rollout (ROA's H-th iteration / RMA phase 2 / every update)."""
        total, gap, vel, n = 0., 0., 0., 0
        for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            obs = batch.observations
            with torch.no_grad():
                z_mu = self.actor.z_privileged(obs)
            v_phi, z_phi = self.actor.adaptation(obs["history"])
            loss = (z_mu - z_phi).norm(p=2, dim=1).mean()
            if self.actor.velocity_head:
                with torch.no_grad():
                    v_true = self.actor.velocity_target(obs)
                v_loss = (v_phi - v_true).pow(2).mean()
                loss = loss + self.velocity_loss_coef * v_loss
                vel += v_loss.item()
            self.adapt_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.adaptation.parameters(), self.max_grad_norm)
            self.adapt_opt.step()
            total += loss.item()
            gap += ((z_mu - z_phi.detach()).pow(2).mean() / z_mu.pow(2).mean().clamp_min(1e-8)).item()
            n += 1
        out = {"adaptation": total / max(n, 1), "latent_gap_rel": gap / max(n, 1)}
        if self.actor.velocity_head:
            out["velocity_head"] = vel / max(n, 1)
        return out

    def _update_ppo(self):
        """RSL-RL 5 PPO.update (no RND / symmetry) + lambda * ||z_mu - sg(z_phi)||_2 on the encoder."""
        lam = self.roa_lambda()
        sums = {"value": 0., "surrogate": 0., "entropy": 0., "roa_reg": 0.}
        n = 0
        for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            obs = batch.observations
            self.actor(obs, stochastic_output=True)
            log_prob = self.actor.get_output_log_prob(batch.actions)
            dist_params = tuple(self.actor.output_distribution_params)
            entropy = self.actor.output_entropy
            values = self.critic(obs)
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.mean(self.actor.get_kl_divergence(batch.old_distribution_params, dist_params))
                    if kl > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif 0.0 < kl < self.desired_kl / 2.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for g in self.optimizer.param_groups:
                        g["lr"] = self.learning_rate
            ratio = torch.exp(log_prob - torch.squeeze(batch.old_actions_log_prob))
            adv = torch.squeeze(batch.advantages)
            surrogate = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)).mean()
            if self.use_clipped_value_loss:
                vc = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max((values - batch.returns).pow(2), (vc - batch.returns).pow(2)).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()
            loss = surrogate + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            reg = torch.zeros((), device=loss.device)
            if lam > 0:
                with torch.no_grad():
                    z_phi = self.actor.z_estimated(obs)
                reg = (self.actor.z_privileged(obs) - z_phi).norm(p=2, dim=1).mean()
                loss = loss + lam * reg
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([p for g in self.optimizer.param_groups for p in g["params"]], self.max_grad_norm)
            self.optimizer.step()
            sums["value"] += value_loss.item()
            sums["surrogate"] += surrogate.item()
            sums["entropy"] += entropy.mean().item()
            sums["roa_reg"] += reg.item()
            n += 1
        return {k: v / max(n, 1) for k, v in sums.items()}
