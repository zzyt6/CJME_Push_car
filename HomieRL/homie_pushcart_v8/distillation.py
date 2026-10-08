"""V8 phase 2: DAgger distillation (RSL-RL Distillation: the student drives, the teacher labels) with two context terms.

loss = MSE(clip(student action), clip(teacher action)) + w_explicit * MSE(student explicit, true explicit)
       + w_latent * MSE(student latent, teacher latent)

The behaviour term compares the actions the environment applies (high level clipped to +-hl_clip, leg residual to
+-leg_clip): the teacher's leg-residual outputs sit far beyond their clip (about +-10 vs +-0.2 at iteration 4800), so an
unclipped MSE would be dominated by magnitudes that never reach the robot.

Before the first update the student copies the teacher's policy body (and freezes it unless configured otherwise),
so it starts from the teacher's controller and only has to learn to estimate the context.
"""
from __future__ import annotations

import torch
from torch import nn
from rsl_rl.algorithms import Distillation

from .student_policy import StudentActor
from .teacher_policy import TeacherActor


class V8Distillation(Distillation):
    def __init__(self, student, teacher, storage, *, explicit_weight=1.0, latent_weight=1.0, dagger_std=0.05,
                 hl_clip=1.0, leg_clip=0.2, **kwargs):
        if not isinstance(student, StudentActor) or not isinstance(teacher, TeacherActor):
            raise TypeError("V8 distillation needs the V8 StudentActor and TeacherActor")
        super().__init__(student, teacher, storage, **kwargs)
        self.explicit_weight, self.latent_weight = float(explicit_weight), float(latent_weight)
        self.dagger_std = float(dagger_std)
        from .task_config import HL_ACTIONS, LEG_ACTIONS
        self.action_bound = torch.tensor([float(hl_clip)] * HL_ACTIONS + [float(leg_clip)] * LEG_ACTIONS)
        # the student optimizer only steps parameters that require gradients (frozen body excluded after the copy)

    def _ensure_body(self):
        if not self.student.body_loaded:
            if not self.teacher_loaded:
                raise ValueError("V8 distillation: load the teacher checkpoint before acting")
            self.student.load_body_from(self.teacher)
            dist = self.student.distribution
            if dist is not None and hasattr(dist, "std_param"):      # small exploration while the student drives
                with torch.no_grad():
                    dist.std_param.fill_(self.dagger_std)
                dist.std_param.requires_grad_(False)

    def act(self, obs):
        self._ensure_body()
        return super().act(obs)

    def update(self):
        self._ensure_body()
        self.num_updates = getattr(self, "num_updates", 0) + 1
        totals = {"behavior": 0., "explicit": 0., "latent": 0.}
        count = 0
        for epoch in range(self.num_learning_epochs):
            for batch in self.storage.generator():
                obs = batch.observations
                actions = self.student(obs)
                bound = self.action_bound.to(actions.device)
                behavior = self.loss_fn(torch.maximum(torch.minimum(actions, bound), -bound),
                                        torch.maximum(torch.minimum(batch.privileged_actions, bound), -bound))
                with torch.no_grad():
                    explicit_t, latent_t = self.teacher.context(obs)
                explicit_s, latent_s = self.student.context(obs)
                explicit = nn.functional.mse_loss(explicit_s, explicit_t)
                latent = nn.functional.mse_loss(latent_s, latent_t)
                loss = behavior + self.explicit_weight * explicit + self.latent_weight * latent
                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_([p for p in self.student.parameters() if p.requires_grad],
                                             self.max_grad_norm)
                self.optimizer.step()
                totals["behavior"] += behavior.item()
                totals["explicit"] += explicit.item()
                totals["latent"] += latent.item()
                count += 1
        out = {k: v / max(count, 1) for k, v in totals.items()}
        out.update(self.explicit_metrics())            # before the rollout is cleared
        self.storage.clear()
        return out

    @torch.no_grad()
    def explicit_metrics(self, max_rows=16384):
        """Per-quantity student estimation error in physical units (cart v, mass, robot v, hand wrenches)."""
        obs = self.storage.observations.flatten(0, 1) if hasattr(self.storage, "observations") else None
        if obs is None:
            return {}
        rows = torch.randperm(obs.batch_size[0], device=self.device)[:max_rows]
        sel = obs[rows]
        est, _ = self.student.context(sel)
        true = sel["privileged"][:, :est.shape[1]]
        err = (est - true).abs().mean(0)
        names = (("cart_vx", 0.5), ("cart_vy", 0.5), ("cart_wz", 2.0), ("cart_mass_norm", 1.0),
                 ("robot_vx", 0.5), ("robot_vy", 0.5), ("robot_vz", 0.5))
        out = {f"student_mae_{n}": float(err[i] * s) for i, (n, s) in enumerate(names)}
        out["student_mae_hand_wrench_scaled"] = float(err[7:19].mean())
        return out
