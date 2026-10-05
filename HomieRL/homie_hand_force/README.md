# HOMIE hand-force research task

Status on 2026-10-04: independently runnable copy of the current HOMIE baseline.
**External hand forces are not implemented or enabled yet.** The task was
created first so later force experiments cannot alter the original baseline.

- Task: `Homie-G1-HandForce-Direct-v0`
- Training: `train_homie_hand_force.ps1`
- Playback: `run_homie_hand_force.ps1 play ...`
- Training logs: `logs/rsl_rl/homie_g1_hand_force/`
- Source provenance: `baseline_manifest.json`

All original training settings, observations (456/79), actions (12), rewards,
upper-body pose curriculum and domain randomization are currently identical.
The task's environment/configuration/adapter files are physically copied; robot
source assets and the original `homie_rsl_rl` implementation are shared.

The original `homie_isaaclab` task and pinned `model_60000.pt` are unchanged.

See [the code and data research report](../../docs/homie_hand_force_research.md)
for the FALCON/UniFP distinction, paper/code differences, measured per-hand
force ranges, and the proposed implementation and ablation contracts.

Validated: 8 environments / 2 PPO updates through the new official training
entrypoint, and 2 environments / 100 playback steps with the existing baseline
checkpoint. These are execution checks, not a trained force-robust policy.
