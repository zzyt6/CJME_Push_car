"""Export a trained V5.3 actor and both estimators; reject older observation schemas."""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'HomieRL'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    from homie_pushcart_v53.hl_policy import HighLevelPolicy
    from homie_pushcart_v53.task_config import task_cfg_from_run, num_actor_obs, num_critic_obs
    from homie_pushcart_v53.observation import frame_layout, CART_INPUT_FRAME_DIM
    cfg=task_cfg_from_run(args.checkpoint)
    policy=HighLevelPolicy(args.checkpoint,num_actor_obs(cfg),num_critic_obs(cfg),18)
    args.output.mkdir(parents=True,exist_ok=True)
    torch.jit.script(policy.actor.as_jit().eval()).save(str(args.output/'actor_and_estimators.jit'))
    manifest=dict(version=53, checkpoint=str(args.checkpoint.resolve()), observable_fields=frame_layout(),
        history=cfg.env.actor_history, estimator_history=policy.actor.estimator_history,
        cart_input_frame_dim=CART_INPUT_FRAME_DIM, heading_source='vision_relative_yaw',
        heading_convention='cart yaw minus upright base yaw, calibrated camera extrinsics',
        cart_velocity='base_link origin velocity; upright cart-heading axes; [vx,vy,world omega_z]',
        methods={'forward':'18 high-level actions','physical_estimates':'cart velocity (m/s,rad/s), hand wrench (N,Nm)'},
        deployment_builder='homie_pushcart_v53.deploy_observation.ObservableHistory',
        required_sensors='27 q/dq/drive-torque readings, IMU, frozen HIM, timestamped vision yaw, executed control history',
        note='Simulator labels are excluded. Torque channel and vision noise assumptions require hardware calibration.')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__': main()
