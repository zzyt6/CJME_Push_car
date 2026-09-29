"""Verify the old Isaac cart reset against MuJoCo compatibility, incl. offsets."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'HomieRL')]


def run(args):
    import numpy as np
    import torch
    import warp as wp
    from homie_pushcart_v52.control_contract import nominal_configs
    from homie_pushcart_v52.task_env import PushCartEnv
    from eval_pushcart_v52_sim2sim import DEFAULT_CKPT
    from pushcart_bench.physx_inertia import cart_reset_inertia, usd_mass_properties
    from pushcart_bench.sim2sim_scene import payload_properties

    torch.manual_seed(73)
    cfg, task = nominal_configs(DEFAULT_CKPT, 'explicit_pd')
    low = Path(cfg.pushcart_saved_checkpoint)
    if not low.is_absolute():
        low = ROOT/low
    env = PushCartEnv(cfg, task, ROOT, args.device, True, level=17, checkpoint=low)
    env_ids = torch.tensor([0], device=args.device)
    props = usd_mass_properties()['bodies']['cart']
    rows = []
    for extent in (0., .1):
        env.task_cfg.randomization.payload_com_offset_m = extent
        for payload in (0., 5., 12.5, 25., 50.):
            for repeat in range(2):
                env.fixed_payload_kg = payload
                env._randomize_cart(env_ids)  # actual training code, no mock setter
                actual = wp.to_torch(env.cart.root_view.get_inertias())[0].cpu().numpy().reshape(-1, 3, 3).transpose(0, 2, 1)
                offset = env.payload_com[0].cpu().numpy()
                body_errors, expected_tensors = {}, {}
                for i, name in enumerate(env.cart.body_names):
                    prop = props[name]
                    intended = np.array(prop['inertia'])
                    if name == 'base_link':
                        _, _, intended = payload_properties(prop['mass'], np.array(prop['com']), intended, payload, offset)
                    expected = cart_reset_inertia(intended, name)
                    expected_tensors[name] = expected.tolist()
                    body_errors[name] = float(np.linalg.norm(expected-actual[i]))
                rows.append(dict(payload_kg=payload, offset_m=offset.tolist(), repeat=repeat,
                                 tensor_error_kg_m2=body_errors,
                                 names=env.cart.body_names, actual=actual.tolist(), expected=expected_tensors))
    maximum = max(max(r['tensor_error_kg_m2'].values()) for r in rows)
    report = dict(cases=rows, max_inertia_tensor_error_kg_m2=maximum,
                  passed=maximum < 5e-6,
                  note='Actual V5.2 _randomize_cart and raw PhysX tensor readback; no policy rollout or neural-score fitting.')
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('cart_reset_inertia', maximum, 'PASS' if report['passed'] else 'FAIL', flush=True)
    if not report['passed']:
        raise AssertionError('Cart inertia compatibility does not match runtime')


if __name__ == '__main__':
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    app = AppLauncher(args)
    try:
        run(args)
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        app.app.close()
