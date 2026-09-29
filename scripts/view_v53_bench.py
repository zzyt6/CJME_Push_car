"""Launch V5.3 in the existing bench Viser interface, optionally with both scenes."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'HomieRL'), str(ROOT/'scripts')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='latest', help='V5.3 model file, or latest V5.3 run')
    p.add_argument('--scene', choices=('both','original','new'), default='both')
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--payload', type=float, default=15.)
    p.add_argument('--v_nom', type=float, default=.3)
    p.add_argument('--level', type=int, default=13)
    p.add_argument('--path', default='straight')
    a = p.parse_args()
    from pushcart_bench.policies_v53 import resolve_v53_checkpoint
    from pushcart_bench.config_io import load_config
    from pushcart_bench import experiments
    import torch
    torch.set_num_threads(1)
    ckpt = resolve_v53_checkpoint(a.checkpoint).resolve()
    scenes = ('original','new') if a.scene=='both' else (a.scene,)
    args = argparse.Namespace(command='path', policies=[f'v53_{s}:{ckpt}' for s in scenes],
        level=a.level, homie_checkpoint=None, scene_version='original', config=None,
        out_dir=None, view=True, port=a.port, speed=1., caster_deg=0., trace_stride=1,
        paths=None, mirror=False, seeds=[0], payload=a.payload, v_nom=a.v_nom,
        cruise=True, mode=None, gain_sweep=False, initial_path=a.path)
    cfg=load_config()
    cfg['trace_stride']=1
    if a.path not in cfg['paths']:p.error(f'Unknown path {a.path}; choose from {list(cfg["paths"])}')
    print(f'[V5.3 bench] checkpoint={ckpt}\n[V5.3 bench] scenes={scenes}; open http://localhost:{a.port}',flush=True)
    experiments.exp_path(args,cfg)


if __name__=='__main__':main()
