"""V6 path viewer on the fixed legacy four-sphere backend, without scene choices."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/"HomieRL")]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--checkpoint",default="latest")
    p.add_argument("--port",type=int,default=8080)
    p.add_argument("--payload",type=float,default=5.)
    p.add_argument("--v_nom",type=float,default=.5)
    p.add_argument("--level",type=int,default=17)
    p.add_argument("--path",default="s_large")
    a=p.parse_args(argv)
    import torch
    from pushcart_bench.policies_v6 import resolve_v6_checkpoint
    from pushcart_bench.config_io import load_config
    from pushcart_bench.experiments import exp_path
    torch.set_num_threads(1)
    checkpoint=resolve_v6_checkpoint(a.checkpoint).resolve()
    cfg=load_config();cfg["trace_stride"]=1
    if a.path not in cfg["paths"]:
        p.error(f"Unknown path: {a.path}")
    args=argparse.Namespace(command="path",policies=[f"v6:{checkpoint}"],level=a.level,
        homie_checkpoint=None,scene_version="original",config=None,out_dir=None,view=True,
        port=a.port,speed=1.,caster_deg=0.,trace_stride=1,paths=None,mirror=False,seeds=[0],
        payload=a.payload,v_nom=a.v_nom,cruise=True,mode=None,gain_sweep=False,initial_path=a.path)
    print(f"V6 fixed legacy four-sphere plant; {checkpoint}; http://localhost:{a.port}",flush=True)
    exp_path(args,cfg)


if __name__=="__main__":
    main()
