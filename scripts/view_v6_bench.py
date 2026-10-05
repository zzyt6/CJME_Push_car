"""V6/V3 path viewer on the fixed legacy four-sphere backend; V6 is the default."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/"HomieRL")]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--checkpoint",default="latest")
    p.add_argument("--run",help="V6 or V3 training run directory/name; newest checkpoint in that run")
    p.add_argument("--port",type=int,default=8080)
    p.add_argument("--payload",type=float,default=5.)
    p.add_argument("--v_nom",type=float,default=.5)
    p.add_argument("--level",type=int,default=17)
    p.add_argument("--path",default="composite_course")
    a=p.parse_args(argv)
    import torch
    from pushcart_bench.view import checkpoint_policy_spec
    from pushcart_bench.config_io import load_config
    from pushcart_bench.experiments import exp_path
    torch.set_num_threads(1)
    if a.run and a.checkpoint != "latest":
        p.error("Use --run or an explicit --checkpoint, not both")
    checkpoint=resolve_checkpoint(a.checkpoint, a.run)
    cfg=load_config();cfg["trace_stride"]=1
    if a.path not in cfg["paths"]:
        p.error(f"Unknown path: {a.path}")
    spec=checkpoint_policy_spec(checkpoint)
    args=argparse.Namespace(command="path",policies=[spec],level=a.level,
        homie_checkpoint=None,scene_version="original",config=None,out_dir=None,view=True,
        port=a.port,speed=1.,caster_deg=0.,trace_stride=1,paths=None,mirror=False,seeds=[0],
        payload=a.payload,v_nom=a.v_nom,cruise=True,mode=None,gain_sweep=False,initial_path=a.path,
        checkpoint_selection=True)
    print(f"V6/V3 viewer, fixed legacy four-sphere plant; {spec}; http://localhost:{a.port}",flush=True)
    exp_path(args,cfg)


def resolve_checkpoint(checkpoint="latest", run=None):
    """Keep latest=V6 while explicit paths and training-run names may select V3."""
    if run:
        from homie_pushcart.mujoco_viewer import find_viewer_checkpoint
        return find_viewer_checkpoint(run=Path(run))
    from pushcart_bench.policies_v6 import resolve_v6_checkpoint
    return resolve_v6_checkpoint(checkpoint).resolve()


if __name__=="__main__":
    main()
