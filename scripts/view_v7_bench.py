"""V7 path viewer on the MuJoCo backend: the V7 plant (square tube, elasto-plastic slide); V3/V6 selectable (legacy plant)."""
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
    p.add_argument("--payload",type=float,default=15.)
    p.add_argument("--v_nom",type=float,default=.5)
    p.add_argument("--level",type=int,default=17)
    p.add_argument("--path",default="composite_course")
    p.add_argument("--version",choices=("v7","v8"),default="v7",help="which family 'latest' resolves to")
    p.add_argument("--no_leg_residual",action="store_true",help="V8: zero the 12 leg-residual actions (ablation)")
    p.add_argument("--plant",choices=("auto","legacy","v7"),default="auto",
        help="auto: V7 -> V7 plant, V6/V3 -> legacy plant; legacy / v7 force one plant for every version (comparable)")
    a=p.parse_args(argv)
    import torch
    from pushcart_bench.view import checkpoint_policy_spec
    from pushcart_bench.config_io import load_config
    from pushcart_bench.experiments import exp_path
    torch.set_num_threads(1)
    if a.run and a.checkpoint != "latest":
        p.error("Use --run or an explicit --checkpoint, not both")
    checkpoint=resolve_checkpoint(a.checkpoint, a.run, a.version)
    cfg=load_config();cfg["trace_stride"]=1
    if a.path not in cfg["paths"]:
        p.error(f"Unknown path: {a.path}")
    spec=checkpoint_policy_spec(checkpoint)
    if spec.startswith("v8:"):
        if a.plant=="legacy":
            p.error("V8 runs on the V7 plant only")
        if a.no_leg_residual:
            spec="v8_legres0:"+spec[3:]
    elif a.plant!="auto":
        kind,_,ckpt=spec.partition(":")
        if kind not in ("v6","v7","passthrough"):
            p.error(f"--plant supports V6 / V7 checkpoints and passthrough, not {kind}")
        spec=f"{kind}@{a.plant}"+(f":{ckpt}" if ckpt else "")
    args=argparse.Namespace(command="path",policies=[spec],level=a.level,
        homie_checkpoint=None,scene_version="original",config=None,out_dir=None,view=True,
        port=a.port,speed=1.,caster_deg=0.,trace_stride=1,paths=None,mirror=False,seeds=[0],
        payload=a.payload,v_nom=a.v_nom,cruise=True,mode=None,gain_sweep=False,initial_path=a.path,
        checkpoint_selection=True)
    print(f"bench viewer (MuJoCo); {spec}; http://localhost:{a.port}",flush=True)
    exp_path(args,cfg)


def resolve_checkpoint(checkpoint="latest", run=None, version="v7"):
    """latest = newest V7 (or V8) run; explicit paths and training-run names may select V3 / V6 / V7 / V8."""
    if run:
        from homie_pushcart.mujoco_viewer import find_viewer_checkpoint
        return find_viewer_checkpoint(run=Path(run))
    if version == "v8":
        from pushcart_bench.policies_v8 import resolve_v8_checkpoint
        return resolve_v8_checkpoint(checkpoint).resolve()
    from pushcart_bench.policies_v7 import resolve_v7_checkpoint
    return resolve_v7_checkpoint(checkpoint).resolve()


if __name__=="__main__":
    main()
