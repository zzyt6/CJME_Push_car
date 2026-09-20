"""The three experiments: path following to a goal, velocity step response, robustness sweep."""
from __future__ import annotations

from datetime import datetime
import itertools
import json
from pathlib import Path
import time

from . import PROJECT_ROOT
from .follower import FollowerConfig
from .paths import build_path
from .policies import make_policy
from .runner import Condition, Disturbance, run_path, run_step


def make_out_dir(out: str | Path | None, kind: str) -> Path:
    path = Path(out) if out else PROJECT_ROOT / "logs" / "bench" / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{kind}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class RunStore:
    """Appends every finished run to <out>/runs.json (rewritten each time, so a crash keeps what was done)."""

    def __init__(self, out_dir: Path, meta: dict):
        self.path = out_dir / "runs.json"
        self.meta = meta
        self.runs = []
        if self.path.is_file():
            old = json.loads(self.path.read_text(encoding="utf-8"))
            self.runs = old.get("runs", [])

    def add(self, run: dict):
        self.runs.append(run)
        self.path.write_text(json.dumps({"meta": self.meta, "runs": self.runs}, indent=1, ensure_ascii=False), encoding="utf-8")


def _policies(args, cfg):
    level = int(getattr(args, "level", None) or cfg["defaults"].get("level", 17))
    homie = getattr(args, "homie_checkpoint", None)
    out = []
    for spec in args.policies:
        t0 = time.perf_counter()
        policy = make_policy(spec, level=level, homie_checkpoint=homie)
        print(f"[bench] policy {policy.name}: {policy.describe()} ({time.perf_counter() - t0:.1f} s)", flush=True)
        out.append(policy)
    return out


def _viewer(args):
    if not getattr(args, "view", False):
        return None
    from .view import BenchViewer
    return BenchViewer(port=args.port, speed=args.speed)


def _meta(args, cfg, kind, follower_cfg=None):
    return {"kind": kind, "created": datetime.now().isoformat(timespec="seconds"), "policies": list(args.policies),
            "follower": follower_cfg.to_dict() if follower_cfg else None, "defaults": cfg["defaults"],
            "overload_thresholds": cfg.get("overload_thresholds"),
            "fairness": "same follower gains / limits, same MuJoCo model and HOMIE checkpoint, same initial pose and seeds, "
                        "two-stage policy uses the deterministic mean action; cart policies are tracked at the cart base, "
                        "robot_only at the pelvis (same path geometry, each aligned to its own start pose)"}


def _print_path_line(label, policy, m):
    ttg = f"{m['time_to_goal']:.1f} s" if m.get("time_to_goal") is not None else ("FAIL " + str(m.get("failure")) if m.get("failure") else "timeout")
    print(f"[bench] {label:34s} {policy.name:28s} {ttg:>12s} | final {m.get('final_pos_err', float('nan')):.3f} m | "
          f"ct rms/max {m.get('cross_track_rms', float('nan')):.3f}/{m.get('cross_track_max', float('nan')):.3f} m | "
          f"cmd rmse vx/wz {m.get('cmd_vx_rmse', float('nan')):.3f}/{m.get('cmd_wz_rmse', float('nan')):.3f}"
          + (f" | grasp p95 {m['grasp_p95']:.0f} N" if "grasp_p95" in m else "")
          + (f" | clip cmd/arm {m['cmd_clip_frac']:.2f}/{m['arm_clip_frac']:.2f}" if "arm_clip_frac" in m else ""), flush=True)


def _path_variants(args, cfg):
    names = args.paths or list(cfg["paths"].keys())
    variants = []
    for name in names:
        if name not in cfg["paths"]:
            raise KeyError(f"unknown path {name!r}; available: {list(cfg['paths'])}")
        variants.append((name, build_path(name, cfg["paths"][name])))
        if getattr(args, "mirror", False) and name in cfg.get("mirror", []):
            variants.append((name + "_mirror", build_path(name, cfg["paths"][name], mirror=True)))
    return variants


# ---------------------------------------------------------------------------- experiment 1

def exp_path(args, cfg) -> Path:
    follower_cfg = FollowerConfig.from_dict({**cfg["follower"], **({"mode": args.mode} if getattr(args, "mode", None) else {})})
    d = cfg["defaults"]
    seeds = args.seeds if args.seeds is not None else d.get("seeds", [0])
    payload = d.get("payload_kg", 5.0) if args.payload is None else args.payload
    caster = d.get("caster_angle_deg", 0.0) if args.caster_deg is None else args.caster_deg
    out_dir = make_out_dir(args.out_dir, "path")
    store = RunStore(out_dir, _meta(args, cfg, "path", follower_cfg))
    viewer = _viewer(args)
    policies = _policies(args, cfg)
    variants = _path_variants(args, cfg)
    gain_grid = [{}]
    if getattr(args, "gain_sweep", False) and cfg.get("gain_sweep"):
        keys = list(cfg["gain_sweep"].keys())
        gain_grid = [dict(zip(keys, vals)) for vals in itertools.product(*[cfg["gain_sweep"][k] for k in keys])]
        print(f"[bench] gain sweep: {len(gain_grid)} combinations of {keys}", flush=True)
    if getattr(args, "v_nom", None) is not None:
        follower_cfg.v_nom = float(args.v_nom)
    if getattr(args, "cruise", False):
        follower_cfg.cruise = True
        follower_cfg.k_curv = 0.0
        follower_cfg.v_min_frac = 1.0
    if viewer is not None:
        return _exp_path_live(args, cfg, policies, viewer, store, follower_cfg, out_dir, payload, caster, seeds)
    print(f"[bench] experiment 1 (path following): {len(policies)} policies x {len(variants)} paths x {len(seeds)} seeds x {len(gain_grid)} gains -> {out_dir}", flush=True)
    for gains in gain_grid:
        fcfg = FollowerConfig.from_dict({**follower_cfg.to_dict(), **gains}) if gains else follower_cfg
        for pname, path in variants:
            for policy in policies:
                for seed in seeds:
                    cond = Condition(payload_kg=float(payload), caster_angle_deg=float(caster), seed=int(seed))
                    label = f"{pname} seed{seed}" + (f" {gains}" if gains else "")
                    res = run_path(policy, path, fcfg, cond, cfg, viewer=viewer, label=label)
                    res["gains"] = gains
                    store.add(res)
                    _print_path_line(label, policy, res["metrics"])
    from .report import write_report
    write_report(out_dir)
    return out_dir


def _exp_path_live(args, cfg, policies, viewer, store, follower_cfg, out_dir, payload, caster, seeds):
    """Interactive path following: viser sliders choose path, policy, payload, commanded vx; no auto slowdown by default."""
    names = args.paths or list(cfg["paths"].keys())
    seed = int(seeds[0]) if seeds else 0
    viewer.setup_live(path_names=names, policy_names=[p.name for p in policies], payload=float(payload),
                      v_nom=follower_cfg.v_nom, cruise=True, follower_defaults=cfg["follower"])
    if "s_large" in names:
        viewer.dd_path.value = "s_large"
    print(f"[bench] live view -> {out_dir}", flush=True)
    print("[bench] viser: 待命中。调好镜头、开录后点「开始」。改路径/策略/载荷会重新摆位。Ctrl+C 退出。", flush=True)
    try:
        while True:
            if not viewer.want_rerun:
                _wait_for_start(viewer, policies, cfg, caster, seed)
            viewer.want_rerun = False
            pname = viewer.dd_path.value
            if pname not in cfg["paths"]:
                print(f"[bench] unknown path {pname!r}", flush=True)
                continue
            policy = next((p for p in policies if p.name == viewer.dd_pol.value), policies[0])
            path = build_path(pname, cfg["paths"][pname])
            fcfg = FollowerConfig.from_dict(follower_cfg.to_dict())
            viewer.sync_follower(fcfg, path_length=path.length)
            cond = Condition(payload_kg=float(viewer.sl_payload.value), caster_angle_deg=float(caster), seed=seed)
            label = f"{pname} {cond.payload_kg:g}kg vx={fcfg.v_nom:.2f}"
            print(f"[bench] start {label}  {policy.name}  cruise={fcfg.cruise}", flush=True)
            res = run_path(policy, path, fcfg, cond, cfg, viewer=viewer, label=label)
            if viewer.want_rerun:
                continue
            store.add(res)
            _print_path_line(label, policy, res["metrics"])
    except KeyboardInterrupt:
        print("[bench] live view stopped", flush=True)
    from .report import write_report
    write_report(out_dir)
    return out_dir


def _ready_key(viewer):
    return (viewer.dd_path.value, viewer.dd_pol.value, round(float(viewer.sl_payload.value), 1))


def _show_ready(viewer, policies, cfg, caster, seed):
    """Reset to the start pose, draw the path, and freeze until the user clicks 开始."""
    pname = viewer.dd_path.value
    if pname not in cfg["paths"]:
        viewer.status.content = f"- 未知路径 `{pname}`"
        return
    policy = next((p for p in policies if p.name == viewer.dd_pol.value), policies[0])
    cond = Condition(payload_kg=float(viewer.sl_payload.value), caster_angle_deg=float(caster), seed=int(seed))
    cond.apply(policy)
    x0, y0, yaw0 = policy.pose()
    path = build_path(pname, cfg["paths"][pname]).transformed((x0, y0), yaw0)
    viewer.begin_run(path, policy, "待命")
    viewer.status.content = (
        f"- **待命** · 路径 `{pname}` · 策略 **{policy.name}** · 载荷 {cond.payload_kg:.1f} kg · "
        f"指定 vx {viewer.sl_vnom.value:.2f} m/s\n"
        "- 调好镜头、开录后点绿色 **开始**"
    )
    if viewer.renderer is not None:
        viewer.renderer.update(policy.env.data)


def _wait_for_start(viewer, policies, cfg, caster, seed):
    last = None
    while not viewer.want_rerun:
        key = _ready_key(viewer)
        if key != last:
            last = key
            _show_ready(viewer, policies, cfg, caster, seed)
        time.sleep(0.05)


# ---------------------------------------------------------------------------- experiment 2

def exp_step(args, cfg) -> Path:
    d = cfg["defaults"]
    steps = [s for s in cfg["steps"] if not args.steps or s["name"] in args.steps]
    payloads = args.payloads if args.payloads is not None else cfg.get("step_payloads", [0.0])
    caster = d.get("caster_angle_deg", 0.0) if args.caster_deg is None else args.caster_deg
    out_dir = make_out_dir(args.out_dir, "step")
    store = RunStore(out_dir, _meta(args, cfg, "step"))
    viewer = _viewer(args)
    policies = _policies(args, cfg)
    print(f"[bench] experiment 2 (step response): {len(policies)} policies x {len(steps)} steps x {len(payloads)} payloads -> {out_dir}", flush=True)
    for step in steps:
        for payload in payloads:
            for policy in policies:
                cond = Condition(payload_kg=float(payload), caster_angle_deg=float(caster), seed=0)
                label = f"{step['name']} {payload:g}kg"
                res = run_step(policy, step, cond, cfg, viewer=viewer, label=label)
                store.add(res)
                m = res["metrics"]
                extra = f" | stop {m['stopping_distance']:.2f} m" if m.get("stopping_distance") is not None else ""
                extra += f" | drift {m['drift_distance']:.2f} m" if m.get("drift_distance") is not None else ""
                fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
                fmt3 = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
                print(f"[bench] {label:24s} {policy.name:28s} rise {fmt(m.get('rise_time'))} s | settle {fmt(m.get('settling_time'))} s | "
                      f"overshoot {fmt(m.get('overshoot'))} | ss err {fmt(m.get('ss_error'))} | ripple {fmt3(m.get('ripple_rms'))} | "
                      f"slow osc {fmt3(m.get('ss_std'))}{extra}"
                      + (f" | FAIL {m['failure']}" if m.get("failure") else ""), flush=True)
    from .report import write_report
    write_report(out_dir)
    return out_dir


# ---------------------------------------------------------------------------- experiment 3

def exp_robust(args, cfg) -> Path:
    follower_cfg = FollowerConfig.from_dict(cfg["follower"])
    d, rb = cfg["defaults"], cfg["robust"]
    path_name = args.path or rb.get("path", "s_curve")
    path = build_path(path_name, cfg["paths"][path_name])
    seeds = args.seeds if args.seeds is not None else rb.get("seeds", d.get("seeds", [0]))
    factors = {k: v for k, v in rb["factors"].items() if not args.factors or k in args.factors}
    out_dir = make_out_dir(args.out_dir, "robust")
    store = RunStore(out_dir, _meta(args, cfg, "robust", follower_cfg))
    viewer = _viewer(args)
    policies = _policies(args, cfg)
    n_runs = sum(len(v) for v in factors.values()) * len(policies) * len(seeds)
    print(f"[bench] experiment 3 (robustness on {path_name}): factors {list(factors)} -> {n_runs} runs -> {out_dir}", flush=True)
    base = dict(payload_kg=d.get("payload_kg", 5.0), caster_angle_deg=d.get("caster_angle_deg", 0.0))
    for factor, values in factors.items():
        for value in values:
            for policy in policies:
                for seed in seeds:
                    cond_kw = dict(base, seed=int(seed))
                    dist = Disturbance(kick_interval_s=rb.get("kick_interval_s", 5.0), stall_prob_per_s=rb.get("stall_prob_per_s", 0.2),
                                       stall_duration_s=rb.get("stall_duration_s", 0.3), seed=int(seed))
                    if factor == "kick":
                        dist.kick_dv = float(value)
                    elif factor == "wheel_stall":
                        dist.stall_factor = float(value)
                    elif factor in ("payload_kg", "caster_angle_deg", "wheel_friction", "payload_com_offset_m"):
                        cond_kw[factor] = float(value)
                    else:
                        raise KeyError(f"unknown robustness factor {factor!r}")
                    cond = Condition(**cond_kw)
                    label = f"{factor}={value:g} seed{seed}"
                    res = run_path(policy, path, follower_cfg, cond, cfg, disturbance=dist, viewer=viewer, label=label)
                    res["factor"], res["factor_value"] = factor, float(value)
                    store.add(res)
                    _print_path_line(label, policy, res["metrics"])
    from .report import write_report
    write_report(out_dir)
    return out_dir
