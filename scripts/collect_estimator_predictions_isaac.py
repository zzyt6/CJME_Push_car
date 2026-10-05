"""Collect three V6-only Isaac trials with exact estimator inputs and targets."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "logs/Picture/force_velocity_predict/isaac")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--payload", type=float, default=15.)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    from pushcart_bench.policies_v6 import resolve_v6_checkpoint
    import torch
    checkpoint = resolve_v6_checkpoint(args.checkpoint or "latest").resolve()
    iteration = int(torch.load(checkpoint, map_location="cpu", weights_only=False)["iter"])
    run_id = uuid.uuid4().hex
    attempt = args.output / "runs" / f"{datetime.now():%Y%m%d_%H%M%S}_{run_id[:8]}"
    attempt.mkdir(parents=True, exist_ok=False)
    collector = ROOT / "scripts/collect_motion_ability_isaac.py"
    manifest = dict(engine="Isaac Sim / PhysX", run_id=run_id, checkpoint=str(checkpoint),
                    checkpoint_sha256=sha(checkpoint), iteration=iteration, payload_kg=args.payload,
                    seed=args.seed, sampling_hz=50, idle_s=0., plateau_s=8.,
                    source_collector_sha256=sha(collector), attempts=str(attempt.resolve()), runs=[])
    for axis in ("vx", "vy", "wz"):
        command = [sys.executable, "-u", str(collector), "--axis", axis, "--method", "CWC",
                   "--checkpoint", str(checkpoint), "--iteration", str(iteration),
                   "--payload", str(args.payload), "--seed", str(args.seed), "--device", args.device,
                   "--output", str(attempt), "--record_estimators"]
        print(f"[estimator-isaac] starting {axis}; log: {attempt / (axis + '.log')}", flush=True)
        with (attempt / f"{axis}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, CWC_MOTION_ABILITY_RUN=run_id),
                                    stdout=log, stderr=subprocess.STDOUT)
        report_path = attempt / "raw" / f"{axis}_CWC.json"
        if result.returncode or not report_path.is_file():
            raise RuntimeError(f"{axis} collection failed; see {attempt / (axis + '.log')}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["completed"] and report["estimator_recording"]
        assert report["run_id"] == run_id
        assert report["checkpoint_sha256"] == manifest["checkpoint_sha256"]
        assert report["npz_sha256"] == sha(report_path.with_suffix(".npz"))
        manifest["runs"].append(dict(axis=axis, samples=report["samples"], completed=True,
                                     source=str(report_path), npz_sha256=report["npz_sha256"]))
        (attempt / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[estimator-isaac] {axis} complete, {report['samples']} samples", flush=True)
    assert sha(checkpoint) == manifest["checkpoint_sha256"]
    destination = args.output / "raw"
    destination.mkdir(exist_ok=True, parents=True)
    for source in (attempt / "raw").iterdir():
        shutil.copy2(source, destination / source.name)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[estimator-isaac] verified raw data: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
