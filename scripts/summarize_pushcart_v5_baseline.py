"""Re-score saved paired trials under the current declared task protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "HomieRL"))
from homie_pushcart_v5.baseline_acceptance import PROTOCOL, summarize_trial


def summarize(directory):
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    trace_path = directory / "traces.npz"
    with np.load(trace_path) as traces:
        cases = []
        for i, case in enumerate(report["cases"]):
            uninterrupted = ~traces["failed"][:, i]
            one = {key: traces[key][uninterrupted, i] for key in
                   ("time_s", "actual", "reference", "cart_yaw", "cart_com")}
            row = {"task": case["task"], "payload_kg": case["payload_kg"],
                   "failure_time_s": case["failure_time_s"],
                   **summarize_trial(one, case["task"], bool((~uninterrupted).any()))}
            wrench = traces["physics_hand_wrench"][:, i]
            time = (np.arange(len(wrench)) + 1) * report["physics_dt"]
            before_reset = np.ones(len(time), dtype=bool)
            if case["failure_time_s"] is not None:
                before_reset &= time <= case["failure_time_s"] + 1e-7
            finite = np.isfinite(wrench).all(axis=(1, 2))
            valid = traces["physics_valid"][:, i].astype(bool) & finite
            row["hand_wrench_valid_fraction"] = float(valid[before_reset].mean()) if before_reset.any() else 0.
            selected = before_reset & valid
            if selected.any():
                force = np.linalg.norm(wrench[selected, :, :3], axis=-1)
                moment = np.linalg.norm(wrench[selected, :, 3:], axis=-1)
                row.update(per_hand_force_norm_p95_n=float(np.percentile(force, 95)),
                           per_hand_force_norm_peak_n=float(force.max()),
                           per_hand_moment_norm_peak_nm=float(moment.max()))
            cases.append(row)
        first = {key: traces[key][0].copy() for key in
                 ("actual", "cart_com", "cart_mass", "cart_yaw", "pose_base")}
    result = {"source": str(directory), "source_trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
              "source_report_sha256": hashlib.sha256((directory / "report.json").read_bytes()).hexdigest(),
              "policy": report["policy"], "protocol": PROTOCOL, "cases": cases,
              "passed_cases": sum(x["pass"] for x in cases), "num_cases": len(cases),
              "terminated_cases": sum(x.get("terminated", x["failure_time_s"] is not None) for x in cases),
              "max_per_hand_force_n": max((x.get("per_hand_force_norm_peak_n", 0.) for x in cases), default=0.),
              "max_per_hand_moment_nm": max((x.get("per_hand_moment_norm_peak_nm", 0.) for x in cases), default=0.)}
    return result, report, first


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zero", type=Path)
    parser.add_argument("trained", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    zero, zr, zfirst = summarize(args.zero)
    trained, tr, tfirst = summarize(args.trained)
    keys = ("seed", "level", "noisy", "dt", "physics_dt", "checkpoint_sha256", "task_config", "observer_geometry")
    checks = {key: zr[key] == tr[key] for key in keys}
    checks["policy_modes"] = zr["policy"] == "zero_high_level_action" and tr["policy"] == "loaded_actor"
    checks["disturbances_disabled"] = not zr["exogenous_disturbances"] and not tr["exogenous_disturbances"]
    checks["case_order"] = [(x["task"], x["payload_kg"]) for x in zr["cases"]] == [
        (x["task"], x["payload_kg"]) for x in tr["cases"]]
    if not all(checks.values()):
        raise ValueError(f"Paired configuration mismatch: {checks}")
    result = {"protocol_revision": PROTOCOL["revision"], "method": "recomputed saved traces; raw reports preserved",
              "pair_checks": checks, "checkpoint": zr["checkpoint"], "checkpoint_sha256": zr["checkpoint_sha256"],
              "initial_first_step_max_abs_difference": {key: float(np.max(np.abs(zfirst[key]-tfirst[key]))) for key in zfirst},
              "load_metric": "200 Hz valid samples through first terminal interval; per-hand moment about each palm; p95 pools individual hands and time, never sums both hands",
              "limitations": ["Single seed, nominal clean plant; no robustness or recovery claim.",
                              "First-step equality is a recorded diagnostic, not an independent proof of identical full simulator state.",
                              "Task pass does not certify acceptable hand load; force cost is reported separately."],
              "modes": {"zero": zero, "trained": trained}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({mode: {k: result["modes"][mode][k] for k in
                     ("passed_cases", "num_cases", "terminated_cases", "max_per_hand_force_n", "max_per_hand_moment_nm")}
                      for mode in ("zero", "trained")}, indent=2))


if __name__ == "__main__":
    main()
