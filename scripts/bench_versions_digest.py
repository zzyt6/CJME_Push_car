"""Print a compact text digest of scripts/bench_versions.py results (per-version metric tables)."""
import json
import math
import sys
from pathlib import Path

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "logs/bench/versions_20k"
ORDER = ["passthrough", "V3", "V3.1", "V3.2"]


def f(v, d=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    if isinstance(v, bool):
        return str(v)
    return f"{v:.{d}f}" if isinstance(v, (int, float)) else str(v)


s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
labels = [label for label in ORDER if label in s["labels"]]
print("success (paths ok / total):", {l: s["success"][l] for l in labels}, "| step failures:",
      {l: s["step_failures"][l] for l in labels})
print("\n== paths (wide S + mirror mean), columns = 0 / 12.5 / 25 kg")
for key, per in s["paths"].items():
    print(f"{key}")
    for l in labels:
        print(f"   {l:12s}", "  ".join(f"{f(v):>9s}" for v in per[l]))
print("\n== steps, columns = 0 / 12.5 / 25 kg")
for key, per in s["steps"].items():
    print(f"{key}")
    for l in labels:
        print(f"   {l:12s}", "  ".join(f"{f(v):>9s}" for v in per[l]))
print("\n== pulses (12.5 kg)")
for key, per in s["pulses"].items():
    print(key)
    for l in labels:
        print(f"   {l:12s}", " ".join(f"{k}={f(v)}" for k, v in per[l].items()))
print("\n== estimator (natural runs in MuJoCo)")
for l in labels:
    e = s["estimator"].get(l) or {}
    if not e:
        continue
    keep = {k: v for k, v in e.items() if (k.endswith("_r2") or k.endswith("_gain") or k.endswith("_rmse") or k.endswith("_true_std"))}
    print(l, " ".join(f"{k}={f(v)}" for k, v in keep.items()))
    print("   lag", e.get("est_handle_lag"))
