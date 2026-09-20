"""Push-cart control benchmark: policy-agnostic comparison experiments in MuJoCo (+ viser).

Independent of the RL training code in ``HomieRL/``: it only imports the MuJoCo environment / runner and the
checkpoint loaders of ``homie_pushcart``. A *policy under test* receives a velocity command in the frame of the
tracked body (cart base or pelvis) and advances the simulation by one control step; everything above that
line (path following, step tests, disturbances, metrics, reports) is shared by every policy.

Entry point: ``scripts/pushcart_bench.py`` (``run_pushcart_bench.ps1``).
"""
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BENCH_ROOT.parent
DEFAULT_CONFIG = BENCH_ROOT / "config" / "benchmark.yaml"
