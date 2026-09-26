"""Run a benchmark while sampling nvidia-smi during the timed rollout."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--console", type=Path, required=True)
    args, benchmark_args = p.parse_known_args()
    if benchmark_args and benchmark_args[0] == "--":
        benchmark_args = benchmark_args[1:]
    args.samples.parent.mkdir(parents=True, exist_ok=True)
    query = [
        "nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw",
        "--format=csv,noheader,nounits", "--loop-ms=200",
    ]
    with args.samples.open("w", encoding="utf-8") as sample_file:
        sampler = subprocess.Popen(query, stdout=sample_file, stderr=subprocess.STDOUT,
                                   text=True)
        try:
            command = [sys.executable, "scripts/benchmark_pushcart_v51_performance.py"]
            command.extend(benchmark_args)
            with args.console.open("w", encoding="utf-8") as console_file:
                result = subprocess.run(command, stdout=console_file, stderr=subprocess.STDOUT,
                                        check=False)
        finally:
            sampler.terminate()
            try:
                sampler.wait(timeout=5)
            except subprocess.TimeoutExpired:
                sampler.kill()
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
