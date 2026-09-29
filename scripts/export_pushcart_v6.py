"""Export V6 with both input histories, both estimators, and Actor normalization."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/"HomieRL")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--output", type=Path, default=ROOT/"logs/export/v6")
    args = parser.parse_args(argv)
    import torch
    from pushcart_bench.policies_v6 import resolve_v6_checkpoint
    from homie_pushcart_v6.hl_policy import HighLevelPolicy
    from homie_pushcart_v6.physics import physics_manifest
    torch.set_num_threads(1)
    checkpoint = resolve_v6_checkpoint(args.checkpoint).resolve()
    policy = HighLevelPolicy(checkpoint)
    exported = torch.jit.script(policy.actor.as_jit().eval())
    args.output.mkdir(parents=True,exist_ok=True)
    output = args.output/"actor_and_estimators.jit"
    exported.save(str(output))
    reloaded = torch.jit.load(str(output)).eval()
    generator = torch.Generator().manual_seed(6)
    x = torch.randn(32,605,generator=generator)
    h = torch.randn(32,462,generator=generator)
    with torch.inference_mode():
        expected, actual = policy(x,h), reloaded(x,h)
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
        estimates = policy.physical_estimates(x,h)
        deployed = reloaded.physical_estimates(x,h)
        for a,b in zip(estimates,deployed):
            torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)
    manifest = {"version":6,"checkpoint":str(checkpoint),
        "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "export_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
        "network":policy.actor.metadata(),"physics":physics_manifest(),
        "cart_prediction_horizon_s":0.02,
        "inputs":{"policy_history":"5x121, oldest first, V3.2 frame minus handle xyz",
                  "him_history":"6x77, oldest first, HOMIE76 plus original relative yaw"},
        "outputs":{"forward":"18 original V3.2 high-level actions",
                   "physical_estimates":"cart COM vx/vy/wz; left/right palm force+moment in cart-heading axes"},
        "verification":{"samples":32,"max_action_error":float((expected-actual).abs().max()),
                        "max_estimate_errors":[float((a-b).abs().max()) for a,b in zip(estimates,deployed)]}}
    (args.output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"V6 exported and checked: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
