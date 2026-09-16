"""Exercise the registered Gymnasium task, wrapper, and timeout contract on GPU."""
from pathlib import Path
import json

from isaaclab.app import AppLauncher

launcher = AppLauncher({"headless": True})
try:
    import gymnasium as gym
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from homie_isaaclab import register_tasks
    from homie_isaaclab.lab_config import HomieDirectEnvCfg

    register_tasks()
    cfg = HomieDirectEnvCfg()
    cfg.scene.num_envs = 4
    cfg.enable_randomization = False
    env = gym.make("Homie-G1-Direct-v0", cfg=cfg)
    wrapped = RslRlVecEnvWrapper(env)
    obs = wrapped.get_observations()
    before = obs.clone()
    again = wrapped.get_observations()
    torch.testing.assert_close(before["policy"], again["policy"], rtol=0, atol=0)
    torch.testing.assert_close(before["critic"], again["critic"], rtol=0, atol=0)
    maximum_contact = 0.0
    for _ in range(100):
        obs, rewards, dones, extras = wrapped.step(torch.zeros(4, 12, device=wrapped.device))
        for tensor in (obs["policy"], obs["critic"], rewards, extras["homie_terminal_critic"]):
            assert torch.isfinite(tensor).all()
        maximum_contact = max(maximum_contact, env.unwrapped.homie.contact_forces.norm(dim=-1).max().item())
    wrapped.episode_length_buf[:] = wrapped.max_episode_length - 1
    obs, rewards, dones, extras = wrapped.step(torch.zeros(4, 12, device=wrapped.device))
    assert dones.all() and extras["time_outs"].all()
    assert extras["homie_terminal_ids"].numel() == 4
    assert extras["homie_terminal_critic"].shape == (4, 79)
    assert not torch.equal(extras["homie_terminal_critic"], obs["critic"])
    assert obs["policy"].shape == (4, 456) and obs["critic"].shape == (4, 79)
    assert maximum_contact > 0
    result = {"result": "PASS", "steps": 101, "num_envs": 4,
              "observation_query_preserves_history": True, "timeout_reset": True,
              "pre_reset_critic_preserved": True, "max_contact_force": maximum_contact}
    target = Path(__file__).resolve().parents[1] / "logs/nvidia_env_check.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    wrapped.close()
    print("OFFICIAL ENV PASS", result, flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    launcher.app.close(exit_code=1)
    raise
else:
    launcher.app.close()
