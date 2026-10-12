"""Collect V8 deployable frames + explicit labels from Isaac, the teacher driving at full difficulty (curriculum d = 1,
randomization on, random commands and payloads), for offline estimator probes (scripts/v8_probe_train.py).

    E:\\envs\\RLisaac601\\python.exe scripts/v8_probe_collect.py --checkpoint <teacher model.pt> --num_envs 512 --steps 2000 --headless
Saves logs/validation/v8_probe/data.npz: frames [T, N, 126] (fp16), explicit [T, N, 19], step [T, N] (episode step).
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]


def main():
    from isaaclab.app import AppLauncher
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num_envs", type=int, default=512)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--out", default=str(ROOT / "logs/validation/v8_probe/data.npz"))
    p.add_argument("--student", default="", help="a distilled student drives instead of the teacher (DAgger data)")
    p.add_argument("--action_noise", type=float, default=0.0, help="Gaussian action noise (distillation: 0.05)")
    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args()
    app = AppLauncher(a)
    try:
        import numpy as np
        import torch
        from homie_isaaclab.config import G1RoughCfg
        from homie_pushcart_v7.baseline import apply_config, homie_baseline
        from homie_pushcart_v8.task_config import task_cfg_from_run
        from homie_pushcart_v8.task_env import PushCartEnv
        from homie_pushcart_v8.training import homie_velocity_file
        from homie_pushcart.hl_policy import import_rsl_rl5
        import_rsl_rl5()
        from homie_pushcart_v8.teacher_policy import load_teacher
        dev, N = a.device, a.num_envs
        cfg = G1RoughCfg(); apply_config(cfg, homie_baseline()); cfg.env.num_envs = N
        task = task_cfg_from_run(a.checkpoint)
        est = homie_velocity_file(a.checkpoint)
        task.v8.homie_velocity.init_checkpoint = str(est) if est else ""
        task.v8.homie_velocity.train = False
        task.curriculum_v7.enabled = False
        task.v8.emit_history = bool(a.student)
        env = PushCartEnv(cfg, task, ROOT, dev, True, level=0)
        env.cur.d[:] = 1.0
        env._refresh_global_level()
        env.reset_idx(torch.arange(N, device=dev)); env.reset()
        pol = load_teacher(a.checkpoint, dev)
        if a.student:
            from homie_pushcart_v8.student_policy import load_student
            stu = load_student(a.student, dev)
            pol = lambda policy, privileged: stu(policy, env.observation_dict()["history"])     # noqa: E731
        frames = np.zeros((a.steps, N, 126), np.float16)
        explicit = np.zeros((a.steps, N, 62), np.float32)          # full privileged vector (explicit = first 19)
        step = np.zeros((a.steps, N), np.int32)
        extra = np.zeros((a.steps, N, 1), np.float32)              # log2 arm drive scale ("priv_extra"), 0 if off
        for t in range(a.steps):
            o = env.observation_dict()
            with torch.inference_mode():
                act = pol(o["policy"], o["privileged"])
                if a.action_noise > 0:
                    act = act + a.action_noise * torch.randn_like(act)
            env.step(act.clone())
            frames[t] = env.frames_short[:, -1].cpu().numpy().astype(np.float16)
            explicit[t] = env.privileged.cpu().numpy()
            step[t] = env.episode_length_buf.cpu().numpy()
            if hasattr(env, "arm_gain_scale"):
                extra[t, :, 0] = torch.log2(env.arm_gain_scale).cpu().numpy()
            if t % 250 == 0:
                print(f"[probe] step {t}/{a.steps}", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.out, frames=frames, privileged=explicit, step=step, priv_extra=extra)
        print("[probe] saved", a.out, flush=True)
    except BaseException:
        import traceback
        traceback.print_exc(); sys.stdout.flush()
        raise
    finally:
        app.app.close()


if __name__ == "__main__":
    main()
