"""Replay archived corrected-scene predictions and report force tolerance rates.

No simulation, training, calibration, smoothing, or time alignment is performed.
Each frozen head is replayed on its own saved closed-loop evaluation samples.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]
SOURCE = ROOT / "logs/bench/v52_sim2sim_root_cause"
CASES = {
    "full": ("重训全输入网络（含力矩与物理先验）", "full_physx_inertia", "isaac_explicit_pd"),
    "fullbody": ("全身本体与全身控制历史网络", "fullbody_final_alignment", "isaac_explicit_pd"),
    "original": ("原检查点内的估力网络", "original_checkpoint_final_alignment", "isaac"),
}
THRESHOLDS = (2, 5, 10, 20)


def fingerprint(path):
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def rates(values):
    return {str(t): dict(count=int((values <= t).sum()), denominator=len(values),
                        percent=100. * float((values <= t).mean())) for t in THRESHOLDS}


def accuracy(prediction, target):
    error = np.linalg.norm(prediction - target, axis=-1)
    load = np.linalg.norm(target, axis=-1)
    result = {"frames": len(error), "both_hands_within_N": rates(error.max(axis=1))}
    for hand, side in enumerate(("left", "right")):
        valid = load[:, hand] >= 10.
        relative = error[valid, hand] / load[valid, hand]
        result[side] = dict(
            within_N=rates(error[:, hand]),
            mean_error_N=float(error[:, hand].mean()),
            median_error_N=float(np.median(error[:, hand])),
            p90_error_N=float(np.quantile(error[:, hand], .9)),
            p95_error_N=float(np.quantile(error[:, hand], .95)),
            max_error_N=float(error[:, hand].max()),
            median_true_force_N=float(np.median(load[:, hand])),
            mean_component_bias_N=(prediction[:, hand] - target[:, hand]).mean(axis=0).tolist(),
            relative_error_for_true_force_ge_10N=dict(
                denominator=int(valid.sum()), excluded=int((~valid).sum()),
                median_percent=100. * float(np.median(relative)) if len(relative) else None,
                within_percent={str(t): dict(count=int((relative <= t / 100.).sum()),
                                            percent=100. * float((relative <= t / 100.).mean()) if len(relative) else None)
                                for t in (10, 20, 30)}))
    return result, error


def replay(capture, report):
    from homie_pushcart_v52.transfer_estimator import load_estimator
    from homie_pushcart_v52.falcon_history import fullbody_history
    if report["estimator_override"]:
        model_path = Path(report["estimator_override"])
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        model, _ = load_estimator(model_path)
        x = (fullbody_history(capture["x"], capture["falcon_x"])
             if model.features == "falcon_fullbody" else capture["x"])
        with torch.inference_mode():
            prediction = model(x)
    else:
        from homie_pushcart_v52.hl_policy import HighLevelPolicy
        from homie_pushcart_v52.task_config import task_cfg_from_run, num_actor_obs, num_critic_obs
        from tensordict import TensorDict
        model_path = Path(report["checkpoint"])
        task = task_cfg_from_run(model_path, verbose=False)
        actor = HighLevelPolicy(model_path, num_actor_obs(task), num_critic_obs(task), task.env.num_hl_actions).actor
        with torch.inference_mode():
            prediction = actor.predict_hand_wrench(TensorDict({"policy": capture["x"]}, batch_size=[len(capture["x"])]))
    return prediction.numpy(), model_path


def main(output):
    from homie_pushcart_v52.observation import HAND_WRENCH_SCALE
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    results, arrays = {}, {}
    for name, (title, folder, prefix) in CASES.items():
        report_path = SOURCE / folder / f"{prefix}_rigid_limit_physx_inertia.json"
        capture_path = report_path.with_name(report_path.stem + "_samples.pt")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        capture = torch.load(capture_path, map_location="cpu", weights_only=False)
        assert capture["schema"] == "v52_transfer_eval_v1" and capture["report"] == report
        assert report["engine"] == "mujoco" and report["completed_conditions"] == report["total_conditions"] == 12
        assert report["metadata"]["grasp_profile"] == "rigid_limit" and report["metadata"]["inertia_profile"] == "physx"
        assert all(row["failure"] is None for row in report["conditions"])
        prediction, model_path = replay(capture, report)
        scale = np.tile(np.asarray(HAND_WRENCH_SCALE), 2)
        prediction = prediction.astype(np.float64) * scale
        target = capture["y"].numpy().astype(np.float64) * scale
        assert prediction.shape == target.shape == (4800, 12)
        assert np.isfinite(prediction).all() and np.isfinite(target).all()
        delta = prediction - target
        # Check archived online statistics without using a new fit or a lag.
        np.testing.assert_allclose(delta.mean(axis=0), report["hands"]["bias"], atol=2e-5, rtol=0)
        replay_r2 = 1. - np.mean(delta * delta, axis=0) / np.var(target, axis=0)
        np.testing.assert_allclose(replay_r2, report["hands"]["r2"], atol=2e-5, rtol=0)
        pred_force, true_force = (value.reshape(-1, 2, 6)[:, :, :3] for value in (prediction, target))
        overall, error = accuracy(pred_force, true_force)
        ids = capture["condition_id"].numpy()
        conditions = []
        for i, row in enumerate(report["conditions"]):
            mask = ids == i
            assert mask.sum() == 400
            metrics, _ = accuracy(pred_force[mask], true_force[mask])
            conditions.append(dict(condition=row["condition"], payload_kg=row["payload_kg"], **metrics))
        # Predefined example: 12.5 kg payload, 0.3 m/s straight ahead, 3/5/8 s.
        chosen = np.flatnonzero(ids == 4)
        examples = []
        for time_s in (3., 5., 8.):
            local_index = round((time_s - report["warmup_s"]) / .02) - 1
            index = int(chosen[local_index])
            examples.append(dict(time_s=time_s, condition="forward03", payload_kg=12.5,
                                 sample_index=index, true_force_N=true_force[index].tolist(),
                                 predicted_force_N=pred_force[index].tolist(), error_N=error[index].tolist()))
        results[name] = dict(title=title, upper_drive=report["upper_drive"], overall=overall,
                             conditions=conditions, examples=examples,
                             provenance=dict(capture=fingerprint(capture_path), report=fingerprint(report_path),
                                             model=fingerprint(model_path)),
                             replay_check=dict(online_bias_max_difference=float(abs(delta.mean(0)-report["hands"]["bias"]).max()),
                                               online_r2_max_difference=float(abs(replay_r2-report["hands"]["r2"]).max())))
        arrays[name] = dict(prediction=pred_force, target=true_force, error=error, ids=ids)
    artifact = dict(
        schema="v52_corrected_scene_force_accuracy_v1", definition="Per-hand 3D Euclidean force error <= tolerance in N.",
        scene=dict(engine="mujoco", grasp_profile="rigid_limit", inertia_profile="physx", site_mapping_applied=False),
        sample_definition="4800 scored control frames per model: 12 conditions x 400 frames, 50 Hz, first 2 s excluded; seed 0.",
        reference="Hand-on-cart force from each MuJoCo grasp constraint, averaged over the control interval; cart-heading axes.",
        limits=["Tolerance thresholds are reporting choices, not established task success requirements.",
                "Consecutive frames are correlated, not 4800 independent trials.",
                "Each head has its own closed-loop trajectories; original uses mixed drives, retrained heads use explicit PD.",
                "Relative-error statistics only include reference force norms >= 10 N; counts are reported.",
                "These historical experimental profiles are not current defaults or proof of PhysX force equivalence."],
        results=results)
    (output / "accuracy.json").write_text(json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(output / "replayed_forces.npz", **{f"{name}_{k}": v for name, values in arrays.items() for k, v in values.items()})

    lines = ["# V5.2 修正配置中的力预测准确度", "",
             "**逐轴评估已单独提供：** 见 [左右手 Fx/Fy/Fz 预测报告](axis_accuracy.md)。本页保留先前的三维向量误差统计，不能当作各轴误差。", "",
             "这里只分析已保存的 `rigid_limit + physx` MuJoCo 实验数据；新提出的 site 抓握映射尚未应用。没有重训、重跑仿真、校正输出或平移时间轴。", "",
             "**准确度定义：** 每只手的三维力向量误差 `||预测力 − 标签力||₂` 不超过指定牛顿数的帧比例。该指标同时受大小和方向影响。5 N、10 N 是本次报告采用的容差示例，不是已经确定的任务验收标准。", "",
             "参照是 MuJoCo 同次仿真的抓握反力标签，方向为手对车、坐标轴为车体航向系。改写成车对手时预测与标签同时取反，以下误差和达标率不变。每个网络有 12 工况、4,800 个评分时刻（每工况 400 帧、50 Hz、排除最初 2 秒），seed=0，全部工况完成。相邻帧相关，因此百分比不是独立实验的成功概率。", "",
             "| 网络 | 左手 ≤5 N | 右手 ≤5 N | 左手 ≤10 N | 右手 ≤10 N | 双手同时 ≤10 N |", "|---|---:|---:|---:|---:|---:|"]
    for row in results.values():
        a = row["overall"]
        numbers = [a[s]["within_N"][str(t)]["percent"] for t in (5, 10) for s in ("left", "right")]
        numbers.append(a["both_hands_within_N"]["10"]["percent"])
        lines.append(f"| {row['title']} | " + " | ".join(f"{v:.2f}%" for v in numbers) + " |")
    lines += ["", "全输入网络包含关节力矩与物理先验；全身历史网络采用全身关节/IMU、下肢动作和上身目标历史。三者分别使用各自闭环轨迹，原检查点使用原混合驱动，两个重训网络使用显式 PD，因此该表是各自配置的实际表现，不是只改变网络的一因素对照。", ""]
    for name, row in results.items():
        lines += [f"## {row['title']}", "", "| 手 | 误差中位数 | 90% 帧的误差上界 | 真实力大小中位数 | 相对误差 ≤20% | 有效相对误差样本 |", "|---|---:|---:|---:|---:|---:|"]
        for side, label in (("left", "左手"), ("right", "右手")):
            a = row["overall"][side]
            r = a["relative_error_for_true_force_ge_10N"]
            lines.append(f"| {label} | {a['median_error_N']:.2f} N | {a['p90_error_N']:.2f} N | {a['median_true_force_N']:.2f} N | {r['within_percent']['20']['percent']:.2f}% | {r['denominator']}/4800 |")
        lines += ["", "相对误差只统计真实力大小 ≥10 N 的时刻；低载荷也完整计入上面的绝对误差达标率。", "",
                  "| 载荷 | 指令 | 左手 ≤5 N | 右手 ≤5 N | 双手同时 ≤10 N |", "|---|---|---:|---:|---:|"]
        for c in row["conditions"]:
            lines.append(f"| {c['payload_kg']:g} kg | {c['condition']} | {c['left']['within_N']['5']['percent']:.2f}% | {c['right']['within_N']['5']['percent']:.2f}% | {c['both_hands_within_N']['10']['percent']:.2f}% |")
        lines += ["", "固定示例：12.5 kg 载荷、0.3 m/s 直行，按预先指定的 3/5/8 秒取值，未按误差大小挑选。向量顺序为 `[Fx,Fy,Fz]`，单位 N。", "",
                  "| 时间 | 手 | 仿真标签 | 网络预测 | 三维误差 |", "|---|---|---|---|---:|"]
        fmt = lambda v: "[" + ", ".join(f"{x:.2f}" for x in v) + "]"
        for e in row["examples"]:
            for h, label in enumerate(("左", "右")):
                lines.append(f"| {e['time_s']:g} s | {label} | {fmt(e['true_force_N'][h])} | {fmt(e['predicted_force_N'][h])} | {e['error_N'][h]:.2f} N |")
        lines += ["", f"输入来源：`{row['provenance']['capture']['path']}`；模型：`{row['provenance']['model']['path']}`。", ""]
    lines += ["## 复核与复现", "", "三个网络离线重放的均值偏差及 R² 均与存档在线统计核对通过。这里没有把 R² 或 `1−相对误差` 当作准确率。原始快照、模型与报告的 SHA-256 见 [accuracy.json](accuracy.json)，逐帧力值见 `replayed_forces.npz`。", "",
              "```powershell", "& E:\\envs\\RLisaac601\\python.exe scripts/summarize_v52_force_accuracy.py", "```", ""]
    (output / "accuracy.md").write_text("\n".join(lines), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained", sharey=True)
    for name, label, color in (("full", "Retrained full input", "#1b6982"), ("fullbody", "Whole-body history", "#be6e24"), ("original", "Original checkpoint", "#757284")):
        for hand, ax in enumerate(axes):
            values = np.sort(arrays[name]["error"][:, hand])
            ax.plot(values, np.arange(1, len(values)+1)/len(values)*100, label=label, color=color, lw=1.7)
    for ax, side in zip(axes, ("Left hand", "Right hand")):
        ax.set(title=side, xlabel="3D force error tolerance (N)", xlim=(0, 20), ylim=(0, 100))
        ax.set_xticks((0, 2, 5, 10, 15, 20))
        ax.grid(alpha=.2)
        for threshold in (5, 10):
            ax.axvline(threshold, color="#999999", lw=.7, ls="--")
    axes[0].set_ylabel("Scored frames within tolerance (%)")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Archived MuJoCo rigid_limit + physx: 4,800 frames per model")
    fig.savefig(output / "force_accuracy.png", dpi=170)
    plt.close(fig)
    print(json.dumps({name: dict(left_5N=r["overall"]["left"]["within_N"]["5"]["percent"],
                                right_5N=r["overall"]["right"]["within_N"]["5"]["percent"],
                                both_10N=r["overall"]["both_hands_within_N"]["10"]["percent"])
                      for name, r in results.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "logs/bench/v52_corrected_force_accuracy")
    main(parser.parse_args().output)
