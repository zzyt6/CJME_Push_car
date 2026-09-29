"""Verify fresh Isaac captures and present six separate force components."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "HomieRL")]
from eval_v52_fresh_isaac_axes import HEADS, CHECKPOINT, CONDITIONS
from summarize_v52_force_accuracy import replay
from summarize_v52_force_axis_accuracy import component_metrics

TITLES = {"full": "重训全输入网络（含力矩与物理先验）", "fullbody": "全身本体与全身控制历史网络", "original": "原检查点估力网络"}
AXES = ("Fx", "Fy", "Fz")


def main(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    torch.set_num_threads(1)
    records = {}
    lines = ["# 新采集 Isaac 数据：左右手各轴的力预测", "",
             "本次实际重新启动 Isaac/PhysX，固定三组既有网络权重，采集全新闭环轨迹。未重训、调整网络输出或用旧评测样本评分。", "",
             "新工况：载荷 **5、15、22.5 kg**；直行速度 **0.25、0.50 m/s**；转向指令 **0.35 m/s、±0.20 rad/s**，共 12 个组合。旧评测为 0、12.5、25 kg 及 0.3/0.6 m/s、±0.3 rad/s，本次载荷和指令组合均不同。这里的“新”指新采集轨迹及新评测组合，不声称所有数值都在训练分布之外。", "",
             "每组网络使用 12 个隔离并行环境，各运行 500 个控制步、10 秒；排除前 2 秒，每工况 400 个评分时刻，每组共 4,800 个。固定 seed=20260927，物理频率 200 Hz，控制频率 50 Hz，采用同一名义无噪声评测设置。机器人与车的物理参数、抓握和控制逻辑沿用 Isaac 原实现，只通过现有固定载荷/指令接口分配测试组合。", "",
             "分别统计左手和右手 Fx/Fy/Fz。偏差为 `预测−标签`，平均绝对偏差为该轴 `|预测−标签|` 的平均值；容差达标率也逐轴统计。参照是同一控制周期内的手部子树动力学标签，手对车方向、车体航向坐标系、单位 N。每帧标签均由 4 个物理子步平均得到。", "",
             "三组网络各自参与闭环。原检查点采用其混合驱动，两个重训网络采用显式 PD，所以跨网络差别包含各自的闭环轨迹与驱动差异。本次没有在 MuJoCo 上采集同一新工况；不能把本次结果和之前 MuJoCo 的差值直接归因于引擎。", ""]
    for name, (_, estimator) in HEADS.items():
        source = folder / name
        report = json.loads((source / "report.json").read_text(encoding="utf-8"))
        capture = torch.load(source / "samples.pt", map_location="cpu", weights_only=False)
        assert capture["report"] == report and report["new_collection"] and report["unchanged_sources"]
        assert report["completed_conditions"] == report["total_conditions"] == 12
        assert report["samples"] == 4800 and report["invalid_scored_steps"] == 0
        for fingerprint in report["source_hashes"]:
            assert hashlib.sha256((ROOT / fingerprint["path"]).read_bytes()).hexdigest() == fingerprint["sha256"]
        ids = capture["condition_id"].numpy()
        time_s = capture["time_s"].numpy()
        np.testing.assert_array_equal(capture["hand_substep_count"].numpy(), np.full(4800, 4.))
        prediction = capture["prediction"].double().numpy().reshape(-1, 2, 6)[:, :, :3]
        target = capture["target"].double().numpy().reshape(-1, 2, 6)[:, :, :3]
        assert np.isfinite(prediction).all() and np.isfinite(target).all()
        # Independent CPU replay catches asynchronous capture/order mistakes.
        raw, _ = replay(capture, dict(estimator_override=str(estimator) if estimator else None, checkpoint=str(CHECKPOINT)))
        replayed = raw * np.tile([50., 50., 50., 5., 5., 5.], 2)
        max_replay_diff = float(abs(replayed - capture["prediction"].numpy()).max())
        np.testing.assert_allclose(replayed, capture["prediction"].numpy(), atol=3e-4, rtol=1e-6)
        for i, c in enumerate(CONDITIONS):
            mask = ids == i
            assert mask.sum() == 400
            np.testing.assert_allclose(time_s[mask], np.arange(101, 501)*.02, atol=1e-10, rtol=0)
            np.testing.assert_allclose(capture["cart_mass_kg"][mask].numpy(), np.full(400, c["payload_kg"]+14.4), atol=1e-4, rtol=0)
            np.testing.assert_allclose(capture["command_task"][mask].numpy(), np.tile(c["command"], (400, 1)), atol=1e-7, rtol=0)
            for key, width in (("x", 136), ("falcon_x", 72)):
                history = capture[key][mask].numpy().reshape(-1, 5, width)
                np.testing.assert_array_equal(history[1:, :-1], history[:-1, 1:])
        metrics = component_metrics(prediction, target)
        assert metrics == report["axes"]
        delta = prediction - target
        index = np.unravel_index(np.argmax(abs(delta)), delta.shape)
        i, h, j = index
        worst = dict(condition=CONDITIONS[int(ids[i])], time_s=float(time_s[i]),
                     hand=("left", "right")[h], axis=AXES[j], target_N=float(target[index]),
                     prediction_N=float(prediction[index]), signed_error_N=float(delta[index]))
        examples = []
        for t in (3., 5., 8.):
            selected = np.flatnonzero((ids == 4) & np.isclose(time_s, t))
            assert len(selected) == 1
            idx = selected[0]
            examples.append(dict(time_s=t, payload_kg=15., command=[.25, 0., 0.],
                                 target_N=target[idx].tolist(), prediction_N=prediction[idx].tolist(),
                                 signed_error_N=delta[idx].tolist()))
        records[name] = dict(title=TITLES[name], axes=metrics, examples=examples, worst_sample=worst,
                             source_report=str((source / "report.json").relative_to(ROOT)),
                             source_samples_sha256=hashlib.sha256((source / "samples.pt").read_bytes()).hexdigest(),
                             verification=dict(history_exact=True, command_and_mass_correct=True,
                                               all_labels_four_substeps=True, cpu_replay_max_difference=max_replay_diff))
        lines += [f"## {TITLES[name]}", "",
                  "| 分量 | 平均绝对偏差 | 90% 样本偏差上界 | 偏差≤2 N | 偏差≤5 N | 最大绝对偏差 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for key, a in metrics.items():
            label = key.replace("left_", "左手 ").replace("right_", "右手 ")
            lines.append(f"| {label} | {a['mean_absolute_error_N']:.3f} N | {a['p90_absolute_error_N']:.3f} N | {a['within_N']['2']['percent']:.2f}% | {a['within_N']['5']['percent']:.2f}% | {a['max_absolute_error_N']:.3f} N |")
        w = worst
        lines += ["", f"最大单轴偏差发生在 {w['condition']['payload_kg']:g} kg、指令 `{w['condition']['command']}`、{w['time_s']:.2f} s：{w['hand']} {w['axis']} 标签 {w['target_N']:.3f} N，预测 {w['prediction_N']:.3f} N，偏差 {w['signed_error_N']:+.3f} N。该样本保留在全部统计中。", "",
                  "固定实例：15 kg、0.25 m/s 直行，在预先指定的 3/5/8 秒取样。数组依次为 `[Fx,Fy,Fz]`，单位 N。", "",
                  "| 时刻 | 手 | 仿真标签 | 网络预测 | 各轴偏差 |", "|---|---|---|---|---|"]
        fmt = lambda value: "[" + ", ".join(f"{v:.2f}" for v in value) + "]"
        for e in examples:
            for h, side in enumerate(("左手", "右手")):
                lines.append(f"| {e['time_s']:g} s | {side} | {fmt(e['target_N'][h])} | {fmt(e['prediction_N'][h])} | {fmt(e['signed_error_N'][h])} |")
        selected = ids == 4
        fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, layout="constrained")
        for j, axis in enumerate(AXES):
            for h, side in enumerate(("Left", "Right")):
                ax = axes[j, h]
                ax.plot(time_s[selected], target[selected, h, j], color="#185d83", lw=1.3, label="Isaac reference")
                ax.plot(time_s[selected], prediction[selected, h, j], color="#d07a2c", ls="--", lw=1.1, label="Frozen prediction")
                ax.set(title=f"{side} {axis}", ylabel="Force (N)")
                ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("Control interval end time (s)")
        fig.suptitle(f"Fresh Isaac / {name}: 15 kg payload, 0.25 m/s\nSix separate axes; no smoothing or time shift", fontsize=12)
        fig.savefig(source / "axis_prediction.png", dpi=155)
        plt.close(fig)
        lines += ["", f"![六轴预测]({name}/axis_prediction.png)", "", f"逐帧数值：[CSV]({name}/axis_samples.csv)。", ""]
        with (source / "axis_samples.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["condition_id", "time_s", "payload_kg", "command_name"] + [
                f"{side}_{axis}_{v}_N" for side in ("left", "right") for axis in AXES for v in ("reference", "prediction", "error")])
            for i in range(len(ids)):
                c = CONDITIONS[int(ids[i])]
                row = [int(ids[i]), float(time_s[i]), c["payload_kg"], c["command_name"]]
                for h in range(2):
                    for j in range(3):
                        row.extend((target[i, h, j], prediction[i, h, j], delta[i, h, j]))
                writer.writerow(row)
    lines += ["## 完整性核验", "",
              "三组均为 12/12 工况完成、每工况 400 个新评分时刻、无无效评分标签。核对了实际载荷和指令、完整四子步标签、历史帧连续性，以及冻结模型在 CPU 重放与在线 GPU 预测的一致性；源模型、HOMIE 权重和 USD 资产哈希保持不变。", "",
              "大多数时刻准确不代表没有瞬时失准。全输入网络在 15 kg、0.50 m/s、5.34 秒出现右手 Fx/Fz 的单帧大偏差，分别约 34.36/22.45 N；紧邻帧恢复到较小偏差。该现象的物理/估计原因尚未独立归因，本报告不剔除此样本。", "",
              "名义无噪声设置、单一随机种子和这些载荷/指令范围限定了结论。当前数据支持网络在 Isaac 源域具有较小的逐轴典型偏差，但不能据此排除跨域泛化问题。", "",
              "```powershell", "& E:\\envs\\RLisaac601\\python.exe scripts/eval_v52_fresh_isaac_axes.py --output logs/bench/another_fresh_isaac_run", "& E:\\envs\\RLisaac601\\python.exe scripts/summarize_v52_fresh_isaac_axes.py --folder logs/bench/another_fresh_isaac_run", "```", ""]
    (folder / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    (folder / "summary.json").write_text(json.dumps(dict(schema="v52_fresh_isaac_axes_summary_v1", results=records), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({name: r["verification"] for name, r in records.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=ROOT / "logs/bench/v52_isaac_fresh_axes_20260927")
    main(parser.parse_args().folder)
