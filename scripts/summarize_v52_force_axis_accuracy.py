"""Report each hand's Fx/Fy/Fz from the verified corrected-scene replay.

Reads saved force arrays only. No new inference, simulation, smoothing, time
shift, hand summation, or vector-norm scoring is used in this report.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AXES = ("Fx", "Fy", "Fz")
SIDES = ("left", "right")
TOLERANCES_N = (2, 5, 10)


def component_metrics(prediction, target):
    delta = prediction - target
    absolute = np.abs(delta)
    return {
        f"{side}_{axis}": dict(
            n=len(delta), mean_absolute_error_N=float(absolute[:, h, j].mean()),
            signed_mean_error_N=float(delta[:, h, j].mean()),
            median_absolute_error_N=float(np.median(absolute[:, h, j])),
            p90_absolute_error_N=float(np.quantile(absolute[:, h, j], .9)),
            p95_absolute_error_N=float(np.quantile(absolute[:, h, j], .95)),
            max_absolute_error_N=float(absolute[:, h, j].max()),
            target_min_N=float(target[:, h, j].min()), target_max_N=float(target[:, h, j].max()),
            within_N={str(t): dict(count=int((absolute[:, h, j] <= t).sum()),
                                   denominator=len(delta),
                                   percent=100. * float((absolute[:, h, j] <= t).mean()))
                      for t in TOLERANCES_N})
        for h, side in enumerate(SIDES) for j, axis in enumerate(AXES)
    }


def main(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    input_report = folder / "accuracy.json"
    input_arrays = folder / "replayed_forces.npz"
    original = json.loads(input_report.read_text(encoding="utf-8"))
    assert original["schema"] == "v52_corrected_scene_force_accuracy_v1"
    result = dict(schema="v52_corrected_scene_force_axis_accuracy_v1",
                  definition="Six separately scored components; signed error = prediction - reference; tolerance uses absolute error.",
                  scene=original["scene"], reference=original["reference"], sample_definition=original["sample_definition"],
                  inputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (input_report, input_arrays)},
                  results={})
    lines = ["# V5.2 修正配置：左右手各轴的力预测", "",
             "本报告分别统计左手 Fx/Fy/Fz 与右手 Fx/Fy/Fz 六个分量。每轴偏差定义为 `预测值 − 仿真标签`，平均绝对偏差和达标率分别在该轴上计算。未进行左右手求和或三维向量合成。", "",
             "数据来自此前 `rigid_limit + physx` MuJoCo 实验；未应用新 site 映射。本次直接读取已核对在线统计的预测重放结果，没有重新仿真、重训、输出标定、平滑或时间平移。每个模型 12 工况、4,800 个评分时刻，每工况 400 帧；保留全部六轴有限值。", "",
             "标签是手对车的力，使用车体航向坐标轴，单位 N；标签按控制周期平均。正负号表示该轴的受力方向。若要表达车对手，同时将预测与标签反号即可。", "",
             "2 N、5 N、10 N 是明确列出的报告容差，尚非任务验收标准。相邻样本相关；三种网络各自参与闭环，原网络使用混合驱动，重训网络使用显式 PD，表格反映各自运行配置，不能据此单独归因于网络结构。", ""]
    with np.load(input_arrays, allow_pickle=False) as saved:
        for name, source in original["results"].items():
            prediction = saved[f"{name}_prediction"]
            target = saved[f"{name}_target"]
            ids = saved[f"{name}_ids"]
            assert prediction.shape == target.shape == (4800, 2, 3)
            assert ids.shape == (4800,) and np.isfinite(prediction).all() and np.isfinite(target).all()
            # Independent ordering check against the earlier full-wrench bias.
            old_report_path = ROOT / source["provenance"]["report"]["path"]
            assert hashlib.sha256(old_report_path.read_bytes()).hexdigest() == source["provenance"]["report"]["sha256"]
            old_report = json.loads(old_report_path.read_text(encoding="utf-8"))
            expected_bias = np.asarray(old_report["hands"]["bias"]).reshape(2, 6)[:, :3]
            np.testing.assert_allclose((prediction-target).mean(axis=0), expected_bias, atol=2e-5, rtol=0)
            overall = component_metrics(prediction, target)
            conditions, examples = [], []
            frame_times = np.empty(len(ids))
            for i, condition in enumerate(source["conditions"]):
                selected = ids == i
                assert selected.sum() == 400
                frame_times[selected] = old_report["warmup_s"] + (np.arange(selected.sum()) + 1) * .02
                conditions.append(dict(condition_id=i, condition=condition["condition"],
                                       payload_kg=condition["payload_kg"],
                                       axes=component_metrics(prediction[selected], target[selected])))
            for e in source["examples"]:
                index = e["sample_index"]
                np.testing.assert_array_equal(prediction[index], e["predicted_force_N"])
                np.testing.assert_array_equal(target[index], e["true_force_N"])
                assert abs(frame_times[index] - e["time_s"]) < 1e-10
                examples.append(dict(time_s=e["time_s"], condition=e["condition"], payload_kg=e["payload_kg"],
                                     target_N=target[index].tolist(), prediction_N=prediction[index].tolist(),
                                     signed_error_N=(prediction[index]-target[index]).tolist()))
            result["results"][name] = dict(title=source["title"], upper_drive=source["upper_drive"],
                                           provenance=source["provenance"], axes=overall,
                                           conditions=conditions, examples=examples)
            lines += [f"## {source['title']}", "",
                      "| 分量 | 平均绝对偏差 | 平均有符号偏差 | 90% 样本绝对偏差上界 | 偏差≤2 N | 偏差≤5 N | 偏差≤10 N |",
                      "|---|---:|---:|---:|---:|---:|---:|"]
            for key, a in overall.items():
                label = key.replace("left_", "左手 ").replace("right_", "右手 ")
                rate = " | ".join(f"{a['within_N'][str(t)]['percent']:.2f}%" for t in TOLERANCES_N)
                lines.append(f"| {label} | {a['mean_absolute_error_N']:.2f} N | {a['signed_mean_error_N']:+.2f} N | {a['p90_absolute_error_N']:.2f} N | {rate} |")
            lines += ["", "平均有符号偏差会抵消正负误差，只用来判断系统偏移；平均绝对偏差表示该轴通常相差多少。", "",
                      "### 固定时刻的逐轴对照", "",
                      "取 12.5 kg 载荷、0.3 m/s 直行的第 3、5、8 秒，沿用先前预先指定的时刻。数值单位 N；向量内部依次为 `[Fx,Fy,Fz]`。", "",
                      "| 时刻 | 手 | 仿真标签 | 网络预测 | 各轴偏差：预测−标签 |", "|---|---|---|---|---|"]
            fmt = lambda v: "[" + ", ".join(f"{x:.2f}" for x in v) + "]"
            for e in examples:
                for h, label in enumerate(("左手", "右手")):
                    lines.append(f"| {e['time_s']:g} s | {label} | {fmt(e['target_N'][h])} | {fmt(e['prediction_N'][h])} | {fmt(e['signed_error_N'][h])} |")

            # Six independent traces; same preselected condition for each model.
            mask = ids == 4
            fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, layout="constrained")
            for j, axis in enumerate(AXES):
                for h, side in enumerate(("Left", "Right")):
                    ax = axes[j, h]
                    ax.plot(frame_times[mask], target[mask, h, j], color="#185d83", lw=1.4, label="Simulation reference")
                    ax.plot(frame_times[mask], prediction[mask, h, j], color="#d07a2c", lw=1.2, ls="--", label="Estimator prediction")
                    ax.set(title=f"{side} {axis}", ylabel="Force (N)")
                    ax.grid(alpha=.2)
            axes[0, 0].legend(fontsize=8, loc="best")
            for ax in axes[-1]:
                ax.set_xlabel("Control interval end time (s)")
            model_title = {"full": "Retrained full input", "fullbody": "Whole-body history", "original": "Original checkpoint"}[name]
            fig.suptitle(f"{model_title}: each hand / axis, 12.5 kg payload, 0.3 m/s\nArchived MuJoCo rigid_limit + physx; no smoothing or time shift", fontsize=12)
            image_name = f"{name}_axis_prediction.png"
            fig.savefig(folder / image_name, dpi=160)
            plt.close(fig)
            lines += ["", f"![六轴预测与标签时序]({image_name})", "",
                      f"逐帧数值：[CSV]({name}_axis_samples.csv)；模型：`{source['provenance']['model']['path']}`。", ""]
            fields = ["condition_id", "condition", "payload_kg", "interval_end_s"] + [
                f"{side}_{axis}_{value}_N" for side in SIDES for axis in AXES for value in ("reference", "prediction", "signed_error")]
            with (folder / f"{name}_axis_samples.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(fields)
                for i in range(len(ids)):
                    condition = source["conditions"][int(ids[i])]
                    row = [int(ids[i]), condition["condition"], condition["payload_kg"], frame_times[i]]
                    for h in range(2):
                        for j in range(3):
                            row.extend((target[i, h, j], prediction[i, h, j], prediction[i, h, j]-target[i, h, j]))
                    writer.writerow(row)
    lines += ["## 数据与复现", "", "全部工况的逐轴统计、原始输入哈希与模型来源见 [axis_accuracy.json](axis_accuracy.json)。旧的三维向量统计保留在 [accuracy.md](accuracy.md)，本页才是逐轴评估口径。", "",
              "```powershell", "& E:\\envs\\RLisaac601\\python.exe scripts/summarize_v52_force_axis_accuracy.py", "```", ""]
    (folder / "axis_accuracy.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (folder / "axis_accuracy.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({name: {axis: dict(mae_N=a["mean_absolute_error_N"], within5_percent=a["within_N"]["5"]["percent"])
                            for axis, a in row["axes"].items()} for name, row in result["results"].items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=ROOT / "logs/bench/v52_corrected_force_accuracy")
    main(parser.parse_args().folder)
