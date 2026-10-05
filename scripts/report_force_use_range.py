"""Write provenance, numerical summary and interpretation for the force-use figures."""
from pathlib import Path
import csv
import hashlib
import json
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "logs/Picture/force_use_range"
raw = json.loads((BASE / "derived/metrics.json").read_text())
smoothed = json.loads((BASE / "derived/smoothed95_metrics.json").read_text())
cap = json.loads((BASE / "derived/capability_bound.json").read_text())
manifest = json.loads((BASE / "manifest.json").read_text())

area_rows = []
for tag, label in (("xy", "Fx-Fy"), ("xz", "Fx-Fz"), ("yz", "Fy-Fz")):
    p = smoothed["planes"][tag]
    a, b = p["baseline"]["hdr95_area_N2"], p["CWC"]["hdr95_area_N2"]
    area_rows.append({"plane": label, "baseline_95pct_density_area_N2": a,
                      "V6_95pct_density_area_N2": b, "reduction_pct": 100*(1-b/a),
                      "common_bandwidth_N": p["bandwidth_N"],
                      "conditional_capability_circle_radius_N": cap["planes"][tag]["outer_circle_radius_N"]})
with (BASE / "derived/envelope_summary.csv").open("w", encoding="utf-8-sig", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(area_rows[0])); writer.writeheader(); writer.writerows(area_rows)

validation = {"initial_physics_and_state_exact_equal": manifest["initial_physics_and_state_exact_equal"],
              "raw_hashes_valid": True, "raw_sources": {}, "density_mass_checks": {},
              "smoothing_window_samples": 25, "capability_postures": len(cap["snapshots"])}
for method in ("baseline", "CWC"):
    path = BASE / "raw" / f"{method}.npz"
    meta = json.loads(path.with_suffix(".json").read_text())
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == meta["npz_sha256"]
    validation["raw_sources"][method] = {"sha256": actual, "samples": meta["samples"],
                                         "force_source": meta["force_measurement"]}
with np.load(BASE / "derived/force_envelopes_smoothed95.npz") as density:
    for tag in ("xy", "xz", "yz"):
        cell_area = float(density[tag+"_cell_area_N2"])
        for method in ("baseline", "CWC"):
            p = density[f"{tag}_{method}_density"]
            level = density[f"{tag}_{method}_thresholds"][-1]
            total = float(p.sum()*cell_area)
            retained = float(p[p >= level].sum()*cell_area)
            assert abs(total-1) < 1e-10 and .95 <= retained <= .953
            validation["density_mass_checks"][tag+"_"+method] = {"total": total, "retained": retained}
(BASE / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

lines = ["# Isaac 双手合力使用范围：V6 与 pass-through", "",
         "本目录为实际 Isaac Sim / PhysX 采集和派生图。15 kg 额外负载、标称巡线速度 0.5 m/s、seed=1、每种方法独立新进程运行一次。最新 V6 为 `2026-09-29_03-34-15_v6_v32_cart_him_hand_estimators_20k/model_19999.pt`。", "",
         "## 建议查看", "",
         "- [force_use_range.pdf](figures/force_use_range.pdf)：黑色能力参考圈与下面的线性放大视图，三列分别为 Fx-Fy、Fx-Fz、Fy-Fz。",
         "- [force_use_range_zoom.pdf](figures/force_use_range_zoom.pdf)：用于论文拼排的简洁三联图。只有平滑后的95%主体密度包络，V6为红色渐变，pass-through为蓝色。",
         "- 所有主图同时提供 PNG 与可编辑 SVG。PDF均经过栅格化目视检查。",
         "- [force_use_range_diagnostics.pdf](figures/force_use_range_diagnostics.pdf)：单独的原始数据审计，包含冲击峰值与运动跟踪。它不是本次要求的平滑主图。",
         "- `figures/full_rollout_envelopes.png`：整段运行的平滑95%密度诊断；两方法终端停靠时长不同，不用于主要配对结论。", "",
         "## 主图的数据处理", "",
         "1. 赛道为最新 `composite_course`：20 m弧长，正弦S弯振幅±0.7 m、2 m连接直线、1.25 m圆角半径、2.7 m绕行高度。两方法共用巡线PID/限速/初始状态/物理参数。cruise=true，末端仍执行同一停靠规则。",
         "2. 主图沿用前文确定的 `0 <= s < 19 m` 有效巡线区间；进度与力按同一物理时刻对齐。不把最后1 m与停车反复调整混入主图。50 Hz记录中，baseline有2121个力采样，V6有1919个；密度分别归一化为1。",
         "3. 对双手合力Fx、Fy、Fz分别使用相同的25点中心滑动平均（名义0.5 s）。边缘窗口截短并重新归一化，不补零、不跨越试验/reset。这是离线显示处理，不反馈给控制器。",
         "4. 每个二维投影使用两种方法共用的物理网格和带宽做Gaussian KDE。仅绘制累计概率质量50%、80%、95%的最高密度区域，最外层为95%。低密度的5%概率尾部不画入主体包络，不再绘制100%全样本凸包。",
         "5. 95%密度质量不等于机械地删除5%的离散样本。各投影的离散点纳入标记已写入 `derived/*_smoothed95_force.csv`；相应纳入率与阈值在 `derived/smoothed95_metrics.json` / `force_envelopes_smoothed95.npz`。平滑会减弱冲击，低密度尾部会被省略，但没有按方法人工选择有利样本。",
         "6. 同一概率定义不保证红色包络严格包含在蓝色内。颜色深浅代表该次轨迹中的时间密度，不代表跨重复试验的置信区间。", "",
         "## 主图结果", "",
         "| 投影 | Pass-through 95%面积 (N²) | V6 95%面积 (N²) | 面积缩小 |",
         "|---|---:|---:|---:|"]
for row in area_rows:
    lines.append(f"| {row['plane']} | {row['baseline_95pct_density_area_N2']:.2f} | {row['V6_95pct_density_area_N2']:.2f} | {row['reduction_pct']:.2f}% |")
lines += ["", "平滑后合力模长的95分位：35.93 N → 14.58 N；平均竖向合力：25.83 N → -0.61 N。前19 m的原始横向跟踪RMSE：0.1747 m → 0.0564 m；航向RMSE：20.12° → 3.47°；到达19 m的时间：42.44 s → 38.40 s。", "",
          "全程中V6在41.16 s满足终点半径0.10 m内保持0.5 s且窗口平均速度小于0.06 m/s的判据；pass-through虽到达路线末段，但在90 s内没有满足该停稳判据。两者没有触发机器人/抓握几何失败。", "",
          "可以据此表述：**在此配对Isaac试验与明确的平滑/密度口径下，V6改善路线跟踪，降低持续竖向载荷，Fx-Fz及Fy-Fz主体施力分布更集中；水平面主体面积基本不变。**", "",
          "不能直接表述为“所有方向的最大力都降低”或“已证明力控制精度提高”。此任务给的是速度/路线命令，没有独立的目标合力，所以包络面积不是力跟踪RMSE。V6与pass-through还同时改变了高层控制及估计器使用，不能把全部差异唯一归因于力估计器。", "",
          "原始数据中V6的最大合力与完整凸包比baseline大，原始局部波动RMS相近；这些记录保留在原始审计中。主图剔除低密度尾部后，结论应限定为主体分布，不应扩展为消除了瞬时冲击。", "",
          "## 黑色外圈究竟表示什么", "",
          "读取Isaac的左右掌心link-origin Jacobian、15个上身驱动关节（含共同腰关节）的实际力矩上限，以及重力补偿力矩。以手对车的两手六维wrench为变量w，构造准静态约束 `-tau_max <= g(q)+J(q)^T*w <= tau_max`。两手的竖向力和掌心力矩均为自由变量；投影到合力的各二维平面。",
          "在两条轨迹前19 m共162个记录姿态（含初始姿态）上求取约束集合的并集支撑函数。每5°方向线性规划一次，用支撑半平面形成保守外多边形；黑圈是该多边形关于零力原点的外接圆。圈半径Fx-Fy约1038 N、Fx-Fz约1074 N、Fy-Fz约803 N。",
          "这是**姿态相关、仅考虑关节驱动限制的乐观能力参考上界**。它没有验证腿部平衡、地面摩擦、手指抓握极限、瞬态惯性或未采样姿态，不等于机器人在任何姿态都能稳定施出的最大合力，也不表示圆内每个力都可实现。未用实测最大力冒充机械上限。由于参考上界与日常施力相差很大，主图提供了独立线性放大窗，未单独缩放任一方法。", "",
          "## 合力测量来源", "",
          "每只手的wrench来自PhysX incoming joint wrench减去掌/指子树Newton-Euler载荷（重力、平动惯性及角动量变化），方向为手作用于车。Fx/Fy为小车航向坐标，Fz为世界竖直方向。两手力直接相加；力矩保持独立，不把N与N·m混合。",
          "主包络使用此物理状态重建量，而不是V6自身预测的力。V6预测另行记录，仅用于估计误差核查。该重建与训练标签一致，但尚未通过独立动态接触传感器校准，因此称“重建的手对车合力”，不声称是独立传感器真值。控制、资产、奖励和接触设置未为这次图修改。", "",
          "## 数据与复现", "",
          "- `raw/`：两次独立Isaac路线记录，手wrench、合力所需分量、命令、实际速度、姿态、动作、估计输出、终止标志、能力计算快照、初末物理参数。",
          "- `derived/*_resultant_force.csv`：完整原始合力与主区间标记；`*_smoothed95_force.csv`：平滑合力及各投影的95%密度归属。",
          "- `derived/envelope_summary.csv`：包络面积对比；`smoothed95_metrics.json`为主图统计；`metrics.json`为原始审计。",
          "- `derived/capability_bound.json`：能力模型定义、支撑函数、外多边形和外圈。",
          "- `manifest.json`、`validation.json`、`plots_manifest.json`：配置、模型/原始数据/输出哈希及核验。",
          "- `runs/`：独立进程日志。第一份失败尝试是能力参数CPU/GPU索引问题，在正式采样前结束；后续成功采集为 `20260930_213440_6773590e`。没有把失败尝试拼入正式数据。",
          "- `archive/unsmoothed_*`：用户要求平滑95%版本之前的图，保留用于追溯。", "",
          "```powershell",
          "# 新采集请使用新的output目录，避免覆盖当前原始数据。",
          "& E:/envs/RLisaac601/python.exe scripts/collect_force_use_range_isaac.py --payload 15 --speed 0.5 --timeout 90 --output logs/Picture/force_use_range_repeat",
          "& E:/envs/RLisaac601/python.exe scripts/analyze_force_use_range.py",
          "& E:/envs/RLisaac601/python.exe scripts/plot_force_use_range.py",
          "& E:/envs/RLisaac601/python.exe scripts/report_force_use_range.py",
          "```", "",
          "Suggested caption: Dual-hand resultant force distributions during matched Isaac Sim route-following trials with a 15 kg payload and a nominal speed of 0.5 m/s. Blue denotes pass-through and red denotes V6. Forces are reconstructed from simulator joint wrenches and hand-subtree dynamics in the cart-heading frame. The first 19 m of the 20 m route are analyzed. Both methods use the same 0.5 s centered moving mean; nested contours contain 50%, 80%, and 95% of the projected KDE probability mass. The black circles bound a posture-conditioned actuator-only quasi-static model and do not certify whole-body force attainability. One nominal rollout is used per method.", ""]
(BASE / "README.md").write_text("\n".join(lines), encoding="utf-8")
prov = BASE / "provenance"; prov.mkdir(exist_ok=True)
for name in ("collect_force_use_range_isaac.py", "analyze_force_use_range.py", "plot_force_use_range.py", "report_force_use_range.py"):
    shutil.copy2(ROOT / "scripts" / name, prov / name)
print("Verified data hashes and 95% density mass; README and summaries saved.")
