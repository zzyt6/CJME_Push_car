"""Path-following figure of the Isaac composite-course runs (scripts/course_eval_isaac_v7.py --tag _plot).

Top row: cart trajectories of every environment over the reference course, per payload.
Bottom row: |cross-track error| along the course progress s, median and p10-p90 band over environments.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "logs/validation/v7_course_isaac"
COLORS = {"V6": "#2a78d6", "V7": "#eb6834"}          # validated categorical slots 1 / 2 (light surface)
REF, INK, INK2, SURFACE = "#52514e", "#0b0b0b", "#52514e", "#fcfcfb"


def load(policy, payload, difficulty, tag):
    f = SRC / f"traj_{policy}_{payload:g}kg_d{difficulty:g}{tag}.npz"
    if not f.is_file():
        return None
    z = np.load(f, allow_pickle=True)
    envs = [z[k] for k in sorted((k for k in z.files if k.startswith("env")), key=lambda k: int(k[3:]))]
    return {"course": z["course_xy"], "status": z["status"], "envs": [e for e in envs if len(e)]}


def first19_rms(envs):
    vals = np.concatenate([e[e[:, 3] < 19.0, 4] for e in envs])
    return float(np.sqrt(np.mean(vals ** 2)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--payloads", nargs="+", type=float, default=[0.0, 12.5, 25.0])
    p.add_argument("--difficulty", type=float, default=0.0)
    p.add_argument("--tag", default="_plot")
    p.add_argument("--labels", nargs=2, default=["V6 model_19999", "V7 rev9 model_10000"])
    p.add_argument("--out", default=str(ROOT / "logs/bench/isaac_course_v6_v7_20261008/path_composite_course_isaac.png"))
    a = p.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    label = dict(zip(("V6", "V7"), a.labels))
    rows = len(a.payloads)
    fig, axes = plt.subplots(rows, 3, figsize=(19, 5.0 * rows), squeeze=False,
                             gridspec_kw={"width_ratios": [2.3, 1.0, 1.7]})
    fig.patch.set_facecolor(SURFACE)
    bins = np.arange(0.0, 20.01, 0.25)
    for j, payload in enumerate(a.payloads):
        ax, zx, bx = axes[j]
        for x in (ax, zx, bx):
            x.set_facecolor(SURFACE)
            for side in ("top", "right"):
                x.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                x.spines[side].set_color("#c9c8c3")
            x.tick_params(colors=INK2, labelsize=9)
            x.grid(color="#e6e5e1", lw=0.8)
        bx.axvspan(12.0, 16.0, color="#f0efec", zorder=0)
        ref_drawn, rms_txt = False, []
        for pol in ("V6", "V7"):
            d = load(pol, payload, a.difficulty, a.tag)
            if d is None:
                continue
            many = len(d["envs"]) > 1
            lw, al = (1.0, 0.45) if many else (2.0, 1.0)
            if not ref_drawn:
                xy = d["course"]
                for x in (ax, zx):
                    x.plot(xy[:, 0], xy[:, 1], ls=(0, (5, 3)), color=REF, lw=1.6, zorder=3, label="参考路线")
                ax.plot(*xy[-1], marker="*", ms=12, color=INK, zorder=4, ls="none", label="终点")
                corner = xy[(xy[:, 0] > 10.8) & (xy[:, 1] > -0.5)]
                ref_drawn = True
            c = COLORS[pol]
            for k, e in enumerate(d["envs"]):
                lab = f"{label[pol]}" + (f"（{len(d['envs'])} 个环境）" if many else "") if k == 0 else None
                ax.plot(e[:, 1], e[:, 2], color=c, lw=lw, alpha=al, zorder=2, label=lab)
                zx.plot(e[:, 1], e[:, 2], color=c, lw=lw, alpha=al, zorder=2)
                inside = e[(e[:, 1] > corner[:, 0].min() - 0.4) & (e[:, 1] < corner[:, 0].max() + 0.4)]
                if len(inside):
                    corner = np.vstack([corner, inside[:, 1:3]])
            per_env = []
            for e in d["envs"]:
                idx = np.digitize(e[:, 3], bins) - 1
                row = np.full(len(bins) - 1, np.nan)
                for b in range(len(bins) - 1):
                    m = idx == b
                    if m.any():
                        row[b] = np.abs(e[m, 4]).mean()
                per_env.append(row)
            P = np.array(per_env)
            mid = 0.5 * (bins[1:] + bins[:-1])
            if many:
                bx.fill_between(mid, np.nanpercentile(P, 10, 0), np.nanpercentile(P, 90, 0), color=c, alpha=0.18, lw=0)
            bx.plot(mid, np.nanmedian(P, 0), color=c, lw=2, label=label[pol])
            ok = int(np.sum(d["status"] == "reached"))
            rms_txt.append(f"{pol} 前 19 m 横向 RMS {first19_rms(d['envs']):.3f} m，最大 "
                           f"{max(np.abs(e[e[:, 3] < 19, 4]).max() for e in d['envs']):.3f} m，到达 {ok}/{len(d['status'])}")
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]", color=INK2)
        ax.set_ylabel("y [m]", color=INK2)
        ax.set_title(f"负载 {payload:g} kg · 全路线", color=INK, fontsize=12.5, loc="left")
        ax.legend(fontsize=9, loc="upper left", frameon=False)
        ax.text(0.0, -0.36, "\n".join(rms_txt), transform=ax.transAxes, color=INK, fontsize=10, va="top")
        zx.set_xlim(corner[:, 0].min() - 0.4, corner[:, 0].max() + 0.4)
        zx.set_ylim(corner[:, 1].min() - 0.4, corner[:, 1].max() + 0.4)
        zx.set_aspect("equal")
        zx.set_title("圆角绕行段放大", color=INK, fontsize=12.5, loc="left")
        zx.set_xlabel("x [m]", color=INK2)
        bx.set_xlim(0, 20)
        bx.set_ylim(bottom=0)
        bx.set_title("|横向误差| 沿路线分布" + ("（中位数与 p10–p90）" if many else ""), color=INK, fontsize=12.5, loc="left")
        bx.set_xlabel("沿路线进度 s [m]（灰底：12–16 m 圆角绕行段）", color=INK2)
        bx.set_ylabel("|横向误差| [m]", color=INK2)
        bx.legend(fontsize=9, loc="upper left", frameon=False)
    top = 1 - 0.13 / rows
    fig.suptitle("Isaac / PhysX（V7 训练场景）· 20 m 综合路线 · 巡线 0.5 m/s", color=INK, fontsize=14, x=0.01, ha="left",
                 y=1 - 0.012 / rows)
    fig.text(0.01, 1 - 0.07 / rows, f"难度 d = {a.difficulty:g}：物理参数在标称值 ±25 % 内随机，无观测噪声、无扰动；两者使用同一个巡线器与 HOMIE",
             color=INK2, fontsize=10, ha="left")
    fig.tight_layout(rect=(0, 0, 1, top), h_pad=4.0)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
