"""Render the original course at 663 x 189 px / 96 dpi with 8 pt text.

The PDF is 497.25 x 141.75 PDF points (175.41875 x 50.00625 mm).
Place it at its native size in the paper to retain the 8 pt typography.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pushcart_bench.paths import build_path

WIDTH_PX, HEIGHT_PX, DPI, FONT_PT = 663, 189, 96, 8
STEM = "composite_course_663x189_8pt"
ROUTE_COLOR = "#146b9f"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "pdf")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the geometry recorded for the user's reference figure.
    geometry = json.loads((ROOT / "docs/images/composite_course_geometry.json").read_text(encoding="utf-8"))
    course = build_path("composite_course", geometry["config"])
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": FONT_PT,
        "axes.labelsize": FONT_PT,
        "axes.titlesize": FONT_PT,
        "xtick.labelsize": FONT_PT,
        "ytick.labelsize": FONT_PT,
        "legend.fontsize": FONT_PT,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "savefig.bbox": None,
        "path.simplify": False,
    })
    fig = plt.figure(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI)
    width_pt, height_pt = np.array(fig.get_size_inches()) * 72
    xlim, ylim = (-0.5, course.goal[0] + 0.6), (-1.15, 3.0)
    # Set the plot box in physical points, retaining equal x/y metre scales.
    axes_width = 439.0
    axes_height = axes_width * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    ax = fig.add_axes([34 / width_pt, 24 / height_pt,
                       axes_width / width_pt, axes_height / height_pt])
    ax.set(xlim=xlim, ylim=ylim, xlabel="x (m)", ylabel="y (m)", aspect="equal")
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = 3
    ax.set_xticks(np.arange(0, 17, 2))
    ax.set_yticks(np.arange(-1, 4))
    ax.tick_params(length=2.5, width=0.6, pad=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#bfc5ca", alpha=0.27, linewidth=0.45)
    ax.set_axisbelow(True)
    ax.axhline(0, color="#aab8ca", linestyle="--", linewidth=0.6, zorder=1)
    ax.plot(*course.xy.T, color=ROUTE_COLOR, linewidth=1.7, solid_capstyle="round", zorder=3)

    for target in [(1.9, 0.45), (6.5, -0.57), (9.7, 0),
                   (11.52, 1.02), (13.52, 2.68), (15.5, 0.04)]:
        index = int(np.argmin(np.linalg.norm(course.xy - target, axis=1)))
        s = course.s[index]
        ax.annotate("", xy=course.point_at(s + 0.13)[0],
                    xytext=course.point_at(s - 0.13)[0], zorder=4,
                    arrowprops={"arrowstyle": "->", "color": ROUTE_COLOR,
                                "lw": 1.15, "mutation_scale": 7,
                                "shrinkA": 0, "shrinkB": 0})

    ax.scatter(*course.xy[0], s=27, color="#219943", edgecolors="none", zorder=5)
    ax.scatter(*course.xy[-1], s=27, color="#d6372f", marker="s", edgecolors="none", zorder=5)
    leaders = {"arrowstyle": "-", "color": "#666666", "lw": 0.7,
               "shrinkA": 2, "shrinkB": 3}
    ax.annotate("Start", xy=course.xy[0], xytext=(0.03, -0.75),
                ha="left", va="top", arrowprops=leaders)
    ax.annotate("Finish", xy=course.xy[-1], xytext=(course.goal[0], -0.75),
                ha="right", va="top", arrowprops=leaders)
    sine_peak = course.xy[int(np.argmax(np.where(course.xy[:, 0] < 8.3, course.xy[:, 1], -np.inf)))]
    ax.annotate("Sinusoidal S (amplitude \u00b10.7 m)", xy=sine_peak,
                xytext=(1.75, 1.68), ha="left", va="bottom", arrowprops=leaders)
    ax.annotate("Rounded rectangular detour", xy=(13.05, 2.7),
                xytext=(12.65, 3.04), ha="center", va="bottom",
                arrowprops=leaders, annotation_clip=False)

    connector_start = course.params["lead"] + course.params["s_length"]
    connector_end = connector_start + course.params["middle"]
    ax.annotate("", xy=(connector_end, -0.52), xytext=(connector_start, -0.52),
                arrowprops={"arrowstyle": "<->", "color": "#555555", "lw": 0.7,
                            "mutation_scale": 7, "shrinkA": 0, "shrinkB": 0})
    ax.text((connector_start + connector_end) / 2, -0.75, "2 m connector",
            ha="center", va="top")

    # No tight cropping: it would change both the requested pixel dimensions
    # and the physical size on which the 8 pt requirement depends.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        if artist.get_fontsize() != FONT_PT:
            raise ValueError(f"Unexpected font size: {artist.get_text()}")
        bounds = artist.get_window_extent(renderer)
        if (bounds.x0 < 0 or bounds.y0 < 0
                or bounds.x1 > WIDTH_PX or bounds.y1 > HEIGHT_PX):
            raise ValueError(f"Text exceeds canvas: {artist.get_text()}: {bounds.bounds}")

    for extension in ("pdf", "png", "svg"):
        destination = args.output_dir / f"{STEM}.{extension}"
        fig.savefig(destination, dpi=DPI, bbox_inches=None, pad_inches=0)
        if extension == "svg":
            # SVG uses the same physical size as PDF, with explicit CSS pixels.
            svg = destination.read_text(encoding="utf-8")
            svg = svg.replace('width="497.25pt" height="141.75pt"',
                              'width="663px" height="189px"', 1)
            destination.write_text(svg, encoding="utf-8")
        print(destination)
    plt.close(fig)


if __name__ == "__main__":
    main()
