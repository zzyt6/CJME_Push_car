"""Rebuild Picture.pptx slide 5 from original simulation records and source photos."""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
from PIL import Image

from plot_motion_ability import load_trial, validate_pair, local_display, plot_panel, sha256

ROOT = Path(__file__).resolve().parents[1]
SIZE_MM = (180., 130.)
AXIS_LEFT = (32., 83., 134.)
ROW_TOP = (46., 86.)
PLOT_MM = (40., 32.5)
AXIS_KEYS = ("vx", "vy", "wz")
TITLES = ("Straight-line motion", "Lateral motion", "Turning motion")
COLORS = {"CWC": "#0072B2", "baseline": "#D55E00"}
PHOTO_SOURCE_RECT = (76.9055277778, 48.6940833333, 156.0015555556, 31.6628888889)


def asset_records(pptx, assets):
    # The original PPT has three photos plus separate transparent arrow images.
    # Preserve each image and its original crop; never extract a plot screenshot.
    sources = [
        ("image60.jpeg", (76.9055278,48.6940833,52.6494722,31.6628889), {"l":3861}),
        ("image59.png", (129.5550278,48.6940833,51.6760278,31.6628889), {"r":5639,"b":7353}),
        ("image58.png", (181.2310556,48.6940833,51.6760278,31.6628889), {"t":1785,"r":2426,"b":18500}),
        ("image64.png", (98.288,54.4666667,9.1175556,3.7632778), {}),
        ("image65.png", (153.1861944,53.1675556,9.1175556,3.7264444), {}),
        ("image64.png", (155.8571944,53.0666667,9.1175556,3.7632778), {}),
        ("image63.png", (203.8879444,62.2437778,6.36225,6.4828056), {}),
    ]
    scale = 153. / PHOTO_SOURCE_RECT[2]
    result = []
    with zipfile.ZipFile(pptx) as archive:
        for index, (name, rect, crop) in enumerate(sources):
            original = archive.read("ppt/media/"+name)
            original_path = assets / name
            if not original_path.exists(): original_path.write_bytes(original)
            img = Image.open(io.BytesIO(original)).convert("RGBA")
            w,h = img.size
            box = (round(w*crop.get("l",0)/1e5), round(h*crop.get("t",0)/1e5),
                   round(w*(1-crop.get("r",0)/1e5)), round(h*(1-crop.get("b",0)/1e5)))
            img = img.crop(box)
            display_path = assets / f"display_{index}.png"
            img.save(display_path)
            x,y,rw,rh = rect
            target = [23.+(x-PHOTO_SOURCE_RECT[0])*scale,
                      11.+(y-PHOTO_SOURCE_RECT[1])*scale,rw*scale,rh*scale]
            result.append(dict(path=str(display_path.resolve()), original_part="ppt/media/"+name,
                               original_sha256=sha256(original_path), crop=crop,
                               rect_mm=target, pixels=list(img.size)))
    return result


def configure_font():
    plt.rcParams.update({"font.family":"Arial", "font.style":"normal", "font.weight":"normal",
        "font.size":8., "axes.labelsize":8., "axes.labelweight":"normal",
        "xtick.labelsize":8., "ytick.labelsize":8., "legend.fontsize":7.,
        "mathtext.fontset":"custom", "mathtext.rm":"Arial", "mathtext.it":"Arial:italic",
        "mathtext.bf":"Arial:bold", "mathtext.sf":"Arial", "mathtext.default":"regular",
        "pdf.fonttype":42, "svg.fonttype":"none", "axes.unicode_minus":True})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,default=Path(r"C:\Users\Administrator\Desktop\Picture.pptx"))
    parser.add_argument("--output",type=Path,default=ROOT/"logs/Picture/motion_composite")
    args=parser.parse_args()
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    assets=output/"assets";assets.mkdir(exist_ok=True)
    derived=output/"data";derived.mkdir(exist_ok=True)
    configure_font()
    images=asset_records(args.reference,assets)
    fig=plt.figure(figsize=np.array(SIZE_MM)/25.4)
    for item in images:
        x,y,w,h=item["rect_mm"]
        ax=fig.add_axes([x/SIZE_MM[0],1-(y+h)/SIZE_MM[1],w/SIZE_MM[0],h/SIZE_MM[1]])
        ax.imshow(Image.open(item["path"]),aspect="auto",interpolation="none")
        ax.set_axis_off()
    texts=[]
    for index,title in enumerate(TITLES):
        x=23.+51.*index+25.5
        fig.text(x/SIZE_MM[0],1-4.9/SIZE_MM[1],title,ha="center",va="center",fontsize=9)
        texts.append(dict(text=title,x_mm=x,y_mm=4.9,font_pt=9,align="center"))
    for row,title in enumerate(("Isaac Sim","MuJoCo")):
        y=ROW_TOP[row]+PLOT_MM[1]/2
        fig.text(11./SIZE_MM[0],1-y/SIZE_MM[1],title,ha="center",va="center",fontsize=9)
        texts.append(dict(text=title,x_mm=11.,y_mm=y,font_pt=9,align="center"))
    fig.text(.555,1-126./SIZE_MM[1],"Time (s)",ha="center",va="center",fontsize=8)
    texts.append(dict(text="Time (s)",x_mm=.555*SIZE_MM[0],y_mm=126.,font_pt=8,align="center"))
    panels=[]; sources={}
    for row,root in enumerate((ROOT/"logs/Picture/motion_ability",ROOT/"logs/Picture/motion_ability/mujoco")):
        previous=json.loads((root/"plots_manifest.json").read_text(encoding="utf-8"))
        window=previous["display"]["window_s"]
        offset=previous["display"]["initialization_hold_s"]
        trials={(axis,method):load_trial(root,axis,method) for axis in AXIS_KEYS for method in COLORS}
        for axis in AXIS_KEYS:validate_pair(trials[(axis,"CWC")],trials[(axis,"baseline")])
        display={key:local_display(trial,window) for key,trial in trials.items()}
        for trial in trials.values():
            for path in trial.source_paths:sources[str(path)]=sha256(path)
            if sha256(trial.source_paths[0]) != trial.metadata["npz_sha256"]:
                raise ValueError("Raw source checksum differs from collection record")
        for col,axis in enumerate(AXIS_KEYS):
            left,top=AXIS_LEFT[col],ROW_TOP[row];width,height=PLOT_MM
            ax=fig.add_axes([left/SIZE_MM[0],1-(top+height)/SIZE_MM[1],width/SIZE_MM[0],height/SIZE_MM[1]])
            handles=plot_panel(ax,axis,trials,display,raw=False,show_legend=col==0,initialization_hold_s=offset)
            ax.set_xlabel("")
            ax.set_ylabel(ax.get_ylabel(),fontsize=8,labelpad=1)
            ax.tick_params(labelsize=8,length=2.2,pad=2.)
            ystep=((.2,.1,.2),(.2,.1,.1))[row][col]
            low,high=ax.get_ylim()
            ax.set_ylim(np.floor(low/ystep)*ystep,np.ceil(high/ystep)*ystep)
            ax.yaxis.set_major_locator(MultipleLocator(ystep))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            handles[0].set_linewidth(.9)
            for line in handles[1:]:line.set_linewidth(.9)
            if col==0:
                legend=ax.get_legend()
                for text in legend.get_texts():text.set_fontsize(6.6)
            series=[]
            command=handles[0].get_path().vertices
            series.append(dict(name="vel cmd",color="#222222",dashed=True,x=command[:,0].tolist(),y=command[:,1].tolist()))
            for method,line in zip(COLORS,handles[1:]):
                series.append(dict(name=method,color=COLORS[method],dashed=False,x=np.asarray(line.get_xdata()).tolist(),y=np.asarray(line.get_ydata()).tolist()))
            bands=[]
            for method,collection in zip(COLORS,ax.collections):
                bands.append(dict(name=method,color=COLORS[method],alpha=.18,
                                  vertices=collection.get_paths()[0].vertices.tolist()))
            panel=dict(engine=previous["simulation_engine"],axis=axis,show_legend=col==0,
                       plot_rect_mm=[left,top,width,height],xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),
                       xticks=ax.get_xticks().tolist(),yticks=ax.get_yticks().tolist(),ylabel=ax.get_ylabel(),
                       mean_window_s=window,initialization_display_hold_s=offset,series=series,bands=bands)
            panels.append(panel)
            # Raw and display values are exported together without changing the originals.
            for method in COLORS:
                trial=trials[(axis,method)];summary=display[(axis,method)]
                rows=np.column_stack([trial.t,trial.t+offset,trial.signal,summary["mean"],summary["lower"],summary["upper"]])
                np.savetxt(derived/f"{('isaac','mujoco')[row]}_{axis}_{method}.csv",rows,delimiter=",",
                           header="simulation_time_s,plot_time_s,raw_velocity,display_mean,display_lower,display_upper",comments="",fmt="%.17g")
    fig.canvas.draw()
    files=[]
    for ext in ("pdf","svg","png"):
        target=output/f"motion_ability_composite.{ext}"
        fig.savefig(target,dpi=600,bbox_inches=None,facecolor="white")
        files.append(target)
    plt.close(fig)
    payload=dict(size_mm=SIZE_MM,font="Arial",font_weight="regular",body_font_pt=8,title_font_pt=9,
                 images=images,texts=texts,panels=panels)
    (derived/"composition.json").write_text(json.dumps(payload),encoding="utf-8")
    manifest=dict(reference_pptx=str(args.reference),reference_sha256=sha256(args.reference),reference_slide=5,
                  size_mm=SIZE_MM,font="Arial",source_hashes=sources,
                  plot_data="recomputed from the original twelve raw simulation trials",
                  presentation_mean_window_s=.5,band="one population standard deviation within the centered local window, truncated at command changes",
                  isaac_initialization="0-4 s holds the measured reset state for display only; simulated samples are shifted by +4 s",
                  mujoco_initialization="0-4 s is a measured simulated idle interval; no time shift",
                  raster_content="three original simulator photographs and four original transparent arrow overlays only",
                  outputs={p.name:sha256(p) for p in files},script_sha256=sha256(Path(__file__)))
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    (output/"README.md").write_text("""# Motion ability composite

Layout follows slide 5 of Picture.pptx: three original motion illustrations,
then Isaac Sim and MuJoCo rows. The six plots are freshly drawn from raw NPZ
records. Text is regular Arial: 9 pt column/row headings, 8 pt axes and shared
time label, 6.6 pt compact legends. Figure size is 180 x 130 mm.

The PDF/SVG retain vector curves, axes and text. Only the original simulation
photos and original transparent direction arrows are raster images.

Curves reproduce the existing 0.5 s centered local mean and mean +/- one
population standard deviation. Windows stop at command-stage boundaries.
The shading represents within-run temporal variation, not a confidence interval.
Isaac's 0-4 s interval is the previously declared display-only hold of the
measured reset state. The simulator was not advanced during this display hold.
MuJoCo's 0-4 s interval was actually simulated. Raw data and physical timing
remain intact, and both simulation/display timestamps are exported in data/.

Use motion_ability_composite.pdf for the paper and the PPTX for editing.
The PPTX uses native editable charts with embedded data; the standard-deviation
envelopes are separate editable vector shapes behind the chart. If chart data
are changed manually, regenerate the envelopes using the scripts as well.

Reproduce: python scripts/build_motion_composite.py
Then: python scripts/build_motion_composite_pptx.py
""",encoding="utf-8")
    print(json.dumps(dict(output=str(output),panels=len(panels),source_files=len(sources)),indent=2))


if __name__=="__main__":main()
