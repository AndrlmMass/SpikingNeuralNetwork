"""Phase 1, averaged over datasets: one box per single change from Base.

The per-dataset companion is plot_phase1_predicted.py. Here each box is the GLMM's
predicted accuracy for that configuration, averaged over the four datasets with equal
weight (joint fixed-effect draws, written by phase1_contrasts.R). The circles are the
observed counterpart: per seed, the accuracy averaged over the same four datasets. Base is
the dashed line, as in the per-dataset figure.

Usage (from the repo root, after Rscript results/interp/model/phase1_contrasts.R):
    python experiments/RF_article/interp/plot_phase1_average.py
"""
import csv, os
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join("results", "interp", "model")
FIG = os.path.join("results", "figures")
INK, RULE = "#26292C", "#B9B4AF"
BASE = 16.5
plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":BASE,"axes.edgecolor":INK,"axes.linewidth":.8,
    "axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,"text.color":INK,
    "xtick.direction":"out","ytick.direction":"out","xtick.major.size":3.5,
    "ytick.major.size":3.5,"xtick.major.width":.8,"ytick.major.width":.8,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":200,"savefig.dpi":400,
})

def read(p):
    with open(p,newline="") as f: return list(csv.DictReader(f))

REF="base_ori"
DS=["mnist","kmnist","notmnist","fmnist"]
ORDER=["triplet","vogels","frozen","ee_off","ie_off","base_rnd"]
LBL={"triplet":"Triplet","vogels":"+iSTDP","frozen":"Frozen","ee_off":"−Recurrence",
     "ie_off":"−Inhibition","base_rnd":"Random"}

D=defaultdict(list)
for r in read(os.path.join(OUT,"phase1_average_draws.csv")):
    D[r["cond"]].append(float(r["acc"]))
A={}
for r in read(os.path.join(OUT,"model_data.csv")):
    if r["phase"]=="1" and r["response"]=="probe":
        A[(r["dataset"],r["cond"],r["seed"])]=float(r["acc"])
SEEDS=sorted({s for (_,_,s) in A}, key=int)
def observed(cd):
    return [np.mean([A[(ds,cd,s)] for ds in DS]) for s in SEEDS]

fig,ax=plt.subplots(figsize=(6.4,4.6))
for i,cd in enumerate(ORDER):
    b=ax.boxplot([D[cd]],positions=[i],widths=.50,patch_artist=True,whis=(2.5,97.5),
                 showfliers=False,zorder=3)
    for p in b["boxes"]:   p.set(facecolor="white",edgecolor="black",linewidth=1.0)
    for k in ("whiskers","caps"):
        for p in b[k]:     p.set(color="black",linewidth=.9)
    for p in b["medians"]: p.set(color="black",linewidth=2.0)
    o=observed(cd)
    ax.scatter(np.full(len(o),i),o,s=16,marker="o",facecolors="white",
               edgecolors="black",linewidths=.8,zorder=5)
# Base is the reference line only, as in the per-dataset figure; the white casing clears
# a y-gridline it may land on.
fz=np.mean(D[REF])
ax.axhline(fz,color="white",lw=5.0,ls="-",zorder=2.4)
ax.axhline(fz,color="black",lw=1.4,ls=(0,(4,3)),zorder=2.5)
ax.annotate("Base",xy=(1.0,fz),xycoords=("axes fraction","data"),xytext=(6,0),
            textcoords="offset points",ha="left",va="center",fontsize=BASE-1,
            annotation_clip=False)
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels([LBL[c] for c in ORDER],rotation=45,ha="right",rotation_mode="anchor",
                   fontsize=BASE-0.5)
ax.set_xlim(-.7,len(ORDER)-.3)
ax.tick_params(axis="y",labelsize=BASE-3)
ax.yaxis.grid(True,color=RULE,lw=.5,alpha=.55); ax.set_axisbelow(True)
ax.set_ylabel("predicted accuracy",fontsize=(BASE-2)*1.2)
fig.tight_layout()
for ext in ("png","pdf"):   # pdf is what the manuscript includes; png for quick viewing
    fig.savefig(os.path.join(OUT,f"fig_phase1_average.{ext}"),bbox_inches="tight")
    fig.savefig(os.path.join(FIG,f"fig_phase1_average_BW.{ext}"),bbox_inches="tight")
for cd in [REF]+ORDER:
    v=np.asarray(D[cd]); o=observed(cd)
    print(f"{cd:<12} model {v.mean():.4f} [{np.percentile(v,2.5):.4f}, "
          f"{np.percentile(v,97.5):.4f}]   observed mean {np.mean(o):.4f}")
