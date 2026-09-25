import csv, os, sys
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_risk_coverage import PALETTES  # noqa: E402

OUT = os.path.join("results", "interp", "model")
# Rose and slate are this figure's own duskier variants of the project pair; sage is
# taken from the shared palette so the frozen reference reads as the SAME condition
# here as in the phase-2 figure, where frozen is also sage.
ROSE, SLATE, INK, RULE = "#B08A93", "#5C6B7C", "#26292C", "#B9B4AF"
SAGE = PALETTES["project"][2]
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
D=defaultdict(list); O=defaultdict(list)
for r in read(os.path.join(OUT,"prediction_draws.csv")):
    D[(int(r["phase"]),r["response"],r["dataset"],r["group"])].append(float(r["value"]))
for r in read(os.path.join(OUT,"model_data.csv")):
    O[(int(r["phase"]),r["response"],r["dataset"],r["cond"])].append(float(r["acc"]))

# Greyscale for print (09-17 feedback): oriented = white box + open circles, random =
# grey box + open triangles, frozen = black dashed reference; all named in a legend.
STYLE={"oriented":("white","",  "o"),"random":("white","","o")}
def box(ax,x,vals,kind,w=.50):
    fc,hatch,_=STYLE[kind]
    b=ax.boxplot([vals],positions=[x],widths=w,patch_artist=True,whis=(2.5,97.5),
                 showfliers=False,zorder=3)
    for p in b["boxes"]:   p.set(facecolor=fc,hatch=hatch,edgecolor="black",linewidth=1.0)
    for k in ("whiskers","caps"):
        for p in b[k]:     p.set(color="black",linewidth=.9)
    for p in b["medians"]: p.set(color="black",linewidth=2.0)

def glyph(ax,x,y,c,label,fs,hw=.055,bh=.030,wh=.058):
    kw=dict(transform=ax.transAxes,clip_on=False,zorder=6)
    ax.plot([x,x],[y-wh,y+wh],color=c,lw=.9,**kw)
    ax.plot([x-hw*.5,x+hw*.5],[y+wh]*2,color=c,lw=.9,**kw)
    ax.plot([x-hw*.5,x+hw*.5],[y-wh]*2,color=c,lw=.9,**kw)
    ax.add_patch(Rectangle((x-hw,y-bh),2*hw,2*bh,facecolor=c,alpha=.30,edgecolor=c,lw=.9,**kw))
    ax.plot([x-hw,x+hw],[y]*2,color=c,lw=2.0,**kw)
    ax.text(x+hw+.07,y,label,transform=ax.transAxes,fontsize=fs,va="center",ha="left",color=INK)

DS=["mnist","kmnist","notmnist","fmnist"]
NICE={"mnist":"MNIST","kmnist":"KMNIST","notmnist":"notMNIST","fmnist":"Fashion","svhn":"SVHN"}
# frozen is the dashed reference line only -- drawing it as a box too would show the same
# cell twice. Colour then carries the one thing the x labels do not say: which weight
# configuration sits underneath. Matches the phase-2 figure (rose = oriented, slate = random).
ORDER=["triplet","base_ori","ee_off","ie_off","vogels","base_rnd"]
LBL={"triplet":"triplet","base_ori":"trace","ee_off":"no E–E","ie_off":"no I–E",
     "vogels":"+ Vogels","base_rnd":"random"}
COL={c:("random" if c=="base_rnd" else "oriented") for c in ORDER}

fig,axes=plt.subplots(1,4,figsize=(15.5,4.3))
flat=axes.ravel()
for j,(ax,ds) in enumerate(zip(flat,DS)):
    fz=np.mean(D[(1,"probe",ds,"frozen")])
    for i,cd in enumerate(ORDER):
        v=D[(1,"probe",ds,cd)]
        if not v: continue
        box(ax,i,v,COL[cd])
        o=O[(1,"probe",ds,cd)]
        ax.scatter(np.full(len(o),i),o,s=16,marker=STYLE[COL[cd]][2],facecolors="white",
                   edgecolors="black",linewidths=.8,zorder=5)
    # Where the frozen value lands on a y-tick (KMNIST 0.7199 against the 0.72 tick,
    # Fashion 0.7799 against 0.78) the sage dashes and the grey gridline sit two pixels
    # apart and blur into one grey rule at print size. A white casing under the reference
    # clears the gridline locally; both sit below the boxes, which draw over them.
    ax.axhline(fz,color="white",lw=5.0,ls="-",zorder=2.4)
    ax.axhline(fz,color="black",lw=1.4,ls=(0,(4,3)),zorder=2.5)
    # Autoscale puts the reference flush with the top of the axes; pad so it clears.
    lo,hi=ax.get_ylim(); span=max(hi,fz)-min(lo,fz)
    ax.set_ylim(min(lo,fz)-.06*span, max(hi,fz)+.10*span)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([LBL[c] for c in ORDER],rotation=45,ha="right",rotation_mode="anchor",fontsize=BASE-0.5)
    ax.set_xlim(-.7,len(ORDER)-.3)
    ax.tick_params(axis="y",labelsize=BASE-3)
    ax.yaxis.grid(True,color=RULE,lw=.5,alpha=.55); ax.set_axisbelow(True)
    ax.set_title(NICE[ds],fontsize=BASE,pad=6)
    if j==0: ax.set_ylabel("predicted accuracy",fontsize=(BASE-2)*1.2)

# No legend: the x labels name each box, and the dashed reference is labelled once,
# to the right of the last panel.
last=flat[len(DS)-1]; fz_last=np.mean(D[(1,"probe",DS[-1],"frozen")])
last.annotate("frozen weights",xy=(1.0,fz_last),xycoords=("axes fraction","data"),
              xytext=(6,0),textcoords="offset points",ha="left",va="center",
              fontsize=BASE-1,annotation_clip=False)
fig.tight_layout()
for ext in ("png","pdf"):   # pdf is what the manuscript includes; png for quick viewing
    fig.savefig(os.path.join(OUT,f"fig_phase1_predicted.{ext}"),bbox_inches="tight")
print("ok")
