"""Generate a run analysis report as a self-contained HTML page.

Usage: python build_run_report.py [RUN_DIR] [OUT_HTML]

Needs `results.json`, `stats/stats.jsonl` and `uncertainty_features.npz` in RUN_DIR.
Written for results/rstdp_thinmargin/run60k_5ep (2026-07-25); the LC / CV5 constants
below are MEASURED for that run and must be re-measured for any other run.
"""
import json, os, math, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
RUN = sys.argv[1] if len(sys.argv) > 1 else _HERE
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    _HERE, os.path.basename(os.path.normpath(RUN)) + "_report.html")

res = json.load(open(os.path.join(RUN, "results.json")))
cfg = res["config"]
traj = res["trajectory"]
netstats = [json.loads(l) for l in open(os.path.join(RUN, "stats", "stats.jsonl")) if '"w_se_mean"' in l]
feat = np.load(os.path.join(RUN, "uncertainty_features.npz"))
yt = feat["y_test"].astype(int)
asg = feat["assignment"]
Xt = feat["X_test"].astype(np.float64)
pred_learn = feat["score_test"].argmax(1)
pred_lin = feat["probe_test"].argmax(1)
pred_pool = np.stack([Xt[:, asg == g].sum(1) for g in range(10)], 1).argmax(1)

# measured probe learning curve — the harness's EXACT probe (L1/saga, C=1, max_iter=400),
# refitted on n images from the final test features, scored on a fixed held-out 2000
LC = [(200, .7325), (500, .8340), (700, .8620), (1000, .8830),
      (2000, .9215), (4000, .9345), (7000, .9485)]
CV5 = (.9438, .0064)   # 5-fold CV (L2, C=0.03), 7200 fit / 1800 eval
LEARNED = float((pred_learn == yt).mean())
LINEAR = float((pred_lin == yt).mean())
POOL = float((pred_pool == yt).mean())

B = [r["batch"] for r in traj]
EP_TICKS = [59, 118, 177, 236, 295]


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------- chart helpers
class Ax:
    """Simple linear axis mapper onto an SVG plot box."""

    def __init__(self, w, h, pad, xlim, ylim):
        self.w, self.h = w, h
        self.l, self.r, self.t, self.b = pad
        self.x0, self.x1 = xlim
        self.y0, self.y1 = ylim

    def X(self, v):
        return self.l + (v - self.x0) / (self.x1 - self.x0) * (self.w - self.l - self.r)

    def Y(self, v):
        return self.t + (1 - (v - self.y0) / (self.y1 - self.y0)) * (self.h - self.t - self.b)

    def frame(self, xticks, yticks, xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}",
              xlabel="", ylabel="", vlines=()):
        p = []
        for v in yticks:
            y = self.Y(v)
            p.append(f'<line class="grid" x1="{self.l:.1f}" x2="{self.w - self.r:.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')
            p.append(f'<text class="tick ty" x="{self.l - 8:.1f}" y="{y + 3.5:.1f}">{yfmt(v)}</text>')
        for v in vlines:
            x = self.X(v)
            p.append(f'<line class="epline" x1="{x:.1f}" x2="{x:.1f}" y1="{self.t:.1f}" y2="{self.h - self.b:.1f}"/>')
        for v in xticks:
            x = self.X(v)
            p.append(f'<text class="tick tx" x="{x:.1f}" y="{self.h - self.b + 16:.1f}">{xfmt(v)}</text>')
        p.append(f'<line class="axis" x1="{self.l:.1f}" x2="{self.w - self.r:.1f}" '
                 f'y1="{self.h - self.b:.1f}" y2="{self.h - self.b:.1f}"/>')
        if xlabel:
            p.append(f'<text class="axlab" x="{(self.l + self.w - self.r) / 2:.1f}" '
                     f'y="{self.h - 4:.1f}">{esc(xlabel)}</text>')
        if ylabel:
            p.append(f'<text class="axlab" transform="translate(13,{(self.t + self.h - self.b) / 2:.1f}) '
                     f'rotate(-90)">{esc(ylabel)}</text>')
        return "".join(p)

    def path(self, xs, ys):
        return "M" + " L".join(f"{self.X(x):.1f},{self.Y(y):.1f}" for x, y in zip(xs, ys))


def line_chart(series, xlim, ylim, xticks, yticks, w=920, h=360, pad=(64, 168, 16, 44),
               xfmt=None, yfmt=None, xlabel="", ylabel="", vlines=(), eplabels=True,
               tipfmt=None, endlabels=True):
    """series: list of dicts {name, xs, ys, slot, dash?}."""
    ax = Ax(w, h, pad, xlim, ylim)
    xfmt = xfmt or (lambda v: f"{v:g}")
    yfmt = yfmt or (lambda v: f"{v:g}")
    tipfmt = tipfmt or (lambda x, y: f"{y:.4g}")
    o = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img">']
    o.append(ax.frame(xticks, yticks, xfmt, yfmt, xlabel, ylabel, vlines))
    if eplabels:
        for i, v in enumerate(vlines):
            o.append(f'<text class="eplab" x="{ax.X(v) - 4:.1f}" y="{ax.t + 11:.1f}">ep{i + 1}</text>')
    # collision-avoiding end labels
    ends = sorted(((s["ys"][-1], s) for s in series), key=lambda t: -t[0])
    placed = []
    for yv, s in ends:
        y = ax.Y(yv)
        while any(abs(y - q) < 15 for q in placed):
            y += 15
        placed.append(y)
        s["_laby"] = y
    for s in series:
        dash = ' stroke-dasharray="5 4"' if s.get("dash") else ""
        o.append(f'<path class="ln s{s["slot"]}" d="{ax.path(s["xs"], s["ys"])}"{dash}/>')
        o.append(f'<circle class="dot s{s["slot"]}" cx="{ax.X(s["xs"][-1]):.1f}" '
                 f'cy="{ax.Y(s["ys"][-1]):.1f}" r="4.5"/>')
        if endlabels:
            o.append(f'<text class="endlab s{s["slot"]}t" x="{w - ax.r + 12:.1f}" '
                     f'y="{s["_laby"] + 4:.1f}">{esc(s["name"])} '
                     f'<tspan class="endval">{yfmt(s["ys"][-1])}</tspan></text>')
        for x, y in zip(s["xs"], s["ys"]):
            o.append(f'<circle class="hit" cx="{ax.X(x):.1f}" cy="{ax.Y(y):.1f}" r="7">'
                     f'<title>{esc(s["name"])} — {tipfmt(x, y)}</title></circle>')
    o.append("</svg>")
    return "".join(o)


def spark(name, xs, ys, slot, note, w=250, h=132):
    ax = Ax(w, h, (40, 12, 22, 26), (min(xs), max(xs)), (min(ys) * .998, max(ys) * 1.002))
    lo, hi = min(ys), max(ys)
    o = [f'<svg class="chart spk" viewBox="0 0 {w} {h}" role="img">']
    o.append(f'<text class="spkttl" x="{ax.l - 28}" y="12">{esc(name)}</text>')
    for v in (lo, hi):
        y = ax.Y(v)
        o.append(f'<line class="grid" x1="{ax.l}" x2="{w - ax.r}" y1="{y:.1f}" y2="{y:.1f}"/>')
        o.append(f'<text class="tick ty" x="{ax.l - 6}" y="{y + 3.5:.1f}">{v:.3g}</text>')
    o.append(f'<path class="ln s{slot}" d="{ax.path(xs, ys)}"/>')
    o.append(f'<circle class="dot s{slot}" cx="{ax.X(xs[-1]):.1f}" cy="{ax.Y(ys[-1]):.1f}" r="4"/>')
    o.append(f'<text class="spknote" x="{ax.l}" y="{h - 6}">{esc(note)}</text>')
    for x, y in zip(xs, ys):
        o.append(f'<circle class="hit" cx="{ax.X(x):.1f}" cy="{ax.Y(y):.1f}" r="6">'
                 f'<title>batch {x} — {y:.4g}</title></circle>')
    o.append("</svg>")
    return "".join(o)


def grouped_bars(cats, series, ylim, yticks, w=920, h=330, ylabel="", xlabel=""):
    pad = (54, 16, 16, 46)
    ax = Ax(w, h, pad, (0, len(cats)), ylim)
    o = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img">']
    o.append(ax.frame([], yticks, yfmt=lambda v: f"{v:.0%}", ylabel=ylabel, xlabel=xlabel))
    slotw = (ax.X(1) - ax.X(0))
    n = len(series)
    bw = (slotw - 14) / n - 2
    base = ax.Y(ylim[0])
    for i, c in enumerate(cats):
        gx = ax.X(i) + 7
        o.append(f'<text class="tick tx" x="{gx + (slotw - 14) / 2:.1f}" y="{h - pad[3] + 16:.1f}">{esc(c)}</text>')
        for j, s in enumerate(series):
            v = s["vals"][i]
            x = gx + j * (bw + 2)
            y = ax.Y(v)
            o.append(f'<rect class="bar s{s["slot"]}f" x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                     f'height="{base - y:.1f}" rx="3"><title>{esc(s["name"])} · class {esc(c)} — '
                     f'{v:.1%}</title></rect>')
    o.append("</svg>")
    return "".join(o)


# Sequential blue ramp, light->dark (dataviz reference ramp, steps 100..700).
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
NSEQ = len(SEQ)
# Dark mode gets its own selected stepping from the same ramp, reversed so that
# "near zero" recedes toward the dark surface rather than glowing off it.
SEQ_D = list(reversed(SEQ))


def seq_css():
    """CSS custom properties for the heatmap ramp: fill + the ink that reads on it."""
    def blk(ramp, flip):
        o = []
        for k in range(NSEQ):
            o.append(f"--q{k}:{ramp[k]}")
            light_step = (k < 7) if not flip else (k >= 6)
            o.append(f"--qt{k}:{'#11181d' if light_step else '#eaf0ef'}")
        return ";".join(o) + ";"
    return blk(SEQ, False), blk(SEQ_D, True)


def heatmap(M, rowlab, collab, w=560):
    n = len(M)
    cell = 42
    lm, tm = 46, 30
    h = tm + n * cell + 26
    lo, hi = float(np.min(M)), float(np.max(M))
    o = [f'<svg class="chart hmap" viewBox="0 0 {w} {h}" role="img">']
    for j, cl in enumerate(collab):
        o.append(f'<text class="tick hm" x="{lm + j * cell + cell / 2:.1f}" y="{tm - 9}">{esc(cl)}</text>')
    for i in range(n):
        o.append(f'<text class="tick hm" x="{lm - 10}" y="{tm + i * cell + cell / 2 + 4:.1f}" '
                 f'text-anchor="end">{esc(rowlab[i])}</text>')
        for j in range(n):
            v = float(M[i][j])
            k = min(NSEQ - 1, max(0, int((v - lo) / (hi - lo + 1e-12) * (NSEQ - 1))))
            ring = ' class="hmc diag"' if i == j else ' class="hmc"'
            o.append(f'<rect{ring} x="{lm + j * cell + 1:.1f}" y="{tm + i * cell + 1:.1f}" '
                     f'width="{cell - 2}" height="{cell - 2}" rx="3" fill="var(--q{k})">'
                     f'<title>true {esc(rowlab[i])} → group {esc(collab[j])}: {v:.2f}</title></rect>')
            o.append(f'<text class="hmv" x="{lm + j * cell + cell / 2:.1f}" '
                     f'y="{tm + i * cell + cell / 2 + 4:.1f}" '
                     f'fill="var(--qt{k})">{v:.2f}</text>')
    o.append("</svg>")
    return "".join(o)


def table(head, rows, cls="", foot=None):
    o = [f'<div class="tw"><table class="{cls}"><thead><tr>']
    for i, hd in enumerate(head):
        o.append(f'<th{" class=num" if i else ""}>{hd}</th>')
    o.append("</tr></thead><tbody>")
    for r in rows:
        o.append("<tr>" + "".join(
            f'<td{" class=num" if i else ""}>{c}</td>' for i, c in enumerate(r)) + "</tr>")
    o.append("</tbody>")
    if foot:
        o.append("<tfoot><tr>" + "".join(
            f'<td{" class=num" if i else ""}>{c}</td>' for i, c in enumerate(foot)) + "</tr></tfoot>")
    o.append("</table></div>")
    return "".join(o)


def legend(items):
    return ('<div class="lg">' + "".join(
        f'<span class="lgi"><i class="sw s{s}f"></i>{esc(n)}</span>' for n, s in items) + "</div>")


def det(summary, body):
    return f'<details class="tv"><summary>{esc(summary)}</summary>{body}</details>'


# ---------------------------------------------------------------------- figures
def g(k):
    return [r[k] for r in traj]


pct = lambda v: f"{v * 100:.0f}%"

FIG_ACC = line_chart(
    [dict(name="learned readout", xs=B, ys=g("readout_learned_acc"), slot=1),
     dict(name="online (train)", xs=B, ys=g("online_acc"), slot=5),
     dict(name="linear probe, refit", xs=B, ys=g("refit_acc"), slot=3),
     dict(name="uniform pool", xs=B, ys=g("pool_acc"), slot=2),
     dict(name="probe frozen at start", xs=B, ys=g("fixed_acc"), slot=4, dash=True)],
    (0, 295), (0.55, 1.0), [0, 59, 118, 177, 236, 295], [.6, .7, .8, .9, 1.0],
    yfmt=lambda v: f"{v * 100:.0f}%", xlabel="checkpoint (batch index; 1 epoch = 59 batches = 60 000 images)",
    ylabel="accuracy", vlines=EP_TICKS[:-1],
    tipfmt=lambda x, y: f"batch {x}: {y:.1%}")

_LCPAD = (64, 30, 40, 46)
_LCLIM = ((math.log10(180), math.log10(11000)), (0.72, 0.985))
FIG_LC = line_chart(
    [dict(name="linear probe", xs=[math.log10(n) for n, _ in LC], ys=[a for _, a in LC], slot=3)],
    *_LCLIM,
    [math.log10(v) for v in (200, 500, 1000, 2000, 5000, 10000)], [.75, .80, .85, .90, .95],
    w=920, h=380, pad=_LCPAD,
    xfmt=lambda v: f"{10 ** v:,.0f}", yfmt=lambda v: f"{v * 100:.0f}%",
    xlabel="images the probe was fitted on  (log scale)", ylabel="test accuracy", eplabels=False,
    endlabels=False, tipfmt=lambda x, y: f"{10 ** x:,.0f} images: {y:.1%}")

_ax = Ax(920, 380, _LCPAD, *_LCLIM)
_ann = []
# two reference lines, labelled INSIDE the plot above each line so they cannot collide
# with each other or with the series
for val, lab, cl, dy in [
        (LEARNED, f"learned readout {LEARNED:.1%} — fitted on 300 000 images", "s1", -7),
        (CV5[0], f"5-fold CV cross-check {CV5[0]:.1%} ±{CV5[1]:.1%} — 7 200 images", "s3", -7)]:
    y = _ax.Y(val)
    _ann.append(f'<line class="ref {cl}" x1="{_ax.l}" x2="{920 - _ax.r}" y1="{y:.1f}" y2="{y:.1f}"/>'
                f'<text class="reflab {cl}t" x="{_ax.l + 8}" y="{y + dy:.1f}">{esc(lab)}</text>')
# the probe's own endpoint, labelled below the curve
_ex, _ey = _ax.X(math.log10(7000)), _ax.Y(LC[-1][1])
_ann.append(f'<text class="reflab s3t" x="{_ex - 8:.1f}" y="{_ey + 17:.1f}" text-anchor="end">'
            f'same probe, 7 000 images: {LC[-1][1]:.1%}</text>')
for n, a in [(700, .8620), (1000, .8830)]:
    _ann.append(f'<circle class="mark" cx="{_ax.X(math.log10(n)):.1f}" cy="{_ax.Y(a):.1f}" r="7"/>')
_ann.append(f'<text class="callout" x="{_ax.X(math.log10(1000)) + 13:.1f}" '
            f'y="{_ax.Y(.883) + 4:.1f}">◀ the sizes the run actually uses '
            f'(700 → refit_acc, 1 000 → test_lin_acc)</text>')
FIG_LC = FIG_LC.replace("</svg>", "".join(_ann) + "</svg>")

FIG_PERP = line_chart(
    [dict(name="uniform pool", xs=B, ys=g("perplexity"), slot=2),
     dict(name="learned readout", xs=B, ys=g("perplexity_readout"), slot=1)],
    (0, 295), (1.0, 10.2), [0, 59, 118, 177, 236, 295], [1, 2, 4, 6, 8, 10],
    h=330, pad=(64, 168, 16, 44), yfmt=lambda v: f"{v:g}",
    xlabel="checkpoint", ylabel="perplexity  (effective no. of classes)", vlines=EP_TICKS[:-1],
    tipfmt=lambda x, y: f"batch {x}: {y:.3f}")
_a2 = Ax(920, 330, (64, 168, 16, 44), (0, 295), (1.0, 10.2))
FIG_PERP = FIG_PERP.replace("</svg>", (
    f'<line class="ref chance" x1="{_a2.l}" x2="{920 - 168}" y1="{_a2.Y(10):.1f}" y2="{_a2.Y(10):.1f}"/>'
    f'<text class="reflab muted" x="{_a2.l + 6}" y="{_a2.Y(10) - 6:.1f}">chance = 10</text>'
    f'<text class="reflab muted" x="{_a2.l + 6}" y="{_a2.Y(1) - 8:.1f}">certain = 1</text></svg>'))

SPARKS = "".join([
    spark("orientation coherence", B, g("orient_coh"), 1, "0.667 → 0.492   −26%"),
    spark("within-group RF diversity", B, g("rf_diversity"), 3, "0.090 → 0.137   +52%"),
    spark("W_se at floor (pruned)", B, g("w_floor_frac"), 4, "2.4% → 14.8%"),
    spark("participation ratio", B, g("pr"), 5, "32.1 → 33.7   +5%"),
    spark("per-neuron η²", B, g("eta2"), 2, "0.169 → 0.161   −5%"),
    spark("grouped η² (val φ)", B, g("val_phi"), 1, "0.178 → 0.275   +55%"),
    spark("readout margin", B, g("margin_readout"), 1, "0.718 → 0.928"),
    spark("probe drift (refit − frozen)", B, g("_drift"), 4, "0.000 → 0.237"),
])

# risk–coverage
COV = [1.0, .95, .90, .80, .50]
RC = {}
for rep in res["uncertainty"]:
    st = next(s for s in rep["statistics"] if s["statistic"] == "entropy")
    RC[rep["readout"]] = [rep["base_acc"]] + [st["acc_at_cov"][f"{c:.2f}"] for c in COV[1:]]
FIG_RC = line_chart(
    [dict(name="learned readout", xs=COV, ys=RC["learned_readout"], slot=1),
     dict(name="linear probe", xs=COV, ys=RC["linear_probe"], slot=3),
     dict(name="uniform pool", xs=COV, ys=RC["pool"], slot=2)],
    (1.02, .48), (0.70, 1.005), [1.0, .9, .8, .7, .6, .5], [.75, .8, .85, .9, .95, 1.0],
    h=330, pad=(64, 168, 16, 46), xfmt=lambda v: f"{v:.0%}", yfmt=lambda v: f"{v * 100:.0f}%",
    xlabel="coverage — fraction of test images the readout answers on (rest abstained)",
    ylabel="accuracy on answered", eplabels=False,
    tipfmt=lambda x, y: f"{x:.0%} coverage: {y:.2%}")

recall = {k: [100 * (p[yt == c] == c).mean() for c in range(10)]
          for k, p in [("learned readout", pred_learn), ("linear probe", pred_lin),
                       ("uniform pool", pred_pool)]}
FIG_CLS = grouped_bars(
    [str(c) for c in range(10)],
    [dict(name="learned readout", vals=[v / 100 for v in recall["learned readout"]], slot=1),
     dict(name="linear probe", vals=[v / 100 for v in recall["linear probe"]], slot=3),
     dict(name="uniform pool", vals=[v / 100 for v in recall["uniform pool"]], slot=2)],
    (0.30, 1.0), [.4, .6, .8, 1.0], ylabel="recall", xlabel="true digit class")

G = np.zeros((10, 10))
for c in range(10):
    m = yt == c
    for gg in range(10):
        G[c, gg] = Xt[np.ix_(m, asg == gg)].mean()
Gn = G / G.max()
FIG_HM = heatmap(Gn, [f"c{i}" for i in range(10)], [f"g{i}" for i in range(10)])

# ------------------------------------------------------------------- tables
per = len(traj) // 5
EPKEYS = [("readout_learned_acc", "learned readout (val)", "{:.3f}"),
          ("online_acc", "online train decisions", "{:.3f}"),
          ("refit_acc", "linear probe, refit", "{:.3f}"),
          ("fixed_acc", "linear probe, frozen at start", "{:.3f}"),
          ("_drift", "representational drift", "{:.3f}"),
          ("pool_acc", "uniform pool", "{:.3f}"),
          ("val_phi", "grouped η² (val φ)", "{:.3f}"),
          ("eta2", "per-neuron η²", "{:.4f}"),
          ("pr", "participation ratio", "{:.2f}"),
          ("n_active", "active neurons", "{:.0f}"),
          ("orient_coh", "orientation coherence", "{:.3f}"),
          ("rf_diversity", "within-group RF diversity", "{:.3f}"),
          ("w_floor_frac", "W_se fraction at floor", "{:.3f}"),
          ("cur_se", "feed-forward drive |SE|", "{:.2f}"),
          ("cur_ie", "inhibitory drive |IE|", "{:.3f}"),
          ("perplexity_readout", "perplexity, learned readout", "{:.3f}"),
          ("perplexity", "perplexity, uniform pool", "{:.3f}"),
          ("margin_readout", "margin, learned readout", "{:.3f}"),
          ("corr_within", "within-group |corr|", "{:.4f}"),
          ("corr_all", "overall |corr|", "{:.4f}"),
          ("dead_frac", "dead fraction", "{:.4f}")]
rows = []
for k, lab, f in EPKEYS:
    v = [float(np.mean([traj[i][k] for i in range(e * per, (e + 1) * per)])) for e in range(5)]
    d = v[-1] - v[0]
    sign = "up" if d > 0 else ("dn" if d < 0 else "fl")
    rows.append([lab] + [f.format(x) for x in v] +
                [f'<span class="d {sign}">{"+" if d > 0 else ""}{f.format(d)}</span>'])
TBL_EP = table(["metric", "epoch 1", "epoch 2", "epoch 3", "epoch 4", "epoch 5", "ep1 → ep5"],
               rows, cls="ep")

NKEYS = [("spikes_exc_mean", "mean exc spike rate", "{:.5f}"),
         ("spikes_inh_mean", "mean inh spike rate", "{:.5f}"),
         ("active_frac_exc", "active fraction, exc", "{:.3f}"),
         ("pop_sparseness", "population sparseness", "{:.3f}"),
         ("ei_ratio_median", "E/I current ratio, median", "{:.3f}"),
         ("ei_ratio_p90", "E/I current ratio, p90", "{:.3f}"),
         ("mean_I_syn_exc", "mean I_syn, exc", "{:.3f}"),
         ("mean_I_syn_inh", "mean I_syn, inh", "{:.3f}"),
         ("mean_mp_exc", "mean membrane potential, exc", "{:.2f}"),
         ("mean_spike_threshold", "mean spike threshold", "{:.2f}"),
         ("mean_adaptation", "mean adaptation", "{:.3f}"),
         ("w_se_mean", "W_se mean  (L1-normalised)", "{:.5f}"),
         ("w_se_std", "W_se std", "{:.4f}"),
         ("w_ei_mean", "W_ei  (E→I, fixed)", "{:.1f}"),
         ("w_ie_mean", "W_ie  (I→E, fixed)", "{:.1f}"),
         ("w_ee_mean", "W_ee  (E→E, disabled)", "{:.1f}"),
         ("rf_mean_cosine", "RF mean pairwise cosine", "{:.4f}"),
         ("rf_participation_ratio", "RF participation ratio", "{:.2f}"),
         ("rf_gini", "RF Gini", "{:.4f}"),
         ("rf_entropy", "RF entropy", "{:.3f}"),
         ("mean_delta_w", "mean Δw  (STDP path)", "{:.1f}"),
         ("mean_ltp", "mean LTP  (STDP path)", "{:.1f}"),
         ("mean_ltd", "mean LTD  (STDP path)", "{:.1f}"),
         ("ltp_ltd_ratio", "LTP/LTD ratio  (STDP path)", "{:.1f}")]
nrows = []
for k, lab, f in NKEYS:
    a, b = float(netstats[0][k]), float(netstats[-1][k])
    d = b - a
    if a == b == 0:
        tag = '<span class="d fl">not recorded</span>'
    elif abs(d) < 1e-9:
        tag = '<span class="d fl">held constant</span>'
    else:
        s = "up" if d > 0 else "dn"
        tag = f'<span class="d {s}">{"+" if d > 0 else ""}{d / abs(a) * 100:.1f}%</span>' if a else \
            f'<span class="d {s}">{d:+.3g}</span>'
    nrows.append([lab, f.format(a), f.format(b), tag])
TBL_NET = table(["network-state metric", "batch 0", "batch 295", "change"], nrows, cls="net")

TBL_FINAL = table(
    ["readout", "what it is", "fitted on", "free params", "test accuracy"],
    [['<b class="s1t">learned readout</b>',
      "dense 1000×10 softmax delta rule, trained online by the reward rule",
      "300 000 images<br><span class=sub>60 000 × 5 epochs</span>", "10 000",
      f'<b class="big">{LEARNED:.2%}</b>'],
     ['<b class="s3t">linear probe</b>',
      "L1 logistic regression on the same 1000 exc rates (external control)",
      "1 000 images<br><span class=sub>the val set</span>", "10 010",
      f'<b class="big">{LINEAR:.2%}</b>'],
     ['<b class="s3t">linear probe, refitted</b>',
      "byte-identical probe, given a comparable amount of data",
      "7 000 images<br><span class=sub>this analysis, not the run</span>", "10 010",
      f'<b class="big">{LC[-1][1]:.2%}</b><span class=sub> · 5-fold CV {CV5[0]:.2%}</span>'],
     ['<b class="s2t">uniform pool</b>',
      "sum rates within each class group, argmax — no fitting at all",
      "—", "0", f'<b class="big">{POOL:.2%}</b>'],
     ['<span class="muted">evaluator PCA+LR</span>',
      "the harness’s own <code>pca_lr</code> scorer — this is <code>test_acc</code> in results.json",
      "train-set features", "—", f'{res["test_acc"]:.2%}']],
    cls="fin")

TBL_RC = table(
    ["coverage", "learned readout", "linear probe", "uniform pool"],
    [[f"{c:.0%}" + (" (no abstention)" if c == 1.0 else "")] +
     [f"{RC[k][i]:.2%}" for k in ("learned_readout", "linear_probe", "pool")]
     for i, c in enumerate(COV)],
    cls="rc",
    foot=["AUROC, entropy ranks errors"] +
         [f'{next(s for s in next(r for r in res["uncertainty"] if r["readout"] == k)["statistics"] if s["statistic"] == "entropy")["auroc"]:.3f}'
          for k in ("learned_readout", "linear_probe", "pool")])

TBL_CLS = table(["digit", "learned readout", "linear probe", "uniform pool", "n"],
                [[str(c), f'{recall["learned readout"][c]:.1f}%',
                  f'{recall["linear probe"][c]:.1f}%', f'{recall["uniform pool"][c]:.1f}%',
                  f"{int((yt == c).sum()):,}"] for c in range(10)], cls="cls")

TBL_LC = table(["images fitted on", "probe test accuracy"],
               [[f"{n:,}" + (' <span class=sub>(≈ refit_acc)</span>' if n == 700 else
                             ' <span class=sub>(≈ test_lin_acc)</span>' if n == 1000 else ""),
                 f"{a:.2%}"] for n, a in LC] +
               [['7 200 <span class=sub>(5-fold CV, L2 C=0.03)</span>',
                 f"{CV5[0]:.2%} ±{CV5[1]:.2%}"],
                ['<b>300 000 <span class=sub>(learned readout)</span></b>', f"<b>{LEARNED:.2%}</b>"]],
               cls="lc")

TBL_HM = table(["true class"] + [f"g{j}" for j in range(10)],
               [[f"c{i}"] + [f"{Gn[i][j]:.2f}" for j in range(10)] for i in range(10)], cls="hm")

CFG_ROWS = [("architecture", f'{cfg["n_exc"]} exc / {cfg["n_inh"]} inh, {cfg["n_groups"]} WTA groups of 100, '
                             f'{cfg["group_layout"]} layout'),
            ("learning rule", f'{cfg["rule"]} (R-STDP), dense readout, readout_lr = {cfg["readout_lr"]}'),
            ("RF prior", f'{cfg["prior"]}, length {cfg["rf_length"]}, thickness {cfg["rf_thickness"]}, '
                         f'centre margin {cfg["center_margin"]}'),
            ("inhibition", f'E→I peak {cfg["peak_ei"]}, I→E peak {cfg["peak_ie"]}, '
                           f'Vogels plasticity {"on" if cfg["use_vogels"] else "off"}'),
            ("recurrence", f'E→E {"on" if cfg["ee"] else "off"}'),
            ("training", f'{cfg["train_all"]:,} images × {cfg["epochs"]} epochs = '
                         f'{cfg["train_all"] * cfg["epochs"]:,} presentations, seed {cfg["seed"]}'),
            ("evaluation", "1 000 val images per checkpoint (60 checkpoints), 9 000 captured test images")]
TBL_CFG = table(["setting", "value"], [[k, v] for k, v in CFG_ROWS], cls="cfg")

both = (pred_learn == yt) & (pred_lin == yt)
lonly = (pred_learn == yt) & ~(pred_lin == yt)
ponly = ~(pred_learn == yt) & (pred_lin == yt)
neither = ~(pred_learn == yt) & ~(pred_lin == yt)

# ------------------------------------------------------------------------ page
SEQ_L_CSS, SEQ_D_CSS = seq_css()
HTML = f"""<title>R-STDP thin-margin run — 60k × 5 epochs</title>
<style>
:root{{
  --paper:#eef1f0; --card:#f9fbfa; --sunk:#e5eae9; --ink:#11181d; --ink2:#4c5a63;
  --ink3:#74838c; --rule:#d2dbd9; --rule2:#c0cbc9;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#4a3aa7;
  --good:#2f7d5c; --warn:#a8720f; --crit:#a33a2e;
  {SEQ_L_CSS}
  --mono:ui-monospace,"Cascadia Mono","Cascadia Code",Consolas,"SF Mono",Menlo,monospace;
  --body:"Segoe UI",system-ui,-apple-system,"Helvetica Neue",Arial,sans-serif;
  color-scheme:light;
}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])){{
  --paper:#11181d; --card:#182126; --sunk:#0d1317; --ink:#eaf0ef; --ink2:#a6b4bb;
  --ink3:#78878f; --rule:#26333a; --rule2:#334148;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#9085e9;
  --good:#4fa87e; --warn:#d3a03c; --crit:#d4685a; color-scheme:dark;
  {SEQ_D_CSS}
}}}}
:root[data-theme="dark"]{{
  --paper:#11181d; --card:#182126; --sunk:#0d1317; --ink:#eaf0ef; --ink2:#a6b4bb;
  --ink3:#78878f; --rule:#26333a; --rule2:#334148;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#9085e9;
  --good:#4fa87e; --warn:#d3a03c; --crit:#d4685a; color-scheme:dark;
  {SEQ_D_CSS}
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:var(--body);
  font-size:16px;line-height:1.62;-webkit-font-smoothing:antialiased}}
.shell{{max-width:1000px;margin:0 auto;padding:0 24px 96px}}
p,li{{max-width:70ch}} p{{margin:0 0 1em}}
h1,h2,h3{{font-family:var(--mono);font-weight:600;letter-spacing:-.02em;text-wrap:balance;margin:0}}
h1{{font-size:clamp(1.85rem,4.4vw,2.9rem);line-height:1.1}}
h2{{font-size:1.32rem;line-height:1.25}} h3{{font-size:1rem;letter-spacing:0}}
.eyebrow{{font-family:var(--mono);font-size:.7rem;letter-spacing:.16em;text-transform:uppercase;
  color:var(--ink3)}}
code{{font-family:var(--mono);font-size:.86em;background:var(--sunk);padding:.1em .34em;border-radius:3px}}
b,strong{{font-weight:600}}
.muted{{color:var(--ink3)}} .sub{{color:var(--ink3);font-size:.82em;font-weight:400}}

/* masthead */
header{{padding:64px 0 34px;border-bottom:2px solid var(--ink)}}
header .meta{{display:flex;flex-wrap:wrap;gap:6px 22px;margin-top:20px;font-family:var(--mono);
  font-size:.76rem;color:var(--ink3)}}
.lede{{font-size:1.1rem;color:var(--ink2);margin-top:18px;max-width:62ch}}

/* sections */
section{{padding:52px 0 8px;border-bottom:1px solid var(--rule)}}
section:last-of-type{{border-bottom:0}}
.shead{{display:flex;align-items:baseline;gap:14px;margin-bottom:20px;flex-wrap:wrap}}
.shead h2{{flex:0 1 auto}}

/* answer panel */
.answer{{background:var(--card);border:1px solid var(--rule2);border-radius:6px;
  padding:26px 28px;margin:30px 0 8px}}
.answer h3{{font-size:.78rem;letter-spacing:.14em;text-transform:uppercase;color:var(--ink3);
  margin-bottom:12px}}
.answer p:last-child{{margin-bottom:0}}
.answer.verdict{{border-left:3px solid var(--s1)}}

/* stat tiles */
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(198px,1fr));gap:14px;margin:26px 0 8px}}
.tile{{background:var(--card);border:1px solid var(--rule);border-radius:6px;padding:18px 20px;
  border-top:3px solid var(--tc,var(--rule2))}}
.tile .k{{font-family:var(--mono);font-size:.68rem;letter-spacing:.13em;text-transform:uppercase;
  color:var(--ink3)}}
.tile .v{{font-family:var(--mono);font-size:2.05rem;font-weight:600;letter-spacing:-.03em;
  font-variant-numeric:tabular-nums;margin:6px 0 2px;line-height:1}}
.tile .n{{font-size:.82rem;color:var(--ink2);line-height:1.45}}

/* figures */
figure{{margin:26px 0 8px}}
figcaption{{font-size:.84rem;color:var(--ink2);margin-top:10px;max-width:80ch}}
figcaption b{{color:var(--ink)}}
.fbox{{background:var(--card);border:1px solid var(--rule);border-radius:6px;padding:16px 14px 8px;
  overflow-x:auto}}
.chart{{display:block;width:100%;min-width:540px;height:auto;overflow:visible}}
.chart.spk{{min-width:0}}
.chart.hmap{{max-width:600px;margin:0 auto;min-width:480px}}
.grid{{stroke:var(--rule);stroke-width:1}}
.axis{{stroke:var(--rule2);stroke-width:1}}
.epline{{stroke:var(--rule2);stroke-width:1;stroke-dasharray:2 4}}
.tick{{font-family:var(--mono);font-size:10.5px;fill:var(--ink3);font-variant-numeric:tabular-nums}}
.ty{{text-anchor:end}} .tx{{text-anchor:middle}} .hm{{text-anchor:middle;font-size:11px}}
.axlab{{font-family:var(--mono);font-size:10.5px;fill:var(--ink3);text-anchor:middle;
  letter-spacing:.04em}}
.eplab{{font-family:var(--mono);font-size:9.5px;fill:var(--ink3);text-anchor:end}}
.ln{{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}}
.dot{{stroke:var(--card);stroke-width:2}}
.endlab{{font-family:var(--mono);font-size:11px;font-weight:600}}
.endval{{font-variant-numeric:tabular-nums;fill:var(--ink2);font-weight:400}}
.hit{{fill:transparent;pointer-events:all}}
.hit:hover{{fill:var(--ink);fill-opacity:.13}}
.ref{{stroke-width:1.5;stroke-dasharray:6 4}} .ref.chance{{stroke:var(--rule2)}}
.reflab{{font-family:var(--mono);font-size:10.5px;font-weight:600}}
.reflab.muted{{fill:var(--ink3);font-weight:400}}
.callout{{font-family:var(--mono);font-size:10.5px;fill:var(--ink2)}}
.mark{{fill:none;stroke:var(--ink);stroke-width:1.5}}
.bar{{stroke:var(--card);stroke-width:1}}
.spkttl{{font-family:var(--mono);font-size:10.5px;fill:var(--ink);font-weight:600}}
.spknote{{font-family:var(--mono);font-size:10px;fill:var(--ink3);font-variant-numeric:tabular-nums}}
.sparks{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px 16px}}
.hmc{{stroke:var(--card);stroke-width:1}} .hmc.diag{{stroke:var(--ink);stroke-width:2}}
.hmv{{font-family:var(--mono);font-size:10px;text-anchor:middle;font-variant-numeric:tabular-nums}}
.s1{{stroke:var(--s1)}} .s2{{stroke:var(--s2)}} .s3{{stroke:var(--s3)}}
.s4{{stroke:var(--s4)}} .s5{{stroke:var(--s5)}}
.s1,.s1 circle{{}} circle.s1,.dot.s1{{fill:var(--s1)}} .dot.s2{{fill:var(--s2)}}
.dot.s3{{fill:var(--s3)}} .dot.s4{{fill:var(--s4)}} .dot.s5{{fill:var(--s5)}}
.s1f{{fill:var(--s1)}} .s2f{{fill:var(--s2)}} .s3f{{fill:var(--s3)}}
.s4f{{fill:var(--s4)}} .s5f{{fill:var(--s5)}}
.s1t{{fill:var(--s1);color:var(--s1)}} .s2t{{fill:var(--s2);color:var(--s2)}}
.s3t{{fill:var(--s3);color:var(--s3)}} .s4t{{fill:var(--s4);color:var(--s4)}}
.s5t{{fill:var(--s5);color:var(--s5)}}

/* legend */
.lg{{display:flex;flex-wrap:wrap;gap:6px 20px;font-family:var(--mono);font-size:.74rem;
  color:var(--ink2);margin:2px 0 14px}}
.lgi{{display:inline-flex;align-items:center;gap:7px}}
.sw{{width:13px;height:3px;border-radius:2px;display:inline-block}}

/* tables */
.tw{{overflow-x:auto;border:1px solid var(--rule);border-radius:6px;background:var(--card);
  margin:22px 0 8px}}
table{{border-collapse:collapse;width:100%;font-size:.85rem}}
th,td{{padding:8px 13px;text-align:left;border-bottom:1px solid var(--rule);vertical-align:top}}
thead th{{font-family:var(--mono);font-size:.68rem;letter-spacing:.09em;text-transform:uppercase;
  color:var(--ink3);font-weight:600;background:var(--sunk);position:sticky;top:0;white-space:nowrap}}
td.num,th.num{{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums;
  white-space:nowrap}}
tbody tr:last-child td{{border-bottom:0}}
tbody tr:hover{{background:var(--sunk)}}
tfoot td{{font-family:var(--mono);font-size:.78rem;background:var(--sunk);color:var(--ink2);
  border-top:2px solid var(--rule2)}}
.fin td:first-child{{white-space:nowrap}} .fin .big{{font-family:var(--mono);font-size:1.15rem}}
.ep td:first-child,.net td:first-child{{white-space:nowrap}}
.d{{font-family:var(--mono);font-size:.82rem;font-weight:600}}
.d.up{{color:var(--good)}} .d.dn{{color:var(--crit)}} .d.fl{{color:var(--ink3);font-weight:400}}
.net .d.up,.net .d.dn{{color:var(--ink2)}}

/* details / table view */
.tv{{margin:14px 0 8px}}
.tv summary{{font-family:var(--mono);font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;
  color:var(--ink3);cursor:pointer;padding:6px 0}}
.tv summary:hover{{color:var(--ink)}}
.tv[open] summary{{color:var(--ink)}}
:focus-visible{{outline:2px solid var(--s1);outline-offset:3px}}

/* rf compare */
.rfrow{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:24px 0 8px}}
.rfcell{{background:var(--card);border:1px solid var(--rule);border-radius:6px;padding:14px}}
.rfcell .k{{font-family:var(--mono);font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink3);margin-bottom:10px}}
.rfcell p{{font-size:.83rem;color:var(--ink2);margin:10px 0 0;max-width:none}}
.rfart{{aspect-ratio:1;border-radius:4px;background:var(--sunk);display:flex;align-items:center;
  justify-content:center;font-family:var(--mono);font-size:.72rem;color:var(--ink3);text-align:center;
  padding:16px;line-height:1.5}}

/* checklist */
ul.chk{{list-style:none;padding:0;margin:18px 0}}
ul.chk li{{padding-left:30px;position:relative;margin-bottom:11px;max-width:74ch}}
ul.chk li::before{{position:absolute;left:0;top:0;font-family:var(--mono);font-weight:600;
  font-size:.9rem}}
ul.chk li.no::before{{content:"✗";color:var(--good)}}
ul.chk li.yes::before{{content:"!";color:var(--warn)}}
footer{{padding:44px 0 0;border-top:1px solid var(--rule);font-family:var(--mono);font-size:.74rem;
  color:var(--ink3)}}
@media (max-width:620px){{ .rfrow{{grid-template-columns:1fr}} .shell{{padding:0 16px 64px}} }}
@media (prefers-reduced-motion:reduce){{*{{transition:none!important;animation:none!important}}}}
</style>

<div class="shell">
<header>
  <div class="eyebrow">results/rstdp_thinmargin/run60k_5ep · seed 0 · 2026-07-24</div>
  <h1>Reward-modulated STDP,<br>thin-margin oriented prior</h1>
  <p class="lede">Five epochs over MNIST, 300 000 presentations. The headline number is
  <b>{LEARNED:.2%}</b> on 9 000 held-out test images. The gap to the linear probe is not a
  property of the representation — it is the probe running out of training data.</p>
  <div class="meta">
    <span>1000 exc / 1000 inh</span><span>10 WTA groups</span>
    <span>oriented prior, thin RFs</span><span>dense plastic readout</span>
    <span>60 checkpoints</span>
  </div>
</header>

<section>
  <div class="shead"><span class="eyebrow">01</span><h2>Final accuracy</h2></div>
  <p>Four different decoders were scored on the same population code, and they disagree by
  twenty points. Which number you quote depends entirely on which decoder you mean, so here
  they all are.</p>
  <div class="tiles">
    <div class="tile" style="--tc:var(--s1)"><div class="k">learned readout</div>
      <div class="v s1t">{LEARNED:.1%}</div>
      <div class="n">the network’s own readout, trained by the reward rule. <b>This is the run’s
      accuracy.</b></div></div>
    <div class="tile" style="--tc:var(--s3)"><div class="k">linear probe</div>
      <div class="v s3t">{LINEAR:.1%}</div>
      <div class="n">external control, fitted on 1 000 images — <b>data-starved</b>, not a
      ceiling.</div></div>
    <div class="tile" style="--tc:var(--s3)"><div class="k">probe, given more data</div>
      <div class="v s3t">{LC[-1][1]:.1%}</div>
      <div class="n">identical probe, 7 000 fitting images. The gap closes to
      {(LEARNED - LC[-1][1]) * 100:.1f} pts.</div></div>
    <div class="tile" style="--tc:var(--s2)"><div class="k">uniform pool</div>
      <div class="v s2t">{POOL:.1%}</div>
      <div class="n">group-sum argmax, no fitting. <b>Here</b> is where the architecture actually
      loses.</div></div>
  </div>
  {TBL_FINAL}
  <p class="sub" style="max-width:80ch">Note that <code>test_acc</code> in
  <code>results.json</code> is <b>{res['test_acc']:.2%}</b> — that is the harness’s own
  <code>pca_lr</code> evaluator, a fifth decoder again, and not the learned readout. The learned
  readout number lives under <code>uncertainty[1].base_acc</code>. Worth renaming before this
  goes in the article.</p>
</section>

<section>
  <div class="shead"><span class="eyebrow">02</span><h2>Why the learned readout beats the linear probe</h2></div>
  <div class="answer verdict">
    <h3>The short answer</h3>
    <p>Both decoders are linear, both have ~10 000 free parameters, and both see the same 1 000
    excitatory rates. The only material difference is how much labelled data each one was fitted
    on: the learned readout got <b>300 000</b> images, the probe got <b>1 000</b>. With 1 000
    features and 1 000 samples the probe is in the <i>p ≈ n</i> regime and is simply
    underfitted.</p>
    <p>The learning curve below settles it. Refit the <i>byte-identical</i> probe — same L1/saga
    logistic regression, same settings — on progressively more data and it walks straight up
    through the reported 88.1% to <b>{LC[-1][1]:.1%}</b>, within
    {(LEARNED - LC[-1][1]) * 100:.1f} points of the learned readout while still fitted on 43× less
    data. <b>The E→I weight of 50 is not implicated.</b></p>
  </div>
  <figure>
    <div class="fbox">{FIG_LC}</div>
    <figcaption><b>Probe accuracy is a function of probe training-set size, not of the
    representation.</b> Same frozen final network, same 1 000 excitatory rates, same
    <code>fit_clf</code> from <code>interp_harness.py</code>; only the number of fitting images
    changes. The two circled points are the sizes the harness actually uses — 700 images for the
    per-checkpoint <code>refit_acc</code>, 1 000 for the final
    <code>test_lin_acc</code>.</figcaption>
  </figure>
  {det("Table view — probe learning curve", TBL_LC)}
  <h3 style="margin-top:34px">Corroborating evidence</h3>
  <p>If the two decoders were reading <i>different information</i>, each would win on its own
  subset of images. They do not — the learned readout almost strictly dominates:</p>
  <div class="tiles">
    <div class="tile"><div class="k">both correct</div><div class="v">{both.mean():.1%}</div>
      <div class="n">the two agree on the easy mass</div></div>
    <div class="tile" style="--tc:var(--s1)"><div class="k">learned only</div>
      <div class="v s1t">{lonly.mean():.1%}</div>
      <div class="n">images the probe misses and the readout gets</div></div>
    <div class="tile" style="--tc:var(--s3)"><div class="k">probe only</div>
      <div class="v s3t">{ponly.mean():.1%}</div>
      <div class="n">the reverse — <b>6.7× smaller</b></div></div>
    <div class="tile"><div class="k">both wrong</div><div class="v">{neither.mean():.1%}</div>
      <div class="n">genuinely hard images</div></div>
  </div>
  <p>A 6.7:1 asymmetry is the signature of one decoder being <i>weaker</i> on the same
  information, not of two decoders reading different codes. Regularisation is a second-order
  effect by comparison: sweeping the penalty on the 1 000-image fit spans only 86.4%
  (unpenalised) to 88.7% (best L2, C = 0.03), against the 88.1% the harness already gets — so at
  most six tenths of a point is available from tuning, versus the seven points that data volume
  buys.</p>
</section>

<section>
  <div class="shead"><span class="eyebrow">03</span><h2>Was it the E→I weight of 50?</h2></div>
  <p>Short answer: no — and the network stats say so from several directions at once. Strong
  non-specific inhibition <i>would</i> show up as runaway or collapsed firing, dead units, or a
  population code that no decoder can read. None of that happened.</p>
  <ul class="chk">
    <li class="no"><b>No rate pathology.</b> Mean excitatory spike rate is flat across all five
    epochs (0.00219 → 0.00223, +1.7%), and the inhibitory population tracks it at the same rate.
    Nothing is being silenced or driven to saturation.</li>
    <li class="no"><b>No dead population.</b> Active excitatory fraction rose 0.950 → 0.971; the
    dead fraction fell to 2.3%; <code>silent_frac</code> is exactly 0.0 — not one test image
    produced a silent network.</li>
    <li class="no"><b>Inhibition is not overwhelming excitation.</b> Median E/I current ratio is
    0.53 and p90 is 2.07. Inhibition delivers roughly half the median excitatory current — firm,
    not crushing.</li>
    <li class="no"><b>The information is there.</b> A linear decoder with enough data reaches
    {LC[-1][1]:.1%}. Whatever the inhibition is doing, it is not destroying class information in
    the population code.</li>
    <li class="yes"><b>Where it plausibly <i>does</i> cost you: the pooled readout.</b> Strong
    global inhibition makes the WTA competition largely non-specific, so each group’s
    <i>mean</i> rate is only weakly tuned to its class. Every group does peak on its own class —
    10/10 on the diagonal below — but the diagonal-to-off-diagonal ratio is only <b>1.56</b>, and
    within-group response correlation (0.095) barely exceeds across-group (0.089). Uniform
    pooling averages that thin margin away, which is why the pool sits at {POOL:.1%} while a
    decoder that can weight individual neurons and use negative cross-group evidence reaches
    {LEARNED:.1%}.</li>
  </ul>
  <figure>
    <div class="fbox">{FIG_HM}</div>
    <figcaption><b>Group mean firing rate by true class</b>, normalised to the global maximum;
    outlined cells are the diagonal. The class structure the E→I inhibition produced is real but
    thin — c0 peaks at 1.00 on its own group, yet g5 still reads 0.80. c8 is the worst case
    (0.84 own group vs 0.78 for g1), and it is exactly the class the pooled readout fails on
    (40.1% recall).</figcaption>
  </figure>
  {det("Table view — group × class rate matrix", TBL_HM)}
  <div class="answer">
    <h3>If you want to test the E→I=50 hypothesis properly</h3>
    <p>The metric to move is <b>pool accuracy</b> and the diagonal ratio above — not the
    learned-vs-linear gap, which is a probe artefact. Sweep <code>peak_ei</code> and watch
    whether group tuning sharpens. Note also that <code>use_vogels</code> is off, so inhibition
    is entirely static in this run: nothing about the E/I balance was learned.</p>
  </div>
</section>

<section>
  <div class="shead"><span class="eyebrow">04</span><h2>Accuracy over training</h2></div>
  {legend([("learned readout", 1), ("online train decisions", 5), ("linear probe, refit", 3),
           ("uniform pool", 2), ("probe frozen at start", 4)])}
  <figure>
    <div class="fbox">{FIG_ACC}</div>
    <figcaption><b>Everything that was going to happen happened in the first half-epoch.</b> The
    learned readout jumps 77.8% → 89.2% within ten checkpoints, then gains only ~5 more points
    across the remaining four and a half epochs. Epoch-mean progression: 89.2 → 92.7 → 93.4 →
    94.0 → 94.2%. The dashed yellow line is the one series moving in earnest — see the next
    section.</figcaption>
  </figure>
  <p>Two things are worth pulling out of that plot. First, the <b>refit probe is flat</b> —
  83.5% in epoch 1, 83.2% in epoch 5. An external linear decoder, refitted from scratch at every
  checkpoint on 700 fresh images, finds the representation no more decodable at the end of
  training than at the start. All of the accuracy gain across five epochs belongs to the readout
  learning to read a code that was already roughly as good as it would get.</p>
  <p>Second, <code>online_acc</code> (95.5% by epoch 5) sits <i>above</i> the held-out learned
  readout (94.2%) — expected, since those are the rule’s own training-time decisions on images it
  is being rewarded on. The 1.3-point gap is a reasonable generalisation gap, not a red flag.</p>
</section>

<section>
  <div class="shead"><span class="eyebrow">05</span><h2>How the representation changed</h2></div>
  <p>This is the most interesting part of the run, and the accuracy curves hide it. The
  representation kept changing for all five epochs even though decodability did not improve.</p>
  <div class="rfrow">
    <div class="rfcell"><div class="k">batch 0 — the prior</div>
      <div class="rfart">weights/rf_first.png<br><span class="sub">clean oriented bars</span></div>
      <p>Every receptive field is a smooth elongated Gaussian blob at some orientation — the
      thin-margin oriented prior, exactly as initialised. Orientation coherence 0.667.</p></div>
    <div class="rfcell"><div class="k">batch 295 — after 300k images</div>
      <div class="rfart">weights/rf_last.png<br><span class="sub">curved stroke fragments</span></div>
      <p>The bars have become <b>curved stroke fragments</b> — arcs, hooks, C-shapes, partial
      loops. Orientation coherence 0.492. The prior was not erased; it was bent into digit
      strokes.</p></div>
  </div>
  <p>That distinction matters for the article. In the earlier trace-STDP harness the oriented
  prior decayed toward an initialisation-independent attractor — the structure was destroyed. Here
  the numbers describe something different: orientation coherence falls 26% while within-group RF
  diversity <i>rises</i> 52% and 14.8% of feed-forward weights get pruned to the floor. Weights
  are being reallocated, not washed out. <code>w_se_mean</code> is held exactly constant by L1
  normalisation (0.33017 at both ends, to 14 decimal places), so all of this happens inside a
  fixed synaptic budget.</p>
  <figure>
    <div class="fbox"><div class="sparks">{SPARKS}</div></div>
    <figcaption>Each panel is one metric across the 60 checkpoints, on its own scale — no shared
    axis is implied. The four in the top row are still moving at batch 295; nothing has
    converged.</figcaption>
  </figure>
  <div class="answer">
    <h3>The drift result</h3>
    <p>The clearest signal in the whole run: a linear probe <b>frozen</b> at checkpoint 0 decays
    from 79.7% to 59.7% while a probe <b>refitted</b> at each checkpoint holds flat at ~83%. The
    gap — 0.000 → 0.237, and still widening at the end — is representational drift. The code stays
    equally decodable while continuously changing which neurons carry what.</p>
    <p>This is a real finding and it is monotone across all five epochs, with no epoch-boundary
    discontinuity. It also means any downstream decoder you train against this network has to keep
    learning, which is precisely what the reward rule’s plastic readout does — and a second reason
    the frozen-probe control reads low.</p>
  </div>
  <p>Two counter-currents are worth naming honestly. <b>Per-neuron η² falls</b> (0.1692 →
  0.1609, −5%) and mean pairwise RF cosine <i>rises</i> 29% — individual neurons get slightly
  less class-selective and slightly more redundant with each other. But <b>grouped η² (val φ)
  rises 55%</b> (0.178 → 0.275) and participation ratio rises to 33.8. The population becomes a
  better class code while its individual units become marginally worse ones. That is a
  distributed-code signature, and it is the same pattern the pool-vs-learned gap points at.</p>
  {det("Table view — per-epoch means, all trajectory metrics", TBL_EP)}
</section>

<section>
  <div class="shead"><span class="eyebrow">06</span><h2>Perplexity and confidence</h2></div>
  <figure>
    <div class="fbox">{FIG_PERP}</div>
    <figcaption><b>Two perplexities, one axis, opposite stories.</b> The learned readout goes from
    1.854 to 1.154 effective classes — near-deterministic. The uniform pool starts at 9.951 and
    ends at 9.325: it never leaves chance.</figcaption>
  </figure>
  <p>The learned readout’s perplexity fell every epoch (1.352 → 1.210 → 1.176 → 1.159 → 1.150)
  and its mean decision margin rose 0.718 → 0.928. It is not just getting more accurate, it is
  getting <i>calibratedly</i> confident: on the final test set mean entropy is 0.063 on images it
  gets right versus 0.601 on images it gets wrong — a 9.6× separation.</p>
  <p>The pooled perplexity staying pinned near 10 is <b>not</b> a representation failure, and it
  is worth not mis-reporting it as one. It is the same thin-margin effect as the heatmap: pooling
  averages a 1.56:1 signal into a near-uniform distribution over ten groups. The pooled figure
  measures the pooling, not the code.</p>
  <figure>
    {legend([("learned readout", 1), ("linear probe", 3), ("uniform pool", 2)])}
    <div class="fbox">{FIG_RC}</div>
    <figcaption><b>Selective prediction — the strongest result in this run.</b> Rank test images
    by the readout’s own entropy and abstain on the least confident. The learned readout reaches
    <b>98.8% at 90% coverage</b> and <b>99.6% at 80%</b>, with entropy-vs-error AUROC 0.941. The
    pooled readout’s entropy is useless for this (AUROC 0.557, barely above the 0.5 coin
    flip).</figcaption>
  </figure>
  {TBL_RC}
  <p>One caveat to carry into the article: only the <i>shape</i> statistics work. Entropy,
  perplexity, margin and max-p all land at AUROC ≈ 0.94 for the learned readout, but the
  <i>scale</i> statistics — <code>total_rate</code> and <code>topk_sum</code> — sit at 0.526,
  indistinguishable from chance. How much the network spiked carries no information about whether
  it was right. The abstention claim has to rest on the readout’s distribution, not on activity
  level.</p>
</section>

<section>
  <div class="shead"><span class="eyebrow">07</span><h2>Per-class breakdown</h2></div>
  {legend([("learned readout", 1), ("linear probe", 3), ("uniform pool", 2)])}
  <figure>
    <div class="fbox">{FIG_CLS}</div>
    <figcaption><b>The three decoders fail in the same places, at different magnitudes.</b>
    Digit 8 is the tell: 40.1% pooled → 76.1% probe → 94.5% learned. Digit 2 follows the same
    ordering (55.8 → 86.3 → 94.6). These are the classes whose group tuning is thinnest in the
    heatmap — consistent with a pooling problem rather than a missing-feature problem.</figcaption>
  </figure>
  {det("Table view — per-class recall", TBL_CLS)}
  <p>The learned readout’s worst class is 9 at 93.6%, and its dominant confusion is 9→4 (22
  images) with 4→9 (23) symmetric — the one genuinely shape-ambiguous pair. Nothing in its
  confusion matrix looks structural; the residual 4.5% error is spread thin. The pooled readout,
  by contrast, is 99.2% on digit 1 and 40.1% on digit 8, which is a broken decoder, not a broken
  representation: digit 1 has the sparsest, most distinctive stroke set, so it survives pooling.</p>
</section>

<section>
  <div class="shead"><span class="eyebrow">08</span><h2>Network state — and two things to fix</h2></div>
  <p>Start-to-end values for every network-state metric logged in <code>stats.jsonl</code>
  (300 records). The picture is a stable, healthy network: no rate drift, no threshold runaway,
  no dead population.</p>
  {TBL_NET}
  <div class="answer">
    <h3>Two instrumentation problems in this run</h3>
    <p><b>1. The plasticity diagnostics are dead.</b> <code>mean_delta_w</code>,
    <code>mean_ltp</code>, <code>mean_ltd</code>, <code>ltp_ltd_ratio</code>,
    <code>mean_x_pre</code> and <code>mean_x_tar_se</code> are all exactly 0.0 at every one of
    the 300 records. Under <code>rule=reward</code> the STDP trace path is not used, so the
    “plasticity balance” panel in <code>stats/stats.png</code> is a flat line at zero and we have
    no direct measurement of update magnitudes. The receptive fields plainly changed and
    <code>w_se_std</code> fell 9.4%, so plasticity did happen — but we are inferring it, not
    measuring it. Worth wiring the reward rule’s own Δw into these fields.</p>
    <p><b>2. Two saved figures are stale.</b> <code>metrics.png</code> in the run root is from
    17:47, mid-run, and only covers the first 36 checkpoints — its y-ranges are misleading as a
    result (per-neuron η² looks like it spans 0.168–0.174 when the full run goes to 0.161). Use
    <code>stats/metrics.png</code> instead. And <code>stats/confusion.png</code> was written at
    01:42, before the run finished, so both test confusion panels still read “pending — end of
    run” even though the matrices are now in <code>results.json</code>. Both are one re-plot
    away.</p>
  </div>
</section>

<section>
  <div class="shead"><span class="eyebrow">09</span><h2>Run configuration</h2></div>
  {TBL_CFG}
</section>

<footer>
  Generated from results.json, stats/stats.jsonl and uncertainty_features.npz.
  Probe learning curve and 5-fold CV computed for this analysis on the saved final features;
  everything else is as logged by the run. Hover any chart mark for exact values; every figure
  has a table view.
</footer>
</div>
"""

# emit pure ASCII: every non-ASCII glyph becomes a numeric entity, so the page renders
# identically no matter what charset the host serves it with.
ASCII = "".join(c if ord(c) < 128 else f"&#{ord(c)};" for c in HTML)
with open(OUT, "w", encoding="ascii") as f:
    f.write(ASCII)
print("wrote", OUT, len(ASCII), "bytes")
