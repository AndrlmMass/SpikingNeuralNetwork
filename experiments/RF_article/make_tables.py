"""
LaTeX tables for the Results section, generated from the fitted models.

Every number here is read from the beta-binomial fit outputs or the run JSON -- none is
transcribed. That matters because the Results text currently contains at least one range
that matches neither fitted decoder, which is exactly the failure a generated table
prevents: regenerate and the table cannot silently disagree with the model.

Sources:
    results/interp/model/prediction_draws.csv   2000 posterior draws per (cell, dataset),
                                                from fit_model.R -- used for phase-1
                                                contrasts, which need draws rather than
                                                marginal predictions to get an interval
                                                on a DIFFERENCE.
    results/interp/model/contrasts.csv          phase-2 oriented-random, already computed
                                                by contrasts.R with per-dataset dispersion
    <phase2 runs>/results.json                  selective-prediction coverage, per seed

Tables written to results/tables/:
    tab_phase1.tex           mechanism ablations vs frozen, plus the RF-vs-random
                             contrast (one table: the second was a difference of two
                             rows of the first)
    tab_phase2_prior.tex     oriented vs random under R-STDP, both decoders
    tab_selective.tex        coverage attainable at three accuracy targets

Usage:
    python experiments/RF_article/make_tables.py
"""
import collections, csv, glob, json, os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL = os.path.join(REPO, "results", "interp", "model")
PHASE2 = os.path.join(REPO, "results", "interp", "json", "experiments", "RF_article",
                      "interp", "mnist_family_sweep", "results", "run_20260819_112250")
OUT = os.path.join(REPO, "results", "tables")
DS = ["mnist", "fmnist", "kmnist", "notmnist"]
PRETTY = {"mnist": "MNIST", "fmnist": "Fashion", "kmnist": "KMNIST",
          "notmnist": "notMNIST", "svhn": "SVHN"}
COND = {"base_ori": "trace-STDP", "triplet": "triplet-STDP", "vogels": "+ iSTDP (Vogels)",
        "ee_off": "no E$\\to$E recurrence", "ie_off": "no I$\\to$E inhibition",
        "base_rnd": "trace-STDP, random init"}


def load_draws():
    D = collections.defaultdict(list)
    with open(os.path.join(MODEL, "prediction_draws.csv")) as f:
        for x in csv.DictReader(f):
            D[(x["phase"], x["response"], x["dataset"], x["group"])].append(float(x["value"]))
    return {k: np.asarray(v) for k, v in D.items()}


def contrast(D, a, b, ds, phase="1", resp="probe"):
    """Posterior mean, 95% interval, and two-sided tail probability for a - b, in pp.

    Computed on the DRAWS, not by differencing two marginal predictions with their own
    intervals: the two cells share fixed-effect uncertainty, so differencing the intervals
    would overstate the width of the difference.
    """
    ka, kb = (phase, resp, ds, a), (phase, resp, ds, b)
    if ka not in D or kb not in D:
        return None
    d = D[ka] - D[kb]
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return 100 * d.mean(), 100 * np.percentile(d, 2.5), 100 * np.percentile(d, 97.5), p


def stars(p):
    """Significance marker as a math-mode superscript, WITHOUT its own $ delimiters.

    Callers embed it inside an existing $...$; returning "$^{***}$" would close and
    immediately reopen math mode, which typesets as "$-2.19$$^{***}$" and shifts the
    baseline.
    """
    return "^{***}" if p < .001 else "^{**}" if p < .01 else "^{*}" if p < .05 else ""


def fmt(c):
    if c is None:
        return "--"
    m, lo, hi, p = c
    return f"${m:+.2f}{stars(p)}$ & $[{lo:+.2f},\\, {hi:+.2f}]$"


def write(name, body):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"  wrote {p}")


# ------------------------------------------------------- table 1 (phase 1, merged)

# Mechanism ablations: oriented prior throughout, so each row changes exactly one thing
# against the oriented frozen reference. base_rnd is deliberately NOT here -- it changes
# the prior AND the rule at once, so it is not comparable to these rows and appears below
# as the prior contrast instead.
LADDER = ["base_ori", "triplet", "vogels", "ee_off", "ie_off"]


def tab_phase1(D):
    """One phase-1 table: mechanism ablations against frozen, then the prior contrast.

    These were two tables until it became clear the second was a difference of two rows of
    the first -- base_ori-frozen minus base_rnd-frozen reproduces the prior contrast to the
    last reported decimal on all five datasets. Only the interval is new information, since
    the two cells covary and the width of a difference cannot be read off two separate
    intervals. So the contrast keeps a row, and the redundant table goes.
    """
    rows = []
    for cond in LADDER:
        cells = [f"${c[0]:+.2f}{stars(c[3])}$" if (c := contrast(D, cond, "frozen", ds))
                 else "--" for ds in DS]
        rows.append(f"{COND[cond]} & " + " & ".join(cells) + " \\\\")
    prior = []
    for ds in DS:
        c = contrast(D, "base_ori", "base_rnd", ds)
        prior.append("--" if c is None else f"${c[0]:+.2f}{stars(c[3])}$")
    header = " & ".join(PRETTY[d] for d in DS)
    # RAW f-string: in a normal one "\begin" is a backspace and "\toprule" a tab, which
    # silently emits control characters into the .tex instead of LaTeX commands.
    return rf"""\begin{{table}}[t]
\centering
\caption{{Unsupervised model, in percentage points of accuracy on the linear probe.
The upper block compares each plastic configuration against frozen weights, holding the
oriented prior fixed, so every row changes one mechanism. The lower block is the
receptive-field contrast, holding trace-STDP fixed and changing only the initialization;
it is reported separately because its reference is random connectivity rather than frozen
weights. Entries without a marker have intervals spanning zero.}}
\label{{tab:phase1}}
\begin{{tabular}}{{lrrrr}}
\toprule
& {header} \\
\midrule
\multicolumn{{5}}{{l}}{{\emph{{against frozen weights, oriented prior}}}} \\
{chr(10).join(rows)}
\midrule
\multicolumn{{5}}{{l}}{{\emph{{against random connectivity, trace-STDP}}}} \\
oriented receptive fields & {" & ".join(prior)} \\
\bottomrule
\end{{tabular}}

\footnotesize $^{{*}}p<0.05$, $^{{**}}p<0.01$, $^{{***}}p<0.001$.
\end{{table}}
"""


# ---------------------------------------------------------------- table 3

def tab_phase2_prior():
    with open(os.path.join(MODEL, "contrasts.csv")) as f:
        rows = list(csv.DictReader(f))
    by = {(r["response"], r["dataset"]): r for r in rows}
    out = []
    for ds in DS:
        cells = []
        for resp in ("learned", "probe"):
            r = by.get((resp, ds))
            if r is None:
                cells.append("-- & --")
                continue
            m, lo, hi, p = (100 * float(r["est"]), 100 * float(r["lo"]),
                            100 * float(r["hi"]), float(r["p"]))
            cells.append(f"${m:+.2f}{stars(p)}$ & $[{lo:+.2f},\\, {hi:+.2f}]$")
        out.append(f"{PRETTY[ds]} & " + " & ".join(cells) + " \\\\")
    disp = sorted({r["disp"] for r in rows if r["response"] == "learned"})
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Oriented receptive fields against random initialization under R-STDP, by
decoder. The learned readout is the network's own decoder; the linear probe is an
external control fitted on the same features. The two agree in sign on every dataset but
not in magnitude, so any statement of this contrast has to name its decoder. Dispersion
is allowed to vary by dataset for the learned readout ($\\phi \\sim$ dataset); for the probe
the seed variance is not identifiable, so that model omits the seed intercept.}}
\\label{{tab:phase2_prior}}
\\begin{{tabular}}{{lrrrr}}
\\toprule
& \\multicolumn{{2}}{{c}}{{Learned readout}} & \\multicolumn{{2}}{{c}}{{Linear probe}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}}
Dataset & $\\Delta$ (pp) & $95\\%$ CI & $\\Delta$ (pp) & $95\\%$ CI \\\\
\\midrule
{chr(10).join(out)}
\\bottomrule
\\end{{tabular}}

\\footnotesize $^{{*}}p<0.05$, $^{{**}}p<0.01$, $^{{***}}p<0.001$.
\\end{{table}}
"""


# ---------------------------------------------------------------- table 4

def tab_selective():
    cov = collections.defaultdict(list)
    auroc = collections.defaultdict(list)
    base = collections.defaultdict(list)
    for f in glob.glob(os.path.join(PHASE2, "*_oriented_s*", "results.json")):
        d = json.load(open(f))
        ds = d["config"]["dataset"]
        for u in d.get("uncertainty", []):
            if u["readout"] != "learned_readout":
                continue
            base[ds].append(u["base_acc"])
            for s in u["statistics"]:
                if s["statistic"] == "entropy":
                    cov[ds].append(s["cov_at_acc"])
                    auroc[ds].append(s["auroc"])
    rows = []
    for ds in DS:
        if ds not in cov:
            continue
        g = lambda k: np.mean([float(x[k]) for x in cov[ds]])
        rows.append(f"{PRETTY[ds]} & ${np.mean(base[ds]):.3f}$ & ${np.mean(auroc[ds]):.3f}$ & "
                    f"${g('0.95'):.3f}$ & ${g('0.98'):.3f}$ & ${g('0.99'):.3f}$ \\\\")
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Selective prediction with the learned readout, thresholding predictive entropy.
Coverage is the largest fraction of the test set that can be answered while holding
selective accuracy at the stated target; AUROC measures how well entropy separates correct
from incorrect predictions. Values are means over five seeds. Abstention buys a usable
operating point on MNIST alone.}}
\\label{{tab:selective}}
\\begin{{tabular}}{{lrrrrr}}
\\toprule
& & & \\multicolumn{{3}}{{c}}{{Coverage at target accuracy}} \\\\
\\cmidrule(lr){{4-6}}
Dataset & Accuracy & AUROC & $95\\%$ & $98\\%$ & $99\\%$ \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def main():
    D = load_draws()
    write("tab_phase1.tex", tab_phase1(D))
    write("tab_phase2_prior.tex", tab_phase2_prior())
    write("tab_selective.tex", tab_selective())

    # console echo, so the numbers can be checked against the manuscript at a glance
    print("\n--- phase 1: RF - random (trace-STDP) ---")
    for ds in DS:
        c = contrast(D, "base_ori", "base_rnd", ds)
        print(f"  {PRETTY[ds]:<9} {c[0]:+6.2f} pp [{c[1]:+6.2f},{c[2]:+6.2f}] p={c[3]:.4f}")
    print("\n--- phase 1: best plastic - frozen ---")
    for ds in DS:
        best = max(LADDER, key=lambda c: D[("1", "probe", ds, c)].mean())
        c = contrast(D, best, "frozen", ds)
        print(f"  {PRETTY[ds]:<9} {best:<9} {c[0]:+6.2f} pp "
              f"[{c[1]:+6.2f},{c[2]:+6.2f}] p={c[3]:.4f}")


if __name__ == "__main__":
    main()
