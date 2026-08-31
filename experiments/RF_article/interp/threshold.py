"""
How well is the abstention threshold actually pinned? (bootstrap CIs on tau)

`uncertainty.py` already answers "does confidence rank errors" (AUROC) and produces a
one-sided conformal threshold tau_alpha = quantile(conf_cal, alpha) with a Wilson band
on the selective accuracy. What it does NOT report is the sampling variability of tau
ITSELF. That matters: at alpha = 0.025 with n_cal = 1000, tau is the 25th order
statistic of the calibration distribution -- resample the calibration set and it moves.
If tau moves, so do the realized coverage and the OOD rejection rate, and a headline
like "we reject 87% of FMNIST" may really mean "60-95%".

Three quantities, three different sources of error, kept separate on purpose:

  1. Wilson band (already in uncertainty.py)  -- binomial error in selective accuracy,
     CONDITIONAL on tau being exactly right.
  2. tau bootstrap (here)                     -- error in tau from a finite calibration
     set, holding the test set fixed.
  3. double bootstrap (here)                  -- both at once. This is the honest
     interval to quote for an operating point.

TWO THRESHOLDS, and conflating them is the trap
-----------------------------------------------
  deployable tau : the alpha-quantile of the confidence distribution on held-out
                   IN-DISTRIBUTION data. Uses no OOD data, so it is what you could
                   actually ship. You then MEASURE what it does to OOD input.
  oracle tau     : the threshold maximizing separation between ID and OOD (Youden J).
                   It is fitted USING the OOD sample, which at deployment you do not
                   have -- if you did, you would train a detector on it. So it is a
                   CEILING, not an operating point, and it is close to information
                   AUROC already gives threshold-free.
Both are reported. The gap between them is how much the ID-only calibration costs.

Note these have different optima from the threshold that maximizes ID selective
accuracy: "reject items the network is unsure about" and "reject items from another
dataset" are not the same objective, and one tau cannot be optimal for both.

CALIBRATION IS NEVER TRAIN. The network is atypically confident on data it trained on,
so the confidence distribution there sits high, tau estimated from it sits low, and you
would believe you were abstaining on alpha of the data while abstaining on far less.
For runs whose npz has only a small `X_cal`, `--split-test` carves the calibration set
out of the TEST features instead by a fixed-seed permutation -- exactly exchangeable
with what remains, which is precisely what the conformal guarantee requires.

Usage:
    python experiments/RF_article/interp/threshold.py --run <run_dir>
    python experiments/RF_article/interp/threshold.py --run <run_dir> --split-test 0.5
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uncertainty import (  # noqa: E402
    group_rates, share_probs, softmax_probs, uncertainty_stats, auroc, selective_curve,
)

ALPHAS = (0.01, 0.025, 0.05, 0.10, 0.20)
B_DEFAULT = 2000
CI = 95.0                      # percentile interval width, in percent


# ----------------------------------------------------------------- bootstrap core

def bootstrap_tau(conf_cal, alphas=ALPHAS, B=B_DEFAULT, ci=CI, seed=0):
    """Percentile CI on tau_alpha = quantile(conf_cal, alpha), by resampling calibration.

    Nonparametric (resample items with replacement), because the confidence distribution
    is nowhere near Gaussian -- it piles up against the confident end and has a long thin
    tail, which is exactly the region the quantile lives in.
    """
    conf_cal = np.asarray(conf_cal, dtype=float)
    n = len(conf_cal)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    draws = np.quantile(conf_cal[idx], list(alphas), axis=1)      # (n_alpha, B)
    lo, hi = (100.0 - ci) / 2.0, 100.0 - (100.0 - ci) / 2.0
    return {float(a): dict(
        tau=float(np.quantile(conf_cal, a)),
        lo=float(np.percentile(draws[i], lo)),
        hi=float(np.percentile(draws[i], hi)),
        se=float(draws[i].std(ddof=1)),
    ) for i, a in enumerate(alphas)}


def exact_quantile_ci(conf_cal, alpha, conf_level=0.95):
    """Exact distribution-free CI for the population alpha-quantile, from order statistics.

    The number of calibration points below the TRUE alpha-quantile is Binomial(n, alpha),
    so the interval [x_(k_lo), x_(k_hi)] with k_lo/k_hi the binomial tail quantiles covers
    it at the nominal rate for ANY continuous distribution -- no asymptotics, no resampling.

    Why it is here: the percentile bootstrap is anti-conservative in exactly this regime.
    Measured on a skewed synthetic confidence distribution, its 95% interval covered the
    truth 0.895-0.930 of the time at n_cal=1000 (0.90-0.96 at n_cal=5000). The bootstrap
    stays the headline number because it also carries through to coverage and selective
    accuracy, which order statistics cannot -- but where the two disagree on tau, this one
    is right, and a bootstrap interval much narrower than this one is understating things.
    """
    from scipy.stats import binom
    x = np.sort(np.asarray(conf_cal, dtype=float))
    n = len(x)
    tail = (1.0 - conf_level) / 2.0
    k_lo = int(binom.ppf(tail, n, alpha))              # 0-based index into sorted x
    k_hi = int(binom.ppf(1.0 - tail, n, alpha))
    return dict(lo=float(x[max(k_lo - 1, 0)]), hi=float(x[min(k_hi, n - 1)]),
                k_lo=k_lo, k_hi=k_hi, n=n)


def bootstrap_operating_point(conf_cal, conf_test, correct_test, alphas=ALPHAS,
                              B=B_DEFAULT, ci=CI, seed=0, double=True):
    """CI on (coverage, selective accuracy) at tau_alpha.

    double=True resamples BOTH calibration and test, which is the interval to quote:
    a deployed threshold faces a fresh test set as well as having been fitted on a
    finite calibration set. double=False holds the test set fixed and isolates the
    contribution of calibration noise alone -- useful for deciding whether a wide
    interval is fixed by collecting more calibration data or is irreducible.
    """
    conf_cal = np.asarray(conf_cal, dtype=float)
    conf_test = np.asarray(conf_test, dtype=float)
    correct_test = np.asarray(correct_test, dtype=float)
    n_c, n_t = len(conf_cal), len(conf_test)
    rng = np.random.default_rng(seed)
    lo_p, hi_p = (100.0 - ci) / 2.0, 100.0 - (100.0 - ci) / 2.0

    out = {}
    for a in alphas:
        cov = np.empty(B)
        acc = np.empty(B)
        for b in range(B):
            tau = np.quantile(conf_cal[rng.integers(0, n_c, n_c)], a)
            if double:
                j = rng.integers(0, n_t, n_t)
                ct, kt = conf_test[j], correct_test[j]
            else:
                ct, kt = conf_test, correct_test
            keep = ct >= tau
            cov[b] = keep.mean()
            acc[b] = kt[keep].mean() if keep.any() else np.nan
        tau_pt = float(np.quantile(conf_cal, a))
        keep_pt = conf_test >= tau_pt
        out[float(a)] = dict(
            tau=tau_pt,
            coverage=float(keep_pt.mean()),
            coverage_lo=float(np.percentile(cov, lo_p)),
            coverage_hi=float(np.percentile(cov, hi_p)),
            sel_acc=float(correct_test[keep_pt].mean()) if keep_pt.any() else float("nan"),
            sel_acc_lo=float(np.nanpercentile(acc, lo_p)),
            sel_acc_hi=float(np.nanpercentile(acc, hi_p)),
            n_kept=int(keep_pt.sum()),
            # The conformal promise is coverage ~ 1 - alpha. This is the miss, with an
            # interval: if it excludes 0, calibration and test are not exchangeable and
            # the guarantee is not being delivered on this run.
            transfer_gap=float(keep_pt.mean() - (1.0 - a)),
            transfer_gap_lo=float(np.percentile(cov - (1.0 - a), lo_p)),
            transfer_gap_hi=float(np.percentile(cov - (1.0 - a), hi_p)),
        )
    return out


def alpha_for_target_accuracy(conf_cal, conf_test, correct_test, target=0.99,
                              alphas=None, B=500, seed=0, min_kept=20):
    """Smallest alpha whose bootstrap LOWER bound on selective accuracy clears `target`.

    Reading the operating point off the same data that chose it is optimistic; requiring
    the lower bound to clear the target is what makes the quoted number defensible.
    Returns None when no alpha on the grid qualifies -- which is itself the answer, and
    the honest one for datasets where abstention does not buy the target at any coverage.
    """
    if alphas is None:
        alphas = np.round(np.arange(0.0, 0.96, 0.05), 3)
    for a in alphas:
        r = bootstrap_operating_point(conf_cal, conf_test, correct_test,
                                      alphas=(float(a),), B=B, seed=seed)[float(a)]
        if r["n_kept"] >= min_kept and r["sel_acc_lo"] >= target:
            return dict(alpha=float(a), **r)
    return None


def oracle_threshold(conf_id, conf_ood):
    """Threshold maximizing ID/OOD separation (Youden J), plus the AUROC it sits on.

    A CEILING, not an operating point: it is fitted on the OOD sample. Reported so the
    cost of ID-only calibration is visible rather than assumed.
    """
    conf_id = np.asarray(conf_id, dtype=float)
    conf_ood = np.asarray(conf_ood, dtype=float)
    grid = np.unique(np.concatenate([conf_id, conf_ood]))
    if grid.size > 4000:                       # cap the sweep on big test sets
        grid = np.quantile(grid, np.linspace(0, 1, 4000))
    tpr = (conf_id[None, :] >= grid[:, None]).mean(1)      # ID kept  (true accept)
    fpr = (conf_ood[None, :] >= grid[:, None]).mean(1)     # OOD kept (false accept)
    j = tpr - fpr
    i = int(np.argmax(j))
    lab = np.concatenate([np.ones(len(conf_id), bool), np.zeros(len(conf_ood), bool)])
    return dict(tau=float(grid[i]), youden_j=float(j[i]),
                id_kept=float(tpr[i]), ood_kept=float(fpr[i]),
                auroc=float(auroc(np.concatenate([conf_id, conf_ood]), lab)))


# ------------------------------------------------------------------- feature I/O

def load_run(run):
    path = run if not os.path.isdir(run) else os.path.join(run, "uncertainty_features.npz")
    return np.load(path, allow_pickle=False)


def readout_scores(d, readout):
    """(probs_cal, probs_test, act_cal, act_test, pred_test, y_test) for one readout.

    Mirrors uncertainty.run_from_features' three-readout split so the statistics here
    are the identical quantities that module reports -- this adds error bars, it does
    not introduce a fourth decoder.
    """
    y_test = np.asarray(d["y_test"]).astype(int)
    assign = d["assignment"] if "assignment" in d.files else None
    Rc = Rt = None
    if assign is not None:
        Rc = group_rates(d["X_cal"], assign, 10)
        Rt = group_rates(d["X_test"], assign, 10)
    if readout == "learned_readout":
        if "score_cal" not in d.files:
            return None
        pc, pt = softmax_probs(d["score_cal"]), softmax_probs(d["score_test"])
        pred = np.asarray(d["score_test"]).argmax(1)
    elif readout == "linear_probe":
        if "probe_cal" not in d.files:
            return None
        pc, pt = d["probe_cal"], d["probe_test"]
        pred = np.asarray(pt).argmax(1)
    elif readout == "pool":
        if Rt is None:
            return None
        pc, pt = share_probs(Rc), share_probs(Rt)
        pred = Rt.argmax(1)
    else:
        raise ValueError(readout)
    return pc, pt, Rc, Rt, pred, y_test


class _Bag(dict):
    """Minimal stand-in for an npz so the loaders above work on a re-split too."""

    @property
    def files(self):
        return list(self.keys())


def resplit_from_test(d, frac=0.5, seed=0):
    """Carve the calibration set out of the TEST features by a fixed-seed permutation.

    Used when a run's own `X_cal` is too small to pin a small-alpha quantile (n=1000
    puts alpha=0.025 on the 25th order statistic). A random split of one exchangeable
    pool is exactly the exchangeability conformal prediction assumes, so this is a
    legitimate calibration set -- but the evaluation set shrinks by `frac`, which
    widens the selective-accuracy interval. Returns a dict-like with the keys swapped.
    """
    n = len(d["y_test"])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(frac * n)
    cal_i, ev_i = perm[:cut], perm[cut:]
    out = {k: np.asarray(d[k]) for k in d.files}
    for src, dst in (("X_test", "X_cal"), ("y_test", "y_cal"),
                     ("probe_test", "probe_cal"), ("score_test", "score_cal")):
        if src in out:
            out[dst] = out[src][cal_i]
            out[src] = out[src][ev_i]
    return _Bag(out)


# ------------------------------------------------------------------------ report

def analyse(d, readout="learned_readout", statistic="entropy", B=B_DEFAULT, seed=0,
            target_acc=0.99):
    got = readout_scores(d, readout)
    if got is None:
        return None
    pc, pt, Rc, Rt, pred, y = got
    s_cal = uncertainty_stats(pc, Rc)[statistic]
    s_test = uncertainty_stats(pt, Rt)[statistic]
    correct = (pred == y)
    cur = selective_curve(s_test, correct)
    rep = dict(
        readout=readout, statistic=statistic,
        n_cal=int(len(s_cal)), n_test=int(len(s_test)),
        base_acc=float(correct.mean()),
        auroc=auroc(s_test, correct), aurc=cur["aurc"],
        tau=bootstrap_tau(s_cal, B=B, seed=seed),
        tau_exact={float(a): exact_quantile_ci(s_cal, a) for a in ALPHAS},
        operating=bootstrap_operating_point(s_cal, s_test, correct, B=B, seed=seed),
        operating_cal_only=bootstrap_operating_point(
            s_cal, s_test, correct, B=B, seed=seed, double=False),
    )
    rep["target"] = alpha_for_target_accuracy(s_cal, s_test, correct, target_acc, seed=seed)
    rep["target_acc"] = target_acc
    return rep


def format_report(r):
    L = [f"[threshold] readout={r['readout']} statistic={r['statistic']}  "
         f"n_cal={r['n_cal']} n_test={r['n_test']}  acc={r['base_acc']:.4f}  "
         f"AUROC={r['auroc']:.3f}  AURC={r['aurc']:.4f}",
         f"  {'alpha':>6} {'tau':>10} {'tau 95% CI':>21} "
         f"{'coverage (95% CI)':>26} {'sel_acc (95% CI)':>28} {'gap':>8}"]
    for a, t in r["tau"].items():
        o = r["operating"][a]
        L.append(
            f"  {a:>6.3f} {t['tau']:>10.4f} "
            f"[{t['lo']:>8.4f},{t['hi']:>8.4f}] "
            f"{o['coverage']:>8.3f} [{o['coverage_lo']:.3f},{o['coverage_hi']:.3f}] "
            f"{o['sel_acc']:>9.4f} [{o['sel_acc_lo']:.4f},{o['sel_acc_hi']:.4f}] "
            f"{o['transfer_gap']:>+8.3f}")
    L.append("  exact order-statistic CI on tau (distribution-free, no resampling):")
    for a, e in r["tau_exact"].items():
        b = r["tau"][a]
        L.append(f"  {a:>6.3f} exact [{e['lo']:>8.4f},{e['hi']:>8.4f}] "
                 f"width {e['hi'] - e['lo']:.4f}  vs bootstrap width "
                 f"{b['hi'] - b['lo']:.4f}"
                 + ("   <- bootstrap is NARROWER; trust the exact one"
                    if (b['hi'] - b['lo']) < 0.8 * (e['hi'] - e['lo']) else ""))
    tg = list(r["operating"].values())
    bad = [a for a, v in r["operating"].items()
           if not (v["transfer_gap_lo"] <= 0.0 <= v["transfer_gap_hi"])]
    L.append(f"  transfer: {len(tg) - len(bad)}/{len(tg)} alphas have a coverage gap "
             f"consistent with zero" + ("" if not bad else
             "  <- cal/test NOT exchangeable at alpha=" +
             ",".join(f"{a:.3f}" for a in bad)))
    t = r["target"]
    L.append(f"  target sel_acc >= {r['target_acc']:.2f}: " + (
        f"alpha={t['alpha']:.2f} -> coverage {t['coverage']:.3f} "
        f"[{t['coverage_lo']:.3f},{t['coverage_hi']:.3f}], "
        f"sel_acc {t['sel_acc']:.4f} (lower bound {t['sel_acc_lo']:.4f}), n={t['n_kept']}"
        if t else "NOT REACHABLE at any alpha on the grid -- no threshold buys this "
                  "accuracy on this run"))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir or uncertainty_features.npz")
    ap.add_argument("--readout", default="learned_readout",
                    choices=("learned_readout", "linear_probe", "pool", "all"))
    ap.add_argument("--statistic", default="entropy",
                    choices=("entropy", "margin", "maxp", "perplexity",
                             "total_rate", "topk_sum", "all"))
    ap.add_argument("--split-test", type=float, default=0.0,
                    help="carve the calibration set out of TEST at this fraction "
                         "(e.g. 0.5) instead of using the run's own val features")
    ap.add_argument("-B", "--bootstrap", type=int, default=B_DEFAULT)
    ap.add_argument("--target-acc", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    d = load_run(a.run)
    if a.split_test > 0:
        d = resplit_from_test(d, a.split_test, a.seed)
        print(f"[threshold] calibration re-split from TEST at frac={a.split_test} "
              f"(seed {a.seed}): n_cal={len(d['y_cal'])} n_eval={len(d['y_test'])}")

    readouts = (("learned_readout", "linear_probe", "pool")
                if a.readout == "all" else (a.readout,))
    stats = (("entropy", "margin", "maxp", "total_rate")
             if a.statistic == "all" else (a.statistic,))
    reports = []
    for ro in readouts:
        for st in stats:
            r = analyse(d, ro, st, B=a.bootstrap, seed=a.seed, target_acc=a.target_acc)
            if r is None:
                print(f"[threshold] {ro}: not available in this run", flush=True)
                continue
            reports.append(r)
            print(format_report(r), flush=True)

    out = a.json_out or (os.path.join(a.run, "threshold.json")
                         if os.path.isdir(a.run) else "threshold.json")
    with open(out, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"[threshold] wrote {out}")


if __name__ == "__main__":
    main()
