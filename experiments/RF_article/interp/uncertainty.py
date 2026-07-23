"""
Does uncertainty rank the network's errors? (go / no-go for selective prediction)

Motivation — Ellingsen, Hubin, Remonato & Saebo (2024), "Outlier Detection in
Bayesian Neural Networks", NMI 4:1-15. Their baseline ENT thresholds the PREDICTIVE
ENTROPY H = -sum_k p_k log p_k of the predictive distribution (NOT cross-entropy:
CE needs the true label, which we do not have at decision time; and perplexity
exp(H) is a monotone rescaling of H, so it gives identical decisions and only
nicer units). Their threshold is a distribution-free conformal p-value against the
statistic's distribution on held-out data (their Eqs. 4-5), with significance alpha
as the single dial. We use only the ONE-SIDED lower tail (their p2): the upper tail
detects Out-Of-Distribution samples, which we do not have -- every input is an
MNIST digit -- while the lower tail detects the In-Between case, i.e. the network
being torn between classes, which is exactly our regime.

BEFORE building any of that apparatus, this module answers the cheap prerequisite
question: does ANY uncertainty statistic actually rank errors? If a statistic
separates correct from incorrect items at AUROC ~ 0.5, then no threshold and no
alpha can buy accuracy by abstaining, and the whole direction is dead. AUROC is
threshold-free, so it settles that in one number per statistic.

Statistics (all per-item, all label-free -- usable at decision time):

    entropy     H(p)                    predictive entropy; the ENT baseline.
    perplexity  exp(H(p))               same ranking as entropy, nicer units.
    margin      p_(1) - p_(2)           top-two gap; their max1 ~ max2 observation.
    maxp        p_(1)                   maximum predicted probability.
    total_rate  sum_c rate_c            SNN analogue of their PRE-ACT statistic.
    topk_sum    sum_{k<K} sorted(rate)  literal transfer of their top-(K-1) sum.

On total_rate / topk_sum: their PRE-ACT works on last-layer PRE-activations, i.e.
before the softmax normalizes the SCALE away. Our readout normalizes the same way
(shares are rates / sum of rates), so the discarded quantity is the total population
spike count -- the structural analogue. Worth measuring even though on MNIST their
own results had plain ENT beating PRE-ACT in 5 of 6 tests.

Conventions:
  * "confidence" always means HIGHER = more likely correct, so entropy is negated.
  * AUROC is of the confidence score discriminating correct (positive) from
    incorrect (negative) items. 0.5 = useless. <0.5 = anti-correlated, which is
    still usable (flip the sign) and is itself a finding.

Standalone use, on the features an interp run dumped:
  python experiments/RF_article/interp/uncertainty.py --features <run_dir>/uncertainty_features.npz
"""
import argparse, json, os
import numpy as np

K_DEFAULT = 10
EPS = 1e-12

# alpha grid for the conformal sweep: fraction of the CALIBRATION distribution we
# are willing to declare too-uncertain-to-answer.
ALPHA_GRID = (0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
ACC_TARGETS = (0.95, 0.98, 0.99, 1.00)
COV_TARGETS = (0.95, 0.90, 0.80, 0.50)
MIN_KEPT = 20          # guard against "coverage 0.2%, accuracy 100%" degeneracies


# ---------------------------------------------------------------- probabilities

def group_rates(X, assignment, n_groups=K_DEFAULT):
    """Per-item mean firing rate of each class group: (n_items, n_groups).

    Mean (not sum) so unequal group sizes do not bias the comparison. This mirrors
    what softmax_readout / spike_share_metrics do in neurosnn, kept local here so
    this diagnostic does not depend on library internals.
    """
    return np.stack(
        [X[:, assignment == g].mean(1) if (assignment == g).any()
         else np.zeros(X.shape[0]) for g in range(n_groups)], axis=1)


def share_probs(rates):
    """L1-normalize rates into a per-item class distribution: (n_items, n_groups).

    NOT a softmax -- an L1 share, so it is scale-invariant and needs no temperature.
    Items that emitted no spikes at all map to uniform (maximum entropy), never NaN;
    `silent_mask` below flags them so they can be accounted for separately.

    NOTE this normalization can only represent evidence FOR a class: it needs
    non-negative mass, so a readout with negative weights (evidence AGAINST a class)
    cannot be expressed here at all -- use `softmax_probs` for those.
    """
    r = np.clip(rates, 0.0, None)
    tot = r.sum(1, keepdims=True)
    return np.where(tot > EPS, r / (tot + EPS), 1.0 / r.shape[1])


def softmax_probs(scores):
    """Softmax of readout scores into a per-item class distribution.

    This is the right normalization for the LEARNED readout, for two reasons:
      1. its weights are sign-free (dense readout: own-class +, competitors -), so
         scores go negative and an L1 share is not even defined on them;
      2. the reward learner trains those weights with a softmax delta rule
         (synapses.py step()), so softmax(scores) is literally the network's own
         probability model, not a post-hoc choice we imposed on it.

    The temperature worry that rules softmax out for RAW rates does not apply here:
    readout_predict max-normalizes the rates per item before the matmul, so scores
    are O(1) rather than ~0.01, and the softmax is not pinned near uniform.
    """
    s = np.asarray(scores, dtype=float)
    e = np.exp(s - s.max(axis=1, keepdims=True))
    return e / (e.sum(axis=1, keepdims=True) + EPS)


def learned_readout_scores(X, rl):
    """Replay the reward learner's own readout on features X -> (n_items, n_classes).

    Mirrors Reward*.readout_predict (synapses.py) but returns the SCORES rather than
    the argmax, so we can look at the whole distribution. Read-only: it touches the
    learner's fitted weights and changes nothing.

    Returns None when the run had no plastic readout (readout_lr == 0), in which
    case the uniform pool IS the network's readout and is already covered.
    """
    if rl is None or getattr(rl, "readout_lr", 0.0) <= 0.0:
        return None
    Xr = X / (X.max(axis=1, keepdims=True) + 1e-9)
    if getattr(rl, "dense_readout", False):
        return Xr @ rl.W_dense
    return (Xr * rl.w_readout) @ rl._A


def silent_mask(rates):
    """Items that produced essentially no spikes anywhere -- uniform p by fiat.

    These sit at maximum entropy and are therefore abstained on FIRST. That may be
    the right behaviour, but if they dominate the abstentions then "entropy detects
    errors" really means "the network went silent", which is a different claim.
    """
    return rates.sum(1) <= EPS


# ------------------------------------------------------------------ statistics

def uncertainty_stats(p, activity=None):
    """All candidate confidence statistics. Returns {name: score}, HIGHER = confident.

    p        -- (n_items, K) predictive distribution. SHAPE statistics come from here,
                so they are readout-specific.
    activity -- (n_items, M) non-negative network activity (group rates). SCALE
                statistics come from here, NOT from p: the total spikes an image
                evokes is a property of the network's response, not of whichever
                decoder we hang off it, so it stays comparable across readouts.
                Passing p itself would be meaningless for any normalized readout --
                a probability vector sums to 1 by construction, making `total_rate`
                a constant. Omit to skip the scale statistics entirely.
    """
    p = np.asarray(p, dtype=float)
    H = -(p * np.log(p + EPS)).sum(1)
    srt = np.sort(p, axis=1)[:, ::-1]
    out = {
        "entropy":    -H,                                   # negated: high = certain
        "perplexity": -np.exp(H),                           # monotone in entropy
        "margin":     srt[:, 0] - srt[:, 1],
        "maxp":       srt[:, 0],
    }
    if activity is not None:
        act = np.clip(np.asarray(activity, dtype=float), 0.0, None)
        srt_a = np.sort(act, axis=1)[:, ::-1]
        out["total_rate"] = act.sum(1)
        out["topk_sum"] = srt_a[:, :max(act.shape[1] - 1, 1)].sum(1)
    return out


# --------------------------------------------------------------------- metrics

def auroc(conf, correct):
    """AUROC of `conf` discriminating correct (positive) from incorrect items.

    Tie-aware (Mann-Whitney U with midranks) -- necessary here because silent items
    all share the identical maximum-entropy score, and naive ranking would score
    those ties arbitrarily.
    """
    conf = np.asarray(conf, dtype=float)
    pos = np.asarray(correct, dtype=bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(conf, kind="mergesort")
    ranks = np.empty(len(conf), dtype=float)
    ranks[order] = np.arange(1, len(conf) + 1)
    s = conf[order]                                   # average ranks within ties
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * ((i + 1) + (j + 1))
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def risk_coverage(conf, correct):
    """Selective accuracy as a function of coverage, answering most-confident first.

    Returns (coverage, sel_acc, aurc). AURC = mean risk (1 - selective accuracy)
    over the sweep; lower is better, and it is the single scalar summary of how
    well the statistic orders errors.
    """
    order = np.argsort(-np.asarray(conf, dtype=float), kind="mergesort")
    c = np.asarray(correct, dtype=float)[order]
    k = np.arange(1, len(c) + 1)
    sel_acc = np.cumsum(c) / k
    return k / len(c), sel_acc, float(np.mean(1.0 - sel_acc))


def coverage_at_accuracy(coverage, sel_acc, target):
    """Largest coverage whose selective accuracy still meets `target`.

    Optimistic by construction (the operating point is read off the same data), so
    it is an upper bound on what the conformal threshold below will deliver. Useful
    as a ceiling: if even this is poor, the calibrated version cannot be better.
    """
    ok = (sel_acc >= target) & (np.arange(1, len(sel_acc) + 1) >= MIN_KEPT)
    return float(coverage[ok].max()) if ok.any() else 0.0


def accuracy_at_coverage(coverage, sel_acc, target):
    """Selective accuracy when answering the `target` fraction of most-confident items."""
    idx = np.searchsorted(coverage, target, side="left")
    return float(sel_acc[min(idx, len(sel_acc) - 1)])


def conformal_sweep(conf_cal, conf_test, correct_test, alphas=ALPHA_GRID):
    """One-sided conformal abstention, Ellingsen et al. Eq. 5 (their p2 tail).

    tau = alpha-quantile of the statistic on CALIBRATION data; abstain on test items
    scoring below it. Calibration is held-out val, NOT train: the network is
    atypically confident on data it trained on, which would bias tau low and make
    coverage look better than it is.

    Note alpha is the abstention rate we ASK for on calibration data; realized test
    coverage differs when the two splits disagree, and that gap is itself the
    honest measure of whether the threshold transfers.
    """
    rows = []
    for a in alphas:
        tau = -np.inf if a <= 0 else float(np.quantile(conf_cal, a))
        keep = conf_test >= tau
        n = int(keep.sum())
        rows.append(dict(
            alpha=float(a), tau=float(tau),
            coverage=float(keep.mean()),
            sel_acc=float(correct_test[keep].mean()) if n else float("nan"),
            n_kept=n,
        ))
    return rows


# ---------------------------------------------------------------------- report

def evaluate_statistic(name, conf_cal, conf_test, correct_test):
    cov, sel, aurc = risk_coverage(conf_test, correct_test)
    return dict(
        statistic=name,
        auroc=auroc(conf_test, correct_test),
        aurc=aurc,
        cov_at_acc={f"{t:.2f}": coverage_at_accuracy(cov, sel, t) for t in ACC_TARGETS},
        acc_at_cov={f"{t:.2f}": accuracy_at_coverage(cov, sel, t) for t in COV_TARGETS},
        conformal=conformal_sweep(conf_cal, conf_test, correct_test),
    )


def uncertainty_report(p_cal, p_test, act_cal, act_test, pred_test, y_test,
                       readout="pool"):
    """Full go/no-go report for one readout.

    p_cal / p_test  -- (n, K) predictive distributions on calibration / test
    act_cal/act_test-- (n, K) class-group rates, for the scale statistics
    pred_test       -- (n_test,) that readout's own predictions on test
    y_test          -- (n_test,) true labels
    """
    correct = (np.asarray(pred_test) == np.asarray(y_test))
    cal_stats = uncertainty_stats(p_cal, act_cal)
    test_stats = uncertainty_stats(p_test, act_test)
    sil = silent_mask(act_test)
    return dict(
        readout=readout,
        n_test=int(len(y_test)),
        base_acc=float(correct.mean()),
        mean_entropy_correct=float(-test_stats["entropy"][correct].mean()),
        mean_entropy_wrong=float(-test_stats["entropy"][~correct].mean())
        if (~correct).any() else float("nan"),
        mean_perplexity=float(-test_stats["perplexity"].mean()),
        # if silent items dominate, high entropy means "network died", not "hard item"
        silent_frac=float(sil.mean()),
        silent_acc=float(correct[sil].mean()) if sil.any() else float("nan"),
        # entropy ties (mostly silent items) bound how finely a threshold can slice
        max_ent_tie_frac=float(np.mean(
            np.isclose(test_stats["entropy"], test_stats["entropy"].min()))),
        statistics=[evaluate_statistic(n, cal_stats[n], test_stats[n], correct)
                    for n in test_stats],
    )


def format_report(rep):
    L = [f"[uncertainty] readout={rep['readout']}  n_test={rep['n_test']}  "
         f"acc={rep['base_acc']:.3f}  silent={rep['silent_frac']:.3f} "
         f"(acc on silent {rep['silent_acc']:.3f})  max-ent ties={rep['max_ent_tie_frac']:.3f}",
         f"  mean entropy: correct {rep['mean_entropy_correct']:.3f} vs wrong "
         f"{rep['mean_entropy_wrong']:.3f}  (ceiling {np.log(10):.3f})  "
         f"perplexity {rep['mean_perplexity']:.2f}",
         f"  {'statistic':<11} {'AUROC':>7} {'AURC':>7} "
         f"{'cov@95':>7} {'cov@99':>7} {'acc@95cov':>10} {'acc@80cov':>10}"]
    for s in sorted(rep["statistics"], key=lambda d: -(d["auroc"] if np.isfinite(d["auroc"]) else 0)):
        L.append(f"  {s['statistic']:<11} {s['auroc']:>7.3f} {s['aurc']:>7.3f} "
                 f"{s['cov_at_acc']['0.95']:>7.3f} {s['cov_at_acc']['0.99']:>7.3f} "
                 f"{s['acc_at_cov']['0.95']:>10.3f} {s['acc_at_cov']['0.80']:>10.3f}")
    best = max(rep["statistics"],
               key=lambda d: d["auroc"] if np.isfinite(d["auroc"]) else 0)
    L.append(f"  best={best['statistic']} AUROC={best['auroc']:.3f} -> " + (
        "GO: uncertainty ranks errors, a threshold can buy accuracy"
        if best["auroc"] >= 0.65 else
        "WEAK: some signal, thresholding will buy little"
        if best["auroc"] >= 0.55 else
        "NO-GO: uncertainty does not rank errors; no alpha will help"))
    L.append("  conformal sweep (one-sided, calibrated on val) for " + best["statistic"] + ":")
    L.append(f"    {'alpha':>6} {'coverage':>9} {'sel_acc':>8} {'n_kept':>7}")
    for r in best["conformal"]:
        L.append(f"    {r['alpha']:>6.2f} {r['coverage']:>9.3f} {r['sel_acc']:>8.3f} {r['n_kept']:>7d}")
    return "\n".join(L)


# ------------------------------------------------------------ harness entry pt

def run_from_features(Xc, yc, Xt, yt, assignment=None, n_groups=K_DEFAULT,
                      probe_cal=None, probe_test=None, probe_pred=None,
                      score_cal=None, score_test=None):
    """Build every available readout's report from raw exc-rate features.

    Three readouts, and the distinction decides what can honestly be claimed:

      pool             uniform pooling over each class group, L1-normalized. No
                       fitted parameters -- the network's decision at its rawest.
                       Can only represent evidence FOR a class.
      learned_readout  the plastic cluster->class weights the reward rule trained,
                       normalized by the same softmax that rule uses internally.
                       Still the NETWORK's decision, and the only one that can
                       express evidence AGAINST a class (negative weights), which
                       uniform pooling structurally cannot.
      linear_probe     an external logistic regression fitted post-hoc on all exc
                       rates using the true labels. NOT part of the network -- a
                       CONTROL, bounding how much class information the population
                       carries at all. Its uncertainty is the probe's, not the
                       network's, so it cannot carry a "the network abstains" claim.

    score_cal / score_test -- learned-readout scores from learned_readout_scores();
    None when the run had no plastic readout (readout_lr == 0), in which case pool
    IS the network's readout.
    """
    reports = []
    Rc = Rt = None
    if assignment is not None:
        Rc = group_rates(Xc, assignment, n_groups)
        Rt = group_rates(Xt, assignment, n_groups)
        reports.append(uncertainty_report(
            share_probs(Rc), share_probs(Rt), Rc, Rt, Rt.argmax(1), yt, readout="pool"))
    if score_cal is not None and score_test is not None:
        reports.append(uncertainty_report(
            softmax_probs(score_cal), softmax_probs(score_test), Rc, Rt,
            np.asarray(score_test).argmax(1), yt, readout="learned_readout"))
    if probe_cal is not None and probe_test is not None:
        pred = probe_test.argmax(1) if probe_pred is None else probe_pred
        reports.append(uncertainty_report(
            probe_cal, probe_test, Rc, Rt, pred, yt, readout="linear_probe"))
    return reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True,
                    help="uncertainty_features.npz written by interp_harness, or its run dir")
    ap.add_argument("--n-groups", type=int, default=K_DEFAULT)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    path = a.features
    if os.path.isdir(path):
        path = os.path.join(path, "uncertainty_features.npz")
    d = np.load(path, allow_pickle=False)
    assignment = d["assignment"] if "assignment" in d else None
    reports = run_from_features(
        d["X_cal"], d["y_cal"], d["X_test"], d["y_test"],
        assignment=assignment, n_groups=a.n_groups,
        probe_cal=d["probe_cal"] if "probe_cal" in d else None,
        probe_test=d["probe_test"] if "probe_test" in d else None,
        score_cal=d["score_cal"] if "score_cal" in d else None,
        score_test=d["score_test"] if "score_test" in d else None,
    )
    for r in reports:
        print(format_report(r))
    out = a.json_out or os.path.join(os.path.dirname(path), "uncertainty.json")
    with open(out, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"[uncertainty] wrote {out}")


if __name__ == "__main__":
    main()
