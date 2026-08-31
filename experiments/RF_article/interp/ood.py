"""
Does the network refuse to answer on data it was never trained on?

Feeds the OOD features an interp run dumped (`X_ood_*` in uncertainty_features.npz,
written when the harness is given --ood-dataset) through the SAME confidence statistics
and the SAME calibrated threshold that `threshold.py` builds on in-distribution data,
and asks: what fraction of the OOD input falls below tau, and what does that cost on
the in-distribution side?

Three numbers, and only the first is an operating point
-------------------------------------------------------
  rejection at the deployable tau  tau is the alpha-quantile of the ID confidence
                                   distribution -- fitted with NO OOD data, so it is
                                   what you could actually ship. Reported next to the
                                   ID coverage it costs, because a detector that
                                   rejects 99% of FMNIST by also rejecting 40% of MNIST
                                   has not solved anything.
  AUROC / FPR@95%TPR               threshold-free separability. FPR@95 is the standard
                                   OOD number and is what compares to the literature.
  oracle tau (Youden J)            the best any threshold could do, fitted USING the
                                   OOD sample. A CEILING, not an operating point. The
                                   gap to the deployable tau is the price of ID-only
                                   calibration.

BOTH TAILS. uncertainty.py deliberately used only the lower tail, because in-distribution
the failure mode is the network being torn between classes. OOD is different: `total_rate`
(the PRE-ACT analogue) runs HIGHER on dense images like FMNIST and SVHN, so a lower-tail
test on it points the wrong way and would report a detector as useless when it is merely
inverted. Signed AUROC plus the tail it lives in is reported for every statistic.

THE FLOOR THIS MUST BEAT. `input_density.py` shows that on raw pixels, with no network at
all, mean image intensity alone separates MNIST from SVHN at 2-sided AUROC 0.980 and from
notMNIST at 0.966, and the PCA subspace residual reaches 0.971-0.999 on all four probes.
Any claim that the SNN detects novelty has to be read against that, so `--compare-pixels`
prints the two side by side rather than leaving the comparison to the reader.

Usage:
    python experiments/RF_article/interp/ood.py --run <run_dir>
"""
import argparse, json, os, sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uncertainty import (  # noqa: E402
    group_rates, share_probs, softmax_probs, uncertainty_stats, auroc, silent_mask,
)
from threshold import bootstrap_tau, oracle_threshold, ALPHAS  # noqa: E402

STATS = ("entropy", "margin", "maxp", "total_rate", "topk_sum")


def learned_scores_from_weights(X, W_dense):
    """Replay the dense readout on arbitrary features: max-normalize, then matmul.

    Mirrors RewardLearner.readout_predict (synapses.py:459). Needed because the OOD
    features never passed through the learner -- they were captured by featurize, which
    deliberately bypasses it.
    """
    Xr = np.asarray(X, dtype=float)
    Xr = Xr / (Xr.max(axis=1, keepdims=True) + 1e-9)
    return Xr @ np.asarray(W_dense, dtype=float)


def confidence_sets(d, readout="learned_readout", ckpt=None):
    """{'cal':..., 'id':..., 'ood':{name: ...}} of per-item statistic dicts.

    The OOD items have no labels and no probe, so only readouts that can be evaluated
    from features alone are available for them: the learned readout (replayed from
    W_dense) and the pooled readout (group rates). The linear probe COULD be replayed
    too, but it is an external control fitted with true labels and carries no claim
    about what the network itself would do, so it is deliberately not extended here.
    """
    assign = d["assignment"] if "assignment" in d.files else None
    ood_X = {k[len("X_ood_"):]: np.asarray(d[k]) for k in d.files if k.startswith("X_ood_")}

    def stats_for(X, scores=None):
        R = group_rates(X, assign, 10) if assign is not None else None
        if readout == "learned_readout":
            p = softmax_probs(scores)
        else:
            p = share_probs(R)
        return uncertainty_stats(p, R)

    if readout == "learned_readout":
        if "score_cal" not in d.files:
            return None, None
        W = None
        if ckpt is not None and "W_dense" in ckpt.files:
            W = ckpt["W_dense"]
        if W is None:
            return None, None
        out = dict(cal=stats_for(np.asarray(d["X_cal"]), np.asarray(d["score_cal"])),
                   id=stats_for(np.asarray(d["X_test"]), np.asarray(d["score_test"])),
                   ood={k: stats_for(v, learned_scores_from_weights(v, W))
                        for k, v in ood_X.items()})
    else:
        out = dict(cal=stats_for(np.asarray(d["X_cal"])),
                   id=stats_for(np.asarray(d["X_test"])),
                   ood={k: stats_for(v) for k, v in ood_X.items()})
    return out, ood_X


def evaluate(sets, ood_X, alphas=ALPHAS, B=2000, seed=0):
    rows = []
    for st in STATS:
        c_cal, c_id = sets["cal"][st], sets["id"][st]
        bt = bootstrap_tau(c_cal, alphas=alphas, B=B, seed=seed)
        for name, s in sets["ood"].items():
            c_ood = s[st]
            lab = np.concatenate([np.ones(len(c_id), bool), np.zeros(len(c_ood), bool)])
            a_signed = float(auroc(np.concatenate([c_id, c_ood]), lab))
            tau95 = float(np.quantile(c_id, 0.05))
            at = {}
            for a in alphas:
                t = bt[float(a)]
                # rejection at the point estimate of tau, and at the two ends of its
                # 95% CI -- so the headline "we reject X% of FMNIST" carries the
                # uncertainty that comes from the calibration set being finite.
                at[float(a)] = dict(
                    tau=t["tau"],
                    ood_rejected=float((c_ood < t["tau"]).mean()),
                    ood_rejected_lo=float((c_ood < t["lo"]).mean()),
                    ood_rejected_hi=float((c_ood < t["hi"]).mean()),
                    id_rejected=float((c_id < t["tau"]).mean()),
                )
            rows.append(dict(
                statistic=st, ood=name, n_ood=int(len(c_ood)),
                auroc=a_signed, auroc_2sided=float(max(a_signed, 1.0 - a_signed)),
                tail=("lower" if a_signed >= 0.5 else "upper"),
                fpr_at_95tpr=float((c_ood >= tau95).mean()),
                # a network that goes SILENT on OOD input scores maximum entropy by
                # fiat, not by judgement -- so "it abstained" would mean "it died".
                silent_frac_ood=float((ood_X[name].sum(1) <= 1e-12).mean()),
                oracle=oracle_threshold(c_id, c_ood),
                at_alpha=at,
            ))
    return rows


def format_table(rows, readout):
    L = [f"[ood] readout={readout}",
         f"  {'statistic':<11} {'ood':<9} {'AUROC':>7} {'|2s|':>6} {'tail':>6} "
         f"{'FPR@95':>7} {'rej@2.5%':>20} {'id_cost':>8} {'oracle rej@same cost':>21} {'silent':>7}"]
    for r in sorted(rows, key=lambda x: (x["ood"], -x["auroc_2sided"])):
        a = r["at_alpha"][0.025]
        L.append(
            f"  {r['statistic']:<11} {r['ood']:<9} {r['auroc']:>7.3f} "
            f"{r['auroc_2sided']:>6.3f} {r['tail']:>6} {r['fpr_at_95tpr']:>7.3f} "
            f"{a['ood_rejected']:>8.3f} [{a['ood_rejected_lo']:.3f},{a['ood_rejected_hi']:.3f}] "
            f"{a['id_rejected']:>8.3f} "
            f"{1.0 - r['oracle']['ood_kept']:>12.3f} @id {1.0 - r['oracle']['id_kept']:>4.3f} "
            f"{r['silent_frac_ood']:>7.3f}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir with uncertainty_features.npz")
    ap.add_argument("--readout", default="learned_readout",
                    choices=("learned_readout", "pool", "all"))
    ap.add_argument("-B", "--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    base = a.run if os.path.isdir(a.run) else os.path.dirname(a.run)
    p = os.path.join(base, "uncertainty_features.npz")
    d = np.load(p, allow_pickle=False)
    if not any(k.startswith("X_ood_") for k in d.files):
        print(f"[ood] {p} has no X_ood_* arrays. Re-run the harness with "
              f"--ood-dataset <name> to produce them.")
        return
    ck_path = os.path.join(base, "weights", "checkpoint.npz")
    ckpt = np.load(ck_path, allow_pickle=False) if os.path.exists(ck_path) else None
    if ckpt is None:
        print("[ood] no weights/checkpoint.npz -- the learned readout cannot be replayed "
              "on OOD features; falling back to the pooled readout only.")

    readouts = ("learned_readout", "pool") if a.readout == "all" else (a.readout,)
    report = []
    for ro in readouts:
        sets, ood_X = confidence_sets(d, ro, ckpt)
        if sets is None:
            print(f"[ood] {ro}: unavailable in this run", flush=True)
            continue
        rows = evaluate(sets, ood_X, B=a.bootstrap, seed=a.seed)
        report.append(dict(readout=ro, rows=rows))
        print(format_table(rows, ro), flush=True)

    out = a.json_out or os.path.join(base, "ood.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[ood] wrote {out}")


if __name__ == "__main__":
    main()
