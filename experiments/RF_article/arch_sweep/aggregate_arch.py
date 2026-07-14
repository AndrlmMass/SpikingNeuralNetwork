"""
Aggregate the architecture sweep: one row per config with final test acc/phi and
the val/representation trajectory endpoints, so the effect of STDP / Vogels / WTA
on ORIENTED RFs is readable at a glance.

Key reads:
  test_acc          vs the 'frozen' row  -> does learning match/beat frozen?
  d_val_phi         (last - first)        -> is the representation improving?
  d_active_frac     (last - first)        -> code getting sparser (<0) or denser (>0)?

  python experiments/RF_article/arch_sweep/aggregate_arch.py --results-dir <run_dir>
"""
import argparse, glob, json, os

ORDER = ["frozen", "exc", "exc_vogels", "wta_frozen", "wta_exc", "wta_exc_noEE", "wta_exc_vogels"]


def first_last(hist, key):
    vals = [h[key] for h in hist if h.get(key) is not None]
    return (vals[0], vals[-1]) if vals else (float("nan"), float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    args = ap.parse_args()

    rows = {}
    for d in glob.glob(os.path.join(args.results_dir, "*")):
        rp = os.path.join(d, "results.json")
        if not os.path.isfile(rp):
            continue
        data = json.load(open(rp))
        tag = data["config"]["tag"]
        vh, sh = data.get("val_history", []), data.get("stats_history", [])
        vphi0, vphi1 = first_last(vh, "val_phi")
        vacc0, vacc1 = first_last(vh, "val_acc")
        af0, af1 = first_last(sh, "active_frac_exc")
        g0, g1 = first_last(sh, "rf_gini")
        rows[tag] = dict(test_acc=data.get("test_acc"), test_phi=data.get("test_phi"),
                         vacc0=vacc0, vacc1=vacc1, dphi=vphi1 - vphi0,
                         dactive=af1 - af0, dgini=g1 - g0,
                         elapsed=data.get("elapsed_s"))

    frozen_acc = rows.get("frozen", {}).get("test_acc")
    hdr = f"{'config':<16} | {'test_acc':>8} | {'vs froz':>7} | {'test_phi':>8} | {'val_acc traj':>14} | {'d_val_phi':>9} | {'d_active':>8} | {'d_gini':>7}"
    print(hdr); print("-" * len(hdr))
    for tag in ORDER:
        if tag not in rows:
            print(f"{tag:<16} |   (missing)")
            continue
        r = rows[tag]
        vs = (r["test_acc"] - frozen_acc) * 100 if (frozen_acc and r["test_acc"]) else float("nan")
        traj = f"{r['vacc0']:.3f}->{r['vacc1']:.3f}" if r['vacc0'] == r['vacc0'] else "     —      "
        print(f"{tag:<16} | {r['test_acc']*100:7.2f}% | {vs:+6.2f}pp | {r['test_phi']:8.4f} | "
              f"{traj:>14} | {r['dphi']:+8.4f} | {r['dactive']:+8.4f} | {r['dgini']:+7.4f}")
    print("\nd = last - first checkpoint.  d_val_phi>0: representation improving; "
          "d_active>0: code getting denser (usually worse for separability).")


if __name__ == "__main__":
    main()
