# Project Diary

Running log of work sessions, **newest first**. Purpose: fast re-entry across
devices/sessions — what was done, why, what landed (commits), and what's next.
Complements git history (the detailed "what") and any tooling notes.

**Convention:** add a `## YYYY-MM-DD — <short title>` section at the TOP for each
session. Capture: focus, key decisions, commits (hashes), and open items / next steps.
Keep it scannable — a few bullets, not a transcript.

---

## 2026-08-18 — Split-leakage fix + two-phase HPC run design (5 seeds × 3 epochs)

**Focus:** Prepare to re-run the whole study on Orion with comparable numbers. Fixed a
train/test **leakage** bug in the data loader, reconstructed the phase-1/phase-2 run plan
from the Notion diary (the 12-Aug meeting), added the metrics that plan needs, and wrote
the four driver scripts. **Nothing launched** — staged for HPC upload. Changes are in the
uncommitted working tree.

### THE BUG — the loader merged train+test and re-carved, contaminating "test"

`ImageDataStreamer` **merged** torchvision's train and test splits into one pool,
reshuffled, and carved train/val/test from the union. So our reported "test" set was a
random draw that mixed in training images — **not** the dedicated test set, and **not
comparable** to any published number. Confirmed live in the old SVHN log: *"Found 99289
image samples"* (73257 train + 26032 test), test drawn from the merger. This is exactly
Hubin's 12-Aug task #1 ("use the standard 10k test set; borrow the 1k val from train").

**Fix (core rewrite, not a patch):** new `neurosnn/_data/_partition_indices()` respects
each dataset's **dedicated** split — train/val drawn ONLY from the canonical train split,
test ONLY from the canonical test split; the two never mix, and because the seed only
shuffles train↔val, **the test set is identical across seeds**. notMNIST is the sole
exception (no published split; deeplake's separation is synthetic) → merge its 18724 and
carve a seed-fixed 15k/1k/2724. Over-subscription raises loudly (pre-run), keeping the
07-30 empty-split guard. Unit-tested: zero train/val/test overlap, no leak in the
dedicated regime, test invariant across seeds while train differs, all guards fire.

**Per-dataset splits (BOTH phases, "same setup"):** mnist/fmnist/kmnist 59000/1000/**10000
full test**; svhn 59000/1000/**26032 full test**; notmnist 15000/1000/2724. CIFAR-10
**dropped** (28×28-grayscale → chance for both priors, the degenerate control that also
nan-crashed last round).

### The run plan (reconstructed from Notion — 12-Aug meeting)

Paper arc: unsupervised model → trace-STDP → recurrency → tune → *show it fails, RFs end
up worse than random* → literature → strip back → pivot to supervised R-STDP.

- **PHASE 1 — unsupervised ablation** on the recurrent, NON-grouped net (input→exc with
  E→E→inh→exc, N_exc=N_inh=1024). One departure from a trace-STDP baseline at a time:
  `base_ori`, `base_rnd` (the RF-vs-random headline), `triplet`, `frozen`, `ee_off`,
  `ie_off` (`--peak-ie 0`), `vogels`. 7 conds × 5 datasets × 5 seeds = **175 runs**.
  Metrics: orientation coherence + **new 2D-Gaussian RF variance/covariance trajectory**
  (Hubin), L1-LR linear-probe acc (fit on 5k), per-neuron η², **corrected dead fraction**,
  participation ratio.
- **PHASE 2 — supervised** tiled R-STDP + dense readout, oriented **vs random** × 5
  datasets × 5 seeds = **50 runs**. Metrics: predictive entropy, readout acc, probe acc,
  coverage–accuracy, AUROC.
- **Both: 5 seeds, 3 epochs** (Andreas' call — 07-30 showed convergence well before 3ep;
  more seeds > more epochs for statistical power). **No PCA** (verified unhelpful; the
  probe was already `StandardScaler`+L1-LogReg, PCA-free — we just stop quoting the
  `pca_lr` evaluator).

### Harness / metric additions (all smoke-tested end-to-end)

- **`--probe-fit-all N`** — fits the FINAL linear probe (`test_lin_acc` + uncertainty) on
  N **train-split** features instead of the ~1k val set, via new `Runner.featurize()` /
  `Model.featurize()` (a no-update eval pass that doesn't touch the Evaluator/captured
  features). Set to **5000** in both sweeps — directly fixes the 07-25 "88% probe was
  data-starved, recovers to ~94.9% with enough fit data" artefact. Per-checkpoint drift
  probe still uses val (cheap); only the final classifier's fit set changes.
- **Corrected dead fraction** (`dead_frac_corrected = 1 − n_active/N_exc`, relative
  activity floor) now logged for EVERY run — previously only emitted when a class
  assignment existed, so the non-grouped phase-1 model had no dead measure.
- **`rf_gaussian_moments()`** in `analysis.py` — energy-weighted mean var_x/var_y/cov_xy +
  elongation + orientation of each RF; unit-verified (vertical bar→high var_y, etc.).

### Scripts (4 files, all `bash -n` clean)

- `interp/phase1_ablation/run_slurm.sh` (array 0-174) + `run_local.sh` twin — NEW.
- `interp/mnist_family_sweep/run_slurm.sh` (array 0-49) + `run_local.sh` — updated to
  oriented+random, 5 datasets, 5 seeds, 3 epochs, dedicated splits, `--probe-fit-all 5000`.
- **SVHN ordered LAST** in all four (highest task IDs / last in each seed pass): dense
  natural images fire ~3-4× more spikes/epoch (rate-coded cost ∝ total spikes) and it now
  carries the full 26032 test — decisively the slowest. This is **inherent, not a bug**;
  nothing to patch, just scheduled last so the fast datasets land first.

**Validation:** phase-1 condition paths smoke-tested on tiny MNIST runs — frozen, random,
vogels (inh. plasticity), ie_off (peak-ie 0), triplet, oriented-trace — all exit 0 with
the probe fit on train features and `test_lin_acc` produced.

**Open / next:**
1. Upload codebase + scripts to Orion; submit phase-1 (175) and phase-2 (50) arrays.
2. Post-hoc (off saved `uncertainty_features.npz`, no extra compute): bootstrapped 95% CI
   entropy rule (one-sided), OOD check (train MNIST → feed FMNIST, expect abstain),
   mixed-effects + multiple-regression of clustering metrics on accuracy (Hubin).
3. Scale-invariance experiment (normalise active-pixel extent; revisit the inward RF
   shift) — the diagnosed cause of notMNIST/FMNIST losing.
4. First draft due **1 Sept** (Andreas at bootcamp 2-10 Sept); TMLR target **1 Oct**.

---

## 2026-07-30 — 6-dataset sweep running; a silent empty-test-split bug cost 46h

**Focus:** Launched the extended dataset sweep (2026-07-28 12:15, still running) and
audited the first completed cells. Found that CIFAR-10 had been training for ~23h a
cell and reporting `test_acc=nan`.

**Sweep:** `results/run_local_20260728_121501`, 6 datasets x {oriented, random} x 3
seeds = 36 cells, 5 epochs, 6-way local parallelism, driven by the untracked
`run_local.sh` (local no-SLURM twin of `run_slurm.sh`). As of 2026-07-30 12:40:
**12 cells complete, 6 in flight, CIFAR-10 now failing fast by design.** ETA ~2026-08-02.

### Seed-0 RF-vs-random — mixed, and that is the result

| dataset | oriented | random | delta | eta2 (val, final ckpt) |
|---|---|---|---|---|
| mnist | **0.8656** | 0.8215 | **+4.4** | 0.167 / 0.177 |
| kmnist | **0.7465** | 0.7044 | **+4.2** | 0.117 / 0.101 |
| fmnist | 0.7194 | **0.7363** | -1.7 | 0.295 / 0.299 |
| notmnist | 0.8177 | **0.8363** | -1.9 | 0.224 / 0.169 |
| svhn | 0.2658 | *running* | — | 0.013 |
| cifar10 | *see below* | *see below* | — | 0.052 / 0.038 |

**SINGLE SEED — do not quote yet.** Seed 1 agrees on mnist so far (oriented 0.8788;
random still running, val 0.807 vs oriented's 0.890). Read provisionally: the oriented
prior helps on the digit-like sets and slightly *hurts* on the two harder ones. If that
survives seeds 1-2 it is a real and reportable boundary on the prior's usefulness — not
the clean win, but a more honest one. eta2 tracks accuracy across datasets in the
expected direction (fmnist 0.29 down to svhn 0.013).

### THE BUG — merged-pool splits silently produced an EMPTY test set

`ImageDataStreamer` **merges** torchvision's train and test splits into one pool and
carves train/val/test from it, so the budget is `len_train + len_test`, NOT `len_train`:

| dataset | pool | requested 59000/1000/10000 |
|---|---|---|
| mnist / fmnist / kmnist | 70000 | fits **exactly** |
| svhn | 99289 | fits |
| **cifar10** | **60000** (50k+10k) | **test = 0** |

The old code **clamped each count to whatever was left instead of erroring**, so
CIFAR-10 got `test = min(10000, max(0, 60000-59000-1000)) = 0`. The test phase never ran
at all (`grep -c "Testing"` = 0 for cifar10 vs 2 for svhn) and it surfaced only as
`test_acc=nan` in the final line of a 23h log. **Both cifar10 seed-0 cells are lost:**
no `uncertainty_features.npz`, no weight checkpoints, so **nothing can be scored
offline** — they must be rerun.

Why it hid: `59000+1000 = 60000` is *exactly* CIFAR-10's pool, so train and val both
filled completely and only test was starved. notMNIST was already special-cased in
`run_local.sh` for the same reason; CIFAR-10 was missed because it very nearly fits.

**Fixes (uncommitted working tree):**
- `neurosnn/_data/get_data.py` — over-subscribed splits now **raise** with the actual
  numbers instead of truncating. Note `>` not `>=`: mnist/fmnist/kmnist sit at exactly
  70000 and must still pass.
- `run_slurm.sh` — had the bug **worse** (hardcoded 59000/1000/10000, no per-dataset
  branch at all), so on Orion it would have failed cifar10 *and* notmnist. Now
  dispatches per dataset: cifar10 49000/1000/10000, notmnist 14000/1500/3000. val/test
  held constant wherever possible so metrics stay comparable; only train volume shrinks.
- `run_cifar10_fix.sh` — staged, **not launched** (see below). Its `is_complete` also
  requires `test_acc` to be **finite**, not merely present — the existing check counts
  `nan` as complete and would skip the broken cells forever.
- `run_local.sh` **deliberately not edited**: bash reads a running script by byte offset,
  and the driver (PID 85794) is live. The two edits it needs are recorded in the fix
  script's header for after it exits.

**Verified in production:** `cifar10_oriented_s1` and `cifar10_random_s1` hit the new
guard at 12:38/12:39 and failed in ~30s each instead of ~23h. `set -uo pipefail` has no
`-e`, so the driver logged `FAIL` and carried on. **~96h of compute saved**; those four
`FAIL` lines in `driver.out` are expected, not new breakage.

**Decision: CIFAR-10 waits for HPC.** Not worth 3 local days at 2 slots. The repair
script + the SLURM per-dataset splits are ready for Orion.

### Instrumentation gap — the sweep collects NO spiking-readout data

`--spiking-readout` is opt-in (`interp_harness.py:142`) and `run_local.sh` does not pass
it, so `test_spiking_acc` is `None` for all 36 cells. The dataset-generality sweep and
the artificial-vs-biological readout comparison (07-27 next-item #3, the one the paper's
faithfulness claim rests on) are therefore **disjoint** — the spiking numbers need their
own run. Not a bug; the delta readout is the intended control here. Worth deciding
whether the HPC round carries `--spiking-readout` so one sweep answers both.

### TRAINING LENGTH — no decay, but epochs 2-5 are wasted compute

Checkpoint-level analysis of the 10 completed cells (one checkpoint per batch, 59/epoch).
**Noise floor first:** the val set is 1000 items, so 2sd binomial noise is **+-2.53pp** —
nothing smaller than that is readable.

**No decay anywhere.** Worst final-minus-peak across all 10 cells is **-0.96pp**, i.e.
every apparent decline is inside noise. Training longer is not destroying anything, and
the 07-16 "over-training DEGRADES" finding was a **learning-RATE** effect (reward_lr),
not an epoch-count effect — at reward_lr 5e-6 the two do not reproduce each other.

**But val_acc saturates within the first ~20% of epoch 1** (~11.8k images; that is an
upper bound — the first checkpoint is already post-training, so it may be earlier):

| prior | epochs 2-5 buy | range |
|---|---|---|
| oriented | **+0.03pp** | -0.57 to +0.70 |
| random | **+0.46pp** | +0.18 to +0.66 |

Both inside the noise floor. **~80% of every 23h cell buys nothing measurable in
accuracy.** Consistent with 07-25's "refit probe is FLAT — decodability did not improve";
this is the same result seen from the accuracy side. Note training is still *doing*
something (eta2 moves monotonically below, and drift was still widening at epoch 5) — it
just does not cash out.

### eta2 moves in OPPOSITE directions for the two priors (the good finding)

Matched on dataset AND seed (n=4 pairs: fmnist/kmnist/mnist/notmnist, all s0), eta2 from
epoch 1 -> epoch 5:

| dataset | oriented | random |
|---|---|---|
| fmnist | 0.312 -> 0.295 (**-0.018**) | 0.269 -> 0.299 (**+0.030**) |
| kmnist | 0.119 -> 0.117 (-0.003) | 0.087 -> 0.101 (+0.014) |
| mnist | 0.171 -> 0.167 (-0.004) | 0.146 -> 0.177 (+0.031) |
| notmnist | 0.232 -> 0.224 (-0.008) | 0.159 -> 0.168 (+0.010) |
| **mean** | **-0.008** | **+0.021** |

**4/4 negative and 4/4 positive — perfect separation.** The oriented prior *front-loads*
per-neuron selectivity and training erodes it; random starts lower, builds it up, and
**overtakes** (fmnist 0.299 vs 0.295, mnist 0.177 vs 0.167). This is the **random control
for the "reward-STDP bends the prior" claim** from 07-25 — the counterfactual says the
erosion is specific to having a structured prior to erode. Seed check: mnist_oriented has
two seeds and the delta reproduces at -0.005 / -0.004, so seed noise on this quantity is
~0.001, well clear of the +-0.02-0.03 contrast.

**The accuracy gap is front-loaded too, and shrinks** (oriented - random, pp):

| dataset | 0.2ep | 1ep | 3ep | 5ep |
|---|---|---|---|---|
| kmnist | +8.70 | +5.90 | +6.50 | **+2.90** |
| mnist | +8.30 | +5.40 | +5.40 | **+5.50** |
| fmnist | +0.00 | -0.50 | -1.70 | **-2.00** |
| notmnist | -1.93 | -4.20 | -2.93 | **-1.87** |
| **mean** | **+3.77** | +1.65 | +1.82 | **+1.13** |

**THE DISSOCIATION worth putting in the paper:** random **overtakes oriented on eta2**
while oriented **still wins on accuracy** (mnist +5.5, kmnist +2.9). So per-neuron
selectivity is **not** the mechanism behind the oriented prior's advantage — which is
exactly 07-25's "individual neurons got less selective and more alike; the class signal
moved into the population", now with a control arm.

**Caveats:** n=4 matched pairs, single seed. A 4/4 sign test is p=0.0625 one-tailed —
directionally perfect but underpowered. Per-dataset accuracy gaps are mostly within
noise; only mnist and kmnist clear it. Scripts in scratchpad, not committed.

**RECOMMENDATION: cut 5 epochs -> 2 for the HPC round.** 2.5x cheaper per cell at no
measurable accuracy cost, and it converts directly into what we actually lack — more
seeds and a `--spiking-readout` arm. Do **not** go to 1 epoch: svhn/cifar10 have slower
dynamics and epoch 2 is where the eta2 curves are still visibly separating.

### Metric framing clarified (no code change)

- `class_eta_squared` is **standard**, not homegrown: eta-squared / Pearson's correlation
  ratio / the R^2 of a one-way ANOVA, and in systems neuroscience it is **PEV**. Cite it
  that way in the paper rather than defending it from scratch.
- It is **monotone in Calinski-Harabasz** for a single feature (CH *is* the ANOVA
  F-statistic; verified numerically to the decimal). The real difference is aggregation:
  ours is **mean-of-ratios** (every neuron one vote, normalized by its own variance), CH
  is **ratio-of-sums** (high-variance neurons dominate). On a synthetic 45-informative /
  5-loud population these give 0.637 vs 0.051 — same data, different question.
- **Bias floor:** raw eta2 has E[eta2] ~ (K-1)/(n-1) under the null (~0.045 at K=10,
  n=200). Constant across cells at fixed n and K so trends are safe, but absolute values
  are inflated. `omega^2` debiases if we ever quote absolutes; omega^2-PEV is the usual
  choice in the literature.
- **Entropy and perplexity are one measurement in two units** (`exp` is monotone -> same
  ranking, same AUROC, same abstention decisions). When the report shows perplexity,
  margin and maxp all at AUROC ~0.94 that is **not** three converging pieces of evidence.
  Margin and maxp are genuinely distinct; entropy and perplexity are not.

**Open / next:**
1. Let the sweep finish (~2026-08-02), then multi-seed the RF-vs-random deltas — the
   sign flip between digit-like and harder datasets is the claim to nail down.
2. **HPC round at 2 epochs, not 5** (see training-length section) — spend the 2.5x saving
   on seeds and a `--spiking-readout` arm. The eta2 divergence is the headline to
   replicate; it needs seeds, not epochs.
3. CIFAR-10 repair run on Orion via `run_cifar10_fix.sh` / fixed `run_slurm.sh`.
4. Decide whether the HPC round carries `--spiking-readout` (see gap above).
5. Apply the two staged `run_local.sh` edits once the local driver exits.
6. Filter `nan` before aggregating: the two dead cifar10 cells will poison any plain
   `mean` over `test_acc`.
7. Re-run the paired eta2 analysis once random seeds 1-2 land — 4/4 at p=0.0625 wants
   12 pairs to be quotable.

---

## 2026-07-27 — Spiking readout: two rule bugs, and negative weights cost only 0.6

**Focus:** Build a **biologically-motivated alternative** to the softmax delta
readout — real LIF output neurons trained by a local three-factor rule — to run
*alongside* the delta rule, not replace it. The comparison is the deliverable:
"artificial vs biological readout on identical activity" is worth reporting
whichever way it lands. Plan at `~/.claude/plans/cheeky-wobbling-avalanche.md`.

**Landed:** `neurosnn/_core/readout.py` (`9748fc9b`) — one LIF neuron per class
with its own mp / threshold / adaptation, plastic `W_ro`, decoded by **spike
count**. Standalone only; **not yet wired into the trainer**.

**Literature.** Goupy et al. 2024 (S2-STDP + Paired Competing Neurons) is the
primary template: error-modulated supervised STDP on the output layer, weights
renormalised to their initial sum after every update — **which is what
`post_norm` already does for us**. 98.59% MNIST. Mozafari et al. 2018 is the
reward/punish-the-winner alternative (first-spike WTA, STDP on correct,
anti-STDP on wrong), 97.2% MNIST with no external classifier. Both **beat our
95.5%**, which is the reason to think this is worth doing rather than merely more
faithful. Frémaux & Gerstner 2016 for the three-factor form (`M = R - b`; the
baseline is not optional). Legenstein/Pecevski/Maass 2008: R-STDP works as a
policy-gradient rule *because of* trial-to-trial firing variability —
`mean_noise`/`var_noise` are **hardcoded to 0.0** in the harness, so we run a
degenerate deterministic version. Worth exposing.

**TWO RULE BUGS, both caught by synthetic tests before any wiring:**
1. **The error was not zero-mean.** `+margin` on the target and `-margin` on each
   of C-1 non-targets leaves a **-(C-2)*margin DC push every sample** (-16 at
   C=10, margin=2). Walks every weight down until the layer falls silent. This is
   exactly the drift the three-factor theory warns about, and the plan had
   asserted the error was "already centred". It was not.
2. **Target-RATE error is the wrong translation of Goupy.** They rank firing
   *times*, which has no ceiling; a desired spike *count* invents one. A target
   correctly firing 20 spikes against a mean of 4 is judged 14 too high and gets
   **depressed**. Measured: frozen readout **1.00** on a separable task, switching
   learning on drove it to **0.40** with 87% of the layer silent — the rule was
   destroying a readout that already worked.

Replaced with a **margin rule**: error is nonzero only while a competitor sits
within `margin` of the target; `+1` on the target, `-1/n` shared among the n
offenders. Zero-sum by construction, no cap on the winner, and it stops updating
once correct. After: frozen 1.00, learning 1.00 (holds), adversarial
shuffled-mapping task 0.10 -> 0.52.

Also: used an **L1** renorm, not `post_norm` — `post_norm` divides by the
*signed* sum, which for mixed-sign weights can pass through zero and blow up.

**DECISIVE RESULT — negative readout weights are worth 0.6 points, not 13.**
Replayed the exact online delta rule on the frozen `run60k_5ep` features, 3 seeds,
7000 train / 3000 eval, identical except the weight floor:

| readout | test acc |
|---|---|
| signs free | **0.9397 +- 0.0033** |
| non-negative (w >= 0) | **0.9336 +- 0.0043** |
| gap | **0.61 points** |

The rule *does* use negatives heavily — 50% of trained weights are negative,
holding 36% of total |w| mass. But the distinction that matters:
- **delete negatives after training -> 0.881** (-5.9 points)
- **train with the constraint -> 0.934** (-0.6 points)

The information is **redundant, not unique**: a non-negative readout re-encodes
it in how it distributes the positive weights. Running only the post-hoc ablation
would have given exactly the wrong answer. (Absolute numbers sit ~1.5 below the
run's 95.51% because this is 56k updates on frozen features vs 300k on a
co-adapting network; the A/B is internally controlled.)

**Consequence — go non-negative.** Andreas' proposal, now backed by measurement.
`w >= 0` makes the silent-output death mode *structurally impossible* (worst case
is zero contribution, never inhibitory drive), and it makes per-neuron sum
conservation do what it should: weakening adversarial synapses automatically
strengthens the useful ones, which L1 conservation **cannot** do cleanly with
mixed signs. The two changes need each other. Removes the need to argue that a
negative feedforward weight stands for a disynaptic inhibitory path.

**Known limitation.** Eligibility is gated on post spikes, so an output that falls
silent stops accumulating eligibility and cannot be potentiated back. Intrinsic
threshold homeostasis mitigates but does not remove it. **Non-negative weights
should make this moot** — the layer can no longer be driven silent by its own
weights — so verify before building graded eligibility.

**Open question being tested next: why not straight R-STDP?** Answer: we
essentially are. The existing `reward_STDP` teacher (`+1` target / `-1` non-target,
baseline `(2-C)/C`) evaluates to **+1.8 / -0.2**, i.e. already zero-sum — the same
numbers the corrected margin rule uses. The only real difference is **when** it
fires: straight R-STDP updates *unconditionally* every sample (correlational /
prototype learning), the margin rule only while the answer is wrong
(discriminative / error-correcting). Both share the silent-post trap, since
`#pre x #post` is zero when the post is silent. Which wins is empirical.

**Next:** three-way comparison on identical activity — delta softmax vs straight
R-STDP (fixed +-1) vs margin R-STDP. Then wire into the trainer for the paired
in-run comparison.

---

## 2026-07-25 — Full 60k x 5-epoch run: 95.5%, and the "linear probe" was the artefact

**Focus:** First full-scale run of the thin-margin oriented prior + dense readout
(`results/rstdp_thinmargin/run60k_5ep`, 60k images x 5 epochs, seed 0), then an
exhaustive post-hoc analysis of every statistic we log — 60-checkpoint trajectory,
300 network-state records, the saved uncertainty features, and the four saved figures.

**Config:** prior `oriented` (rf_length 3.0 / rf_thickness 1.2 / center_margin 4.0),
rule `reward`, grouped WTA `block` 10x100, N_exc=N_inh=1000, `dense_readout=True`,
readout_lr 0.1, peak_ei 50 / peak_ie -2, `use_vogels=False`, `ee=False`,
normalize_weights on, 350 steps/sample.

### HEADLINE: learned readout = 0.9551 on 9000 held-out test images

Up from ~0.79 at 15k with the block-diagonal readout. The dense readout cashed in the
ceiling exactly as predicted on 2026-07-16. **Naming trap that cost us an hour:**
`results.json["test_acc"]` = **0.8647** is the harness's own `pca_lr` evaluator, NOT the
learned readout — that number lives in `uncertainty[1].base_acc`. Five decoders are
logged per run and the top-level key is the one nobody wants. Rename before the paper.

| decoder | test acc | what it is |
|---|---|---|
| **learned readout (dense, online)** | **0.9551** | the model's own answer — 1000x10, softmax delta, 300k samples |
| linear probe (`test_lin_acc`) | 0.8811 | L1-LR fit on **1000** val images — **data-starved, see below** |
| `pca_lr` evaluator (`test_acc`) | 0.8647 | harness Evaluator, scaler+PCA+LR |
| uniform pool | 0.7557 | fixed block-diagonal pooling |
| online train decisions | 0.9546 | matches the readout — no train/test gap |

### The 7-point learned-vs-linear gap is a CONTROL artefact, not a representation fact

`interp_harness.fit_clf` fits ~10k parameters on **700–1000 samples in 1000-D** (p ~= n).
Refitting the *byte-identical* probe on the frozen final features, varying only n:

| n_train | 700 | 1000 | 2000 | 4000 | **7000** |
|---|---|---|---|---|---|
| probe acc | 0.862 | 0.883 | 0.922 | 0.935 | **0.9485** |

(1000 reproduces the reported 0.8811; 5-fold CV cross-check at 7200/1800 = 0.944 +- 0.006.)
The gap collapses 7.4 -> 0.7 points. Regularisation is second-order: the whole penalty
sweep on the 1000-image fit spans 0.864 (unpenalised) to 0.883 (best L2) — ~0.6 points
vs the ~7 that data volume buys. Corroboration: learned-only-correct **8.7%** vs
probe-only-correct **1.3%** (6.7:1) = near-strict dominance, i.e. one *weaker* decoder on
the same information, not two different codes.

**Andreas' hypothesis (peak_ei=50 too strong) is ruled out** — four independent checks:
exc spike rate flat across all 5 epochs (0.002193 -> 0.002231, +1.7%) with inhibition
tracking it; `active_frac_exc` **rose** 0.950 -> 0.971 and dead_frac is 2.3% (vs 0.56 in
the old tiled runs — the thin-margin prior fixed the dead-neuron problem outright);
`ei_ratio` median 0.53 / p90 2.07 (firm, not crushing); and a linear decoder with enough
data hits 0.9485, so the class information is intact. **Where inhibition plausibly DOES
cost us is the pooled readout, not the gap:** all 10 groups peak on their own class, but
diag:off-diag is only **1.56**, and within-group response correlation (0.121) barely
exceeds across-group (0.113). Uniform pooling averages that thin margin away -> 0.756.
**If we sweep `peak_ei` again, the metric to move is pool accuracy and the diagonal
ratio — not the learned-vs-linear gap.**

### How the representation changed: BENT, not eroded

| metric | start | end | delta |
|---|---|---|---|
| orientation coherence | 0.667 | 0.492 | **-26%** |
| within-group RF diversity | 0.090 | 0.137 | **+52%** |
| w_floor_frac (pruned synapses) | 0.024 | 0.148 | +14.8pp |
| rf_mean_cosine | 0.097 | 0.125 | +29% |
| per-neuron eta2 | 0.169 | 0.161 | -5% |
| grouped eta2 (val_phi) | 0.178 | 0.275 | **+55%** |
| pop_sparseness | 0.485 | 0.552 | +14% |
| participation ratio | 32.1 | 33.7 | +5% |
| w_se_mean | 0.330175 | 0.330175 | **exactly constant** (L1 norm holding) |

Visually (`weights/rf_first.png` -> `weights/rf_last.png`): oriented Gaussian bars become **curved
stroke fragments** — arcs, hooks, C-shapes, partial loops. All of it inside a fixed
synaptic budget, so this is **reallocation**, not decay.

**This is materially different from the trace-STDP result** (`project_mechanism_stdp_erodes_prior`),
where the prior collapsed to an init-independent attractor. Reward-STDP *bends the prior
into digit strokes*. Better story for the article: the prior is a useful starting basis
that supervision refines, not a structure that plasticity destroys.

Counter-current worth owning honestly: per-neuron eta2 FELL 5% and RF pairwise cosine ROSE
29% while grouped phi rose 55%. Individual neurons got *less* selective and *more* alike;
the class signal moved into the population. Distributed-code signature — and exactly why
the dense readout (which reads all 1000) beat pooling (which reads 100).

### Representational drift is the cleanest signal in the run

A probe frozen at ckpt 0 decays **0.797 -> 0.597** while a refit probe holds ~flat at
0.83. `_drift` goes 0.000 -> 0.237, monotone across all 5 epochs, **still widening at
epoch 5**, no epoch-boundary discontinuity. Two readings, both important:
- The code keeps moving under a fixed decoder for the whole run — genuine drift, and we
  have a clean quantitative handle on it.
- **The refit probe being FLAT means decodability did not improve.** All 5 epochs of
  accuracy gain belong to the readout learning to read a code that was already about as
  decodable as it would ever get. Sobering for "does reward-STDP improve the features?"
  — on this measure, no. The architecture + prior set the ceiling; reward-STDP + readout
  reach it.

### Perplexity and confidence

Learned readout perplexity **1.854 -> 1.154** effective classes (per epoch:
1.227/1.199/1.175/1.148/1.154), margin 0.718 -> 0.928, entropy 0.499 -> 0.107. On test:
mean perplexity 1.124, entropy 0.063 on correct vs 0.601 on wrong (**9.6x separation**).

**Do not confuse the two perplexities.** Trajectory `perplexity` (9.951 -> 9.333, i.e.
pinned at chance) is the *pooled* readout's, and measures the pooling, not the code.
The readout's own is `perplexity_readout`. Reporting the former as a representation
failure would be plain wrong.

**Selective prediction — the strongest result in the run.** Learned readout entropy-vs-error
AUROC **0.941**, AURC 0.004, **98.8% accuracy at 90% coverage** and **99.6% at 80%**;
coverage at 95% accuracy = 1.00 (it is already above 95%). Linear probe AUROC 0.850;
pooled 0.557 (chance). **Caveat that must ship with the claim:** only *shape* statistics
work — entropy/perplexity/margin/maxp all ~0.94, but `total_rate` and `topk_sum` sit at
0.526–0.528, indistinguishable from chance. Abstention rests on the readout's output
distribution, not on activity level. Worth stating plainly in the paper rather than
quoting the 0.941 alone.

### Two instrumentation problems found

1. **Plasticity diagnostics are dead under `rule=reward`.** `mean_delta_w`, `mean_ltp`,
   `mean_ltd`, `ltp_ltd_ratio`, `mean_x_pre`, `mean_x_tar_se` are **exactly 0.0 at all
   300 records** — the STDP trace path is unused in the reward rule, so the "plasticity
   balance" panel in `stats/stats.png` is a flat zero line and update magnitudes are
   *inferred* (from RF change + w_se_std -9.4%), never measured. Agreed 2026-07-25: not
   worth fixing now, the RF/weight evidence is sufficient.
2. **Stale figure.** `stats/confusion.png` was written 01:42, before the run finished
   at 01:55, so both TEST panels still read "pending — end of run"; the matrices *are*
   in `results.json` (`test_cm_linear`, `test_cm_readout`). Low priority, may not be
   worth re-plotting at all. `stats/metrics.png` and `stats/stats.png` are current
   (01:42 = last checkpoint) and are the ones to read. Note the RF grids live in
   `weights/` (`rf_first.png`, `rf_last.png`, `group_rfs.png`), not the run root.
   (Separately: `run15k/metrics.png` at the run root is a mid-run leftover superseded by
   `run15k/stats/metrics.png` — the 60k run writes no root-level metrics.png at all.)

**Analysis artefacts:** full report generator saved next to the run at
`results/rstdp_thinmargin/run60k_5ep/build_report.py` (self-contained HTML, 14 charts,
every figure with a table view); published copy at
https://claude.ai/code/artifact/2cc8773e-802e-4592-9c54-6a35ed9c1e90

### Reflections

- **Our own control was the bottleneck for a whole analysis cycle.** We nearly wrote a
  causal story about inhibition to explain a number that was just an undertrained probe.
  New rule: when the probe and the readout disagree, **refit the probe on more data
  before reasoning about the representation**. `refit_acc` is "a data-limited external
  control", never a decodability ceiling.
- **The dense readout was the right call and is now done paying off** (0.79 -> 0.955).
  The remaining spectrum is pool 0.756 -> learned 0.955 -> probe-with-enough-data 0.949,
  i.e. the learned readout has *caught* the linear ceiling. There is no readout headroom
  left to harvest on MNIST; further gains must come from the representation.
- **The thin-margin prior quietly solved the dead-neuron problem** (0.56 -> 0.023 dead)
  that we spent two sweeps on in July. Worth a sentence in the paper.
- **We now have a drift measurement**, which is a paper-grade result in its own right and
  not one we set out to get.

**Open / next (agreed order):**
1. **Longer run on MNIST** (main dataset) — how far does 5 epochs -> 15–20 epochs go?
   Drift was still widening at epoch 5, so the run had not converged in any sense.
2. **Other datasets, same length** (60k x 5) to test generality vs specialization:
   Fashion-MNIST, KMNIST/EMNIST, then SVHN and CIFAR-10 (the last two are the real test —
   colour + natural statistics vs an oriented-bar prior). Launch in parallel on Orion.
3. **Replace the softmax delta readout with a spiking, reward-modulated readout** —
   convert each class output into an actual LIF neuron, decode by spike RATE, train it
   with reward-STDP rather than a delta rule. This is the biological-faithfulness step
   and matters most for the paper's claim; the current dense readout stays as the
   fallback/upper-bound control. Report both.
4. Multi-seed before any published number (everything above is seed 0).

---

## 2026-07-16 — RF-size + inhibitory-learning sweeps

**Focus:** Is the RF too large (whole-digit "quintessential" templates), and is static
uniform inhibition optimal? Two sweeps, both at the tuned config (reward_lr 5e-6,
readout_lr 0.1, peak_ei 50), 6k images/config, single seed.

**Tooling / harness flags added:** `--sigma-se` (structural RF footprint = sigma_se_mean;
0 keeps default 3.0), `--sigma-se-lognormal` (heterogeneous RF sizes), `--vogels-lr`,
`--vogels-rho0` (were hardcoded 0.01/0.1). All recorded in run config. Per-class RF grid
still overwrites each checkpoint (frames+GIF is a TODO). New: `sweep_rfsize.py`,
`sweep_inhib.py`. New backlog file `IDEAS.md` (parked pursuits + rationale).

**Key mechanism finding (why RFs are whole-digit):** the tiled init wires each SE neuron
to ~1/3 of the image (sigma_se=3.0 -> median 257 nonzero syn, ~19px radius), and
reward-STDP only ever touches STRUCTURAL synapses (`nonzero_pre_idx`), so the footprint is
bounded by init, not free to spread — but the init footprint is already near-global, so
reward-STDP paints the class-average template and the spatial tiling is cosmetic.
sigma_se 1.5 -> 64 syn/6px (local), 1.0 -> 30 syn/3.6px (tight).

**RF-size sweep (results/rstdp_rfsize/rfsize_main/):**
| config | refit-LR ceiling | learned | uniform pool | dead | win_ent |
|---|---|---|---|---|---|
| baseline_3.0 | 0.805 | 0.788 | 0.686 | 0.56 | 0.36 |
| local_1.5 | 0.797 | 0.778 | 0.575 | 0.37 | 0.40 |
| tight_1.0 | 0.800 | 0.747 | 0.471 | 0.31 | 0.41 |
| hetero_2.0ln | 0.825 | 0.772 | 0.344 | 0.78 | 0.13 |
- **Feature ceiling is ~FLAT (0.80–0.825) across all RF sizes** — local RFs lose no
  linearly-decodable class info. Ceiling spread is small + single-seed (don't crown hetero).
- **Smaller RF -> healthier** (dead 0.56->0.31, winners spread) but the simple pooling
  readouts collapse (uniform 0.69->0.47) because one local patch is weakly class-selective.
  Gap between ceiling and pooled readout blows open -> the READOUT is the bottleneck.
- **hetero (lognormal): median RF smaller (106 syn) but a heavy tail up to 784 (whole
  image).** A few whole-image detectors dominate the WTA (dead 0.78, win_ent 0.13 = most
  monopolized) -> highest ceiling (few strong templates, easy for a fitted classifier) but
  WORST uniform pool (0.344) and lowest diversity. Confirms: RF-size variance concentrates
  class info in few neurons = more separable-for-a-classifier but less representational
  width. hetero is NOT "best in total" — best only on the (noisy) ceiling.
- **Interpretation:** two objectives pull opposite ways — max separability-for-a-fitted-
  classifier favors few large templates (baseline/hetero, ~template-matching, mostly dead);
  healthy diverse distributed part-code favors small local RFs (tight/local, but needs a
  compositional readout). Ceiling ~flat means the distributed local code loses nothing real
  -> **dense readout is the pivotal next step**, not more RF tuning.

**Inhibitory-LR sweep (results/rstdp_inhib/inhib_main/, baseline RF, Vogels rho0=0.1):**
| vogels_lr | learned | uniform | dead | win_ent | ceiling |
|---|---|---|---|---|---|
| off (static) | 0.783 | 0.690 | 0.56 | 0.356 | 0.812 |
| 0.005 | 0.790 | 0.677 | 0.55 | 0.357 | 0.802 |
| 0.02 | 0.778 | 0.676 | 0.53 | 0.367 | 0.803 |
| 0.05 | 0.770 | 0.689 | 0.52 | 0.391 | 0.805 |
| 0.1 | 0.760 | 0.656 | 0.50 | 0.402 | 0.798 |
- Proper LR sweep (the old on/off hid this): **monotonic** — plastic inhibition DOES revive
  neurons + spread winners (dead 0.56->0.50, win_ent 0.356->0.402) but **weakly, at an
  accuracy cost** (learned 0.783->0.760). No LR gives both better health AND accuracy.
- **RF locality is ~4x stronger than inhibitory plasticity for the dead-neuron problem**
  (static 0.56 / Vogels-best 0.50 / local RF 0.31). Vogels rescues marginal neurons, not
  deeply-dead ones (strong 1:1 E->I hitman + reward-starved RFs still lose the drive).
- **Verdict:** leave vogels_lr ~0.005 (marginal accuracy peak) or off; don't invest more.

**Open / next (priority order):**
- **Dense/full readout** (N_exc x 10, delta-rule) — cash in the flat ~0.80 ceiling with the
  healthy local-RF code; report spectrum uniform->block-diag->full->LR-ceiling.
- Adaptive-threshold homeostasis (NCG-style) — direct lever on dead neurons + diversity.
- Test plastic inhibition in the LOCAL-RF regime (only tested at baseline so far).
- Causal/gated eligibility trace — parked, "not sold"; principled but modest gain expected
  with rate-coded input, and won't fix dead neurons. See IDEAS.md.
- Multi-seed everything before any paper claim (all above single-seed).

---

## 2026-07-16 — Hyperparameter tuning + paper venue/structure

**Focus:** Tune the tiled reward-STDP config (readout_lr, inhibitory drive, cluster
lr; Vogels on/off), and plan the write-up.

**Tuning (short 3k-image sweeps, tune.py; results/rstdp_tune/):**
- `readout_lr`: best ~**0.1** (learned readout 0.742); flat plateau 0.05–0.1, drops
  at 0.2. Uniform (cluster) flat ~0.64 across — cleanly separable from features.
- `peak_ei` (E→I drive): best ~**50** (learned 0.756, uniform 0.678, up from 0.63 at
  default 20). **dead_frac RISES monotonically with drive** (0.45→0.72) — stronger
  inhibition SHARPENS the WTA (kills more losers, cleaner winners), so "44% dead" is
  WTA sharpness, not weak drive. Over-inhibition (>=200) hurts. Peak knob alone
  suffices — no multi-input E→I architecture change needed. (1:1 WTA: density_ei/ie
  are inert; only the peaks matter.)
- `reward_lr` (cluster/SE lr): best **≤5e-6** (learned 0.762, uniform 0.721) —
  **MONOTONIC: gentler is better, over-training DEGRADES** the representation. This is
  the supervised echo of the trace-STDP instability theme — even reward-STDP erodes
  the representation when over-applied. Publication-relevant.
- **Best config: reward_lr 5e-6, readout_lr 0.1, peak_ei 50 → learned 0.762, uniform
  0.721** (vs ~0.65 at original defaults).
- **Vogels on/off (5k imgs, tuned config):** off = learned 0.783 / test 0.827;
  on = learned 0.789 / test 0.837. **!! INVALID — a BUG meant Vogels never fired in
  reward runs** (the ilearner.step lived inside the trace-path `update_weights_now`
  block, gated off for reward). W_ie stayed perfectly uniform at -2 regardless of
  --use-vogels. Fixed (Vogels now applied in the reward path at the sample boundary;
  verified W_ie differentiates, inh_in std 0 -> 0.115). **The Vogels comparison must be
  RE-RUN.** (Andreas spotted this from the all-black I->E panel in the live plot.)
  Note: tuned config at 5k still reaches learned ~0.79 / fitted-LR test ~0.83 (that
  part unaffected — it was the reward+readout, not inhibition).
- **Vogels RE-RUN (fixed, results/rstdp_controls/vogels2_*):** off = learned 0.790 /
  uniform 0.706 / dead 0.56 / test 0.841; on = learned 0.784 / uniform 0.683 / dead
  0.54 / test 0.847. **Now valid, and still ~neutral** (within single-seed noise).
  Vogels DOES act (revives a few dead neurons, spreads winners: dead 0.56->0.54,
  win_ent 0.358->0.377) but that redistribution slightly HURTS the class-pooled
  readout (uniform 0.706->0.683). Fixed uniform WTA scaffold already well-matched ->
  **leave --use-vogels OFF for the final run.** (Multi-seed would firm this up.)

**Tooling:** `tune.py` general sequential sweeper (reward_lr/readout_lr/peak_ei/peak_ie),
reports learned + uniform + dead + win_ent. `--peak-ei`/`--peak-ie` exposed in harness.

**Paper / venue:**
- **NeurIPS ruled out** — it's a 9-page conference with a novelty/performance bias;
  SNN+MNIST+~0.75 is a poor fit and would be squeezed. 
- **Recommended: TMLR** (primary — rigor-first, no page limit, ML-community esteem,
  tolerant of MNIST scope if owned); Neural Computation / IEEE TNNLS (topical) as
  alternatives. Nature MI / JMLR high-esteem but poor fit for the scope.
- **Framing:** mechanism + novel architecture + CONTROLS, NOT performance (we're far
  below SOTA ~97–99%). Organize Results by CLAIM, not by chronological phases.
- Two tuning findings feed the paper: over-training degradation (reward_lr) and the
  inhibition/dead-neuron tradeoff (peak_ei).

**Open / next:**
- Final full run (~15k) at the optimal config (reward_lr 5e-6, readout_lr 0.1,
  peak_ei 50; Vogels optional — marginal) → real test-set number + confusion matrices
  + live class-tiled spikes.
- **Full/dense readout option** (Andreas' idea): each class output reads ALL neurons
  (not just its own cluster), delta-rule trained -> strengthens own-class, weakens
  competitors' -> should recover most of the 0.79->0.83 ceiling gap via cross-cluster
  negative evidence. Report the SPECTRUM: uniform (0.65) -> block-diag learned (0.76-79)
  -> full learned (~0.83) -> external LR ceiling (0.83-85). Optional sign-constrained
  middle (own cluster +, others -) for a cleaner bio story. ~10-line change to the
  readout learner (full W_readout N_exc x 10 instead of block-diagonal).
- Consider a 2nd dataset (Fashion-MNIST/CIFAR) to de-risk the venue submission.
- Rework the article structure for TMLR (claims-driven, full-length).

---

## 2026-07-15 — Supervised reward-STDP V1 + tiled per-class architecture

**Focus:** Pivot from unsupervised STDP (which *erodes* the oriented prior) to
**supervised reward-modulated STDP** with class-assigned neurons; then a **tiled
per-class excitatory layout** so the live spike raster is interpretable.

**Landed (commits):**
- Reward-STDP rule: count-product eligibility `#pre x #post` gated by per-neuron
  reward `R_i = +1 target / -1 non-target`, baseline-centered `(2-C)/C`, applied once
  per sample. Kernel `reward_STDP` + `RewardLearner` (`ec2b8fc4`), wired through
  trainer/runner + `snn.learner.RewardSTDP` (`d11c40ad`).
- Interp-harness V1 cell `R1_ori_reward_ff` + pool-by-label readout + coverage
  diagnostics (`b75886f1`); reward-lr tuner (`c49e0527`); live sweep plotter (`ff3a4f57`).
- Per-class confusion matrices (readout + linear) per checkpoint + test (`651b6448`).
- **Tiled architecture** (`99832bbd`): `grouped_excitatory(tiled=True)` -> N_exc=1000,
  10 classes x 10x10 tile; each class = contiguous block whose RF centers tile the FULL
  input on a regular torus offset grid. Bypasses the sheet-grid square asserts
  (non-square N_exc OK). Harness `--tiled`. Verified cover=100%/class.
- **Class-tiled live spike plot** (`0a6bf73e`): exc+inh activity as a 2x5 meta-grid of
  10x10 class tiles + embedded readout + I->E inhibition-received panel + |W_se|.
  Harness `--live-plot`. Verified on real data at N_exc=1000.

**Key decisions:**
- It's *supervised* learning via a policy-gradient-style three-factor rule (not RL/PPO);
  reward signal is label-derived. Judge rules by pool-by-label sample efficiency + stability.
- No decay on eligibility (fixed-length trial, reward at boundary). No trace in the plot
  (we use mean activation).
- V1 = only SE plastic, everything else static; no reward on recurrent weights.
- Tiled RF shape selectable (isotropic default; oriented works). Center grid = regular
  torus offset grid (maximizes MIN spacing = even coverage), not distance-maximization.

**Cross-device note:** parallel work on another machine added the grouped-excitatory
architecture + analysis.py refactor + softmax readout + network-graph plots; merged into
main (`53ec0be8` and around). Nothing lost.

**CONTROLS RESULT (decisive — reward-STDP WORKS):** control suite
(results/rstdp_controls/controls_main) — reward vs reward-off (lr=0) vs
shuffled-labels, 5k images each. Readout (softmax) accuracy: **reward 0.55->0.65,
reward_off flat at 0.11 (chance), shuffle flat at ~0.10 (chance)**. Online train
accuracy same pattern. So the readout gain is genuinely the correct-label reward
signal, NOT Normalize/homeostasis (reward_off at chance) and NOT an artifact
(shuffle at chance). Correction to the earlier "weak/ambiguous" read: the tiled
init is at CHANCE (all class groups tile identically -> identical responses);
reward breaks the symmetry chance->0.65. The full-run "0.58->0.68" looked weak
only because its first checkpoint was already post-1000-images of reward. Effect
is real and large; it saturates ~0.65 vs 0.85 the features support (fitted LR) ->
the FIXED uniform pooling readout is now the bottleneck -> learnable readout next.

**LEARNABLE READOUT WORKS (readout was the bottleneck):** added plastic
cluster->class readout (RewardLearner.w_readout, block-diagonal, softmax delta
rule, --readout-lr; readout_learned_acc metric). 5k-image test
(results/rstdp_controls/readout_test5k): uniform pool 0.64, **learned readout
0.74**, fitted-LR ceiling 0.83. So the learned readout recovers ~half the
0.64->0.83 gap, and learns fast (already 0.69 at first checkpoint). Remaining gap
is the block-diagonal constraint (each class neuron reads only its own cluster,
can't use cross-cluster info). Plastic inhibition available via --use-vogels
(intra-group, block-masked) but not yet tested.

**Open / next:**
- Full interp sweep run (results/interp/) comparing R1 vs B1/B2; **reward-lr (2e-5) needs
  tuning** (`tune_reward_lr.py`).
- Milestone 5 sanity controls: label-shuffle (learning must vanish) + baseline on/off.
- Then variants: V2 (reward + Vogels — makes the I->E panel dynamic), V3/V4 (reward on
  inhibition).
- `random` RF shape under tiling is NOT wired (isotropic/oriented only).
- Watch the inhibitory-activity plot panel on a longer run (was sparse in a tiny untrained
  frame).
