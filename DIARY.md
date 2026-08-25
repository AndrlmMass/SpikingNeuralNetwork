# Project Diary

Running log of work sessions, **newest first**. Purpose: fast re-entry across
devices/sessions — what was done, why, what landed (commits), and what's next.
Complements git history (the detailed "what") and any tooling notes.

**Convention:** add a `## YYYY-MM-DD — <short title>` section at the TOP for each
session. Capture: focus, key decisions, commits (hashes), and open items / next steps.
Keep it scannable — a few bullets, not a transcript.

---

## 2026-08-25 — HPC sweeps landed: RF geometry plots, a decoder naming trap, and a Results-plan rethink

**Focus:** Pull the two finished HPC sweeps, build the 2D-Gaussian RF geometry plots
Hubin asked for, then stress-test the proposed Results-section ordering against the
numbers. Two scripts added to `experiments/RF_article/interp/`:
`plot_rf_geometry.py` (appendix figures) and `plot_coherence_retention.py` (main-text
figure). Outputs in `results/interp/` (gitignored).

**Both sweeps are complete**: phase 1 `run_20260819_112254` 175/175, phase 2
`run_20260819_112250` 50/50, every run with a finite `test_acc`, 5 seeds/cell, 3 epochs.
HPC paths are `/mnt/users/andreama/projects/biosnn4/experiments/RF_article/interp/
{phase1_ablation,mnist_family_sweep}/results/<RUN_ID>/<tag>/`.

### NAMING TRAP — there are THREE decoders and `test_acc` is none of the good ones

Same class of trap as the `val_phi` one from 08-10, and it bit the whole first pass of
this session's analysis:

| key                        | what it actually is                                       |
| -------------------------- | --------------------------------------------------------- |
| `test_acc`                 | the **PCA+LR evaluator** probe — in BOTH phases            |
| `test_lin_acc`             | L1-LR probe, fit on 5000 train features                    |
| `test_cm_readout`          | the **pooled/softmax** readout (mnist 0.760) — NOT learned |
| `risk_coverage.csv` `base_acc` / traj `readout_learned_acc` | the **learned delta readout** — the article's number |

MNIST oriented, same runs: 0.884 (pca probe) / 0.935 (L1) / 0.760 (pooled) / **0.9455
±0.0067 (learned readout, full 10k test, 5 seeds)**. The abstract's 95.5 is the
online/val figure; **the test-set number across seeds is 94.6** — reconcile before
submission. Any phase1-vs-phase2 comparison must name its decoder, and phase 1 has no
learned readout at all.

### RF geometry (Hubin request) — what the moments add, and what they do not

`rf_gaussian_moments` was already logging var_x / var_y / cov_xy / elongation at every
checkpoint; only the aggregation was missing. Findings:

**(1) Coherence and the moments dissociate — this is the payoff.** MNIST trace-STDP
drops coherence 33% (0.710 -> 0.479) *before the first logged checkpoint* while
`var_x` moves 1.7% (4.06 -> 3.99). Structure-tensor coherence reacts to
high-spatial-frequency weight noise; the mass moments track the RF **envelope**.
Reporting coherence alone overstates how fast the geometry itself moves.

**(2) The erosion is purely feedforward.** `ee_off`, `ie_off` and `vogels` change final
coherence by **<1e-3** — three orders below the erosion itself. Not a null run: the
ablations verifiably bite (`cur_ee`->0, `cur_ie`->0, `cur_ie`->41.8 respectively) and
move accuracy by up to 1.8pp. The recurrent circuit changes the **responses** and
leaves `W_se` alone. One sentence in Results, not a figure.

**(3) Coherence retention is the number that carries the claim** (init 0.710, from the
`frozen` cell — the only no-plasticity view of t=0):

| condition                  | final coh | retained | per-dataset |
| -------------------------- | --------- | -------- | ----------- |
| oriented RF + trace-STDP   | 0.382     | **54%**  | 44–58%      |
| oriented RF + R-STDP       | 0.488     | **69%**  | 65–71%      |
| oriented RF + triplet-STDP | 0.658     | **93%**  | 87–97%      |
| random weights (floor)     | 0.081     | n/a      | 0.022–0.160 |

Dose-response: the rule that retains most prior (triplet, 93%) is also the best plastic
rule. **Every plastic rule still loses to frozen** (base_ori -1.27 to -3.86pp paired).

**(4) DROPPED CLAIM — "eroded vs expanded" does not survive.** Measured from the true
init, RF size *grows* under trace-STDP on 4 of 5 datasets (-9% MNIST, +6 to +36%
elsewhere). Both rules grow the envelope; only coherence separates them. The framing is
**eroded vs largely preserved**, not eroded vs expanded.

**(5) `rf_elongation` is broken — do not plot it.** sqrt(l_max/l_min) with l_min floored
at 1e-12, so any near-collinear RF blows up: every random-prior run reports ~2e4 where
the truth is ~1. Figures use a derived bounded `rf_anisotropy` =
sqrt((vx-vy)^2+4cxy^2)/(vx+vy) instead. Anisotropy reads near-zero **by construction**
because it is computed on the tensor-of-means and near-uniform preferred orientations
cancel — the informative version is the mean of *per-neuron* anisotropies, which is what
elongation was meant to be. Not recoverable from these runs (raw `W_se` not saved);
fix in `analysis.py` before the next sweep.

**(6) The first checkpoint already contains plasticity.** ~2/3 of the MNIST-family
coherence loss happens before it. SVHN is the exception and erodes gradually (0.694 at
ckpt 0 -> 0.311). **Do not claim erosion rates** off these runs; next sweep needs a
checkpoint before the first weight update.

### Results-section plan — pushed back on 3 of 6 steps

Proposed order was: RF-vs-random -> supervised beats unsupervised -> because the prior
erodes -> abstention -> OOD. What the numbers say:

**"Supervised is better across the board" is FALSE.** Best unsupervised condition is
`frozen` on **all five** datasets. Against it (learned readout vs frozen's L1 probe):
MNIST +1.78, KMNIST +3.46, SVHN +3.70, **FMNIST -5.66, notMNIST -8.53**. Also biased by
construction — max-over-7-conditions vs max-over-2 — and phase1->phase2 changes **six
things at once** (rule, `grouped`, `n_exc` 1024->1000, `peak_ei` 20->50,
`center_margin` 0->4, readout), so nothing is attributable to supervision. Contradicts
our own "phase 1 diagnostic / phase 2 confirmatory" design. A model-level claim has to
come from the beta-binomial with model as a factor (§2.6.3 equation still empty).

**"Supervised wins BECAUSE unsupervised erodes the prior" does not connect.** R-STDP
erodes too (69% vs 54%) — erosion is not what separates the models. The defensible
causal claim is entirely **within phase 1**: plasticity erodes the prior AND every
plastic rule loses to frozen, with triplet retaining most and performing best.

**DRAFT §3.3 IS CONTRADICTED.** "around 99% accuracy or more ... across the five
datasets" — coverage at 99% accuracy is MNIST 0.838, notMNIST 0.179, KMNIST 0.162,
**FMNIST 0.009, SVHN 0.000**. The abstention result is a **MNIST result**. Fix the
sentence.

**RF-vs-random is the strongest opener, and the sign differs by phase** — which is more
interesting than the plan assumed:

| dataset  | phase 1 (probe) | phase 2 (probe) | phase 2 (learned readout) |
| -------- | --------------- | --------------- | ------------------------- |
| mnist    | +0.50 ±0.52     | +5.77 ±0.74     | +3.02                     |
| fmnist   | **+2.32** ±0.15 | **-1.05** ±0.40 | **-3.20**                 |
| kmnist   | +4.34 ±0.92     | +5.93 ±1.17     | +3.39                     |
| notmnist | **+1.52** ±0.64 | **-2.80** ±0.73 | **-6.31**                 |
| svhn     | +2.24 ±0.62     | +12.49 ±0.37    | +6.53                     |

Under trace-STDP the prior helps on **all five**; under R-STDP it reverses on FMNIST and
notMNIST. The abstract currently describes only the phase-2 pattern. **Phase-2 signs
replicate the 08-10 figure** (3 seeds/5 epochs: SVHN +7.3, MNIST +4.2, KMNIST +2.2,
notMNIST -5.2, FMNIST -8.4) — independent runs, same ordering.

### Proposed Results order (organise by FACTOR, not by model)

1. Does the prior help? (both phases, one relative plot — the table above)
2. Does learning help? No: frozen beats every plastic rule, all 5 datasets
3. Why: the prior is eroded (retention table + ablations in one sentence)
4. Supervision changes the sign: R-STDP retains 69% **and is the only configuration
   with a working internal readout at all** — that is the honest supervised claim, not
   "higher accuracy"
5. Abstention (scoped to MNIST, with the coverage table)
6. OOD MNIST->FMNIST

### Open items

- [ ] Re-run one phase-1 cell **without `--no-plots`** (`run_slurm.sh:178`) — phase 1 has
      no `rf_first/rf_last` images, so there is no before/after RF picture for the
      unsupervised model. One run.
- [ ] Phase 2 has **no frozen control** and differs in 4 config params, so its "69% of
      0.710" borrows phase 1's init. Instantiate the phase-2 model and measure coherence
      at init with no training — seconds, makes the headline comparison clean.
- [ ] Fix `rf_elongation` -> bounded per-neuron anisotropy in `analysis.py`.
- [ ] Reconcile abstract 95.5 vs test-set 94.6; fix draft §3.3 coverage sentence.
- [ ] Write the beta-binomial spec for the RF-vs-random effect (§2.6.3).

---

## 2026-08-10 — Article framing, the Phase-2 dataset gap, and a metric audit

**Focus:** Decide what the article actually claims, then test the claim against the
saved runs. Two analyses written and run this session (scratchpad only, not yet in
the repo — promote to `experiments/` if we re-run): `metric_corr.py` /
`metric_corr2.py` (which metrics predict accuracy, 60 runs with trajectories) and
`data_stats.py` (data-side statistics vs the RF-minus-random gap).

### Framing decision — "does biological realism pay?"

Chosen over "we close the SNN performance gap", which the numbers do not support
(Goupy et al. 2024: MNIST 98.59 / FMNIST 87.12 / CIFAR-10 62.81 against our 95.5 /
~75 / ~21). Realism becomes an empirical variable measured on three axes —
structure (RF vs random, oriented vs isotropic), rule (frozen / trace-STDP /
R-STDP), task (5 datasets) — with the mechanism claim ("the prior needs a rule that
protects it") as the spine and selective prediction as a headline result rather than
the frame. **Abstract rewritten accordingly and now in the draft.** The old SOTA-parity
sentence is gone.

### Phase-2 RF vs random: the gap changes SIGN across datasets

Learned-readout test accuracy, 3 seeds, 5 epochs. Deltas are as printed on the figure;
absolute values read off the boxplots and therefore approximate.

| dataset  | RF    | random | delta (pp) |
| -------- | ----- | ------ | ---------- |
| SVHN     | ~23.5 | ~16.5  | **+7.3**   |
| MNIST    | ~95   | ~90.5  | **+4.2**   |
| KMNIST   | ~85.5 | ~83.3  | **+2.2**   |
| notMNIST | ~81   | ~86    | **-5.2**   |
| FMNIST   | ~70.5 | ~78.8  | **-8.4**   |

SVHN is above chance in both conditions (chance 10), so the +7.3 is not a
near-chance artefact — but it is a gap between two weak models and must be reported
as such. **Raw Phase-2 outputs are not on this machine**; only the figure. Same for
notMNIST generally — `data/datasets/notmnist/image_cache` is empty and the dataset
sweep script does not accept `notmnist` in `--dataset` choices.

### What predicts the gap: orientation content, NOT scale

Ran the project's own `orientation_coherence` on the *images* instead of the weight
columns, plus stroke-width / fill / registration statistics (2000 imgs each, same
grayscale-28x28-[0,1] pipeline):

| dataset | gap  | img orient coh | stroke width px | fill frac |
| ------- | ---- | -------------- | --------------- | --------- |
| SVHN    | +7.3 | 0.395          | 4.36            | 0.371     |
| MNIST   | +4.2 | 0.347          | 2.59            | 0.153     |
| KMNIST  | +2.2 | 0.262          | 2.54            | 0.215     |
| FMNIST  | -8.4 | 0.235          | 4.08            | 0.355     |

**Image-space orientation coherence orders all four perfectly (Spearman +1.00,
Pearson +0.857).** The stroke-width / registration hypothesis is NOT supported
(+0.40; SVHN has the widest strokes and the largest gain). n=4 with 7 statistics
tested, so this is hypothesis-generating, not a finding — and **notMNIST is the
decisive missing point**, since it is the case that intuitively should break the
orientation story. Regenerating that cache is the single highest-value open item: it
either yields a data-side predictor of when the prior pays (computable before
training) or kills the idea.

### Metric audit — grouped eta2 works, the structural metrics do not

**(0) THE ONE THAT WORKS: grouped eta2 (`val_phi`).** Three different quantities are
easy to confuse and we had been conflating them: trajectory `eta2` is **per-neuron**
`class_eta_squared` (the `class_selectivity` replacement), `val_phi` is the evaluator's
`Phi`, and `eta_squared` on raw rates is the one whose own docstring warns it
anti-correlates with accuracy under WTA. Correlating the right one:

|                                           | vs learned acc | vs pool acc | vs refit ceiling |
| ----------------------------------------- | -------------- | ----------- | ---------------- |
| grouped eta2 (`val_phi`), 60k within-run  | **+0.749**     | +0.130      | -0.224           |
| grouped eta2, rfgeom between-design (n=6) | **+0.657**     | +0.314      | +0.029           |
| grouped eta2, all runs logging it (n=44)  | **+0.334**     | +0.345      | —                |
| per-neuron eta2, 60k within-run           | -0.604         | -0.122      | +0.280           |
| per-neuron eta2, rfgeom (n=6)             | -0.371         | +0.829      | +0.543           |

`val_phi` rises 0.178 -> 0.275 over the 60k run and is the only representation metric
with the right sign in every regime. Per-neuron eta2 carries the **opposite** sign
against learned accuracy, exactly as `group_eta_squared`'s docstring predicts.

**Clean division of labour that follows:** grouped eta2 tracks **achieved** accuracy
(it is measured in the pooled space the readout uses), while PR and the correlation
pair track the **decodability ceiling**. They are complementary, not competing — two
metrics with two different jobs, which is a much simpler story than the panel we had.

**NAMING TRAP — `val_phi` is TWO different quantities** (same class of trap as
`test_acc` being the pca_lr evaluator). `Phi.score` (`_evaluation/evaluation.py:34-40`)
branches on `group_assignment`: **grouped** runs get `group_eta_squared` (the article's
metric), **ungrouped** runs get raw multivariate `eta_squared` — the one that
anti-correlates with accuracy under WTA. `interp_harness.py:291-295` only sets
`group_assignment` when `a.grouped`, so **every trace-STDP mechanism cell (A1/A2/B1/B2/B3)
logs raw eta2 under the name `val_phi`**, and its rise (B1: 0.159 -> 0.230) is NOT
comparable to the reward runs' grouped values. The correlations in the table above are
all from runs with a learned readout, i.e. grouped, so they are genuinely grouped eta2
and stand — but any trace-vs-reward comparison on `val_phi` from existing runs would be
wrong. **To get grouped eta2 for the erosion-vs-bending contrast, re-run trace-STDP with
`--grouped`** (the harness supports it; grouping is independent of rule, per the comment
at `interp_harness.py:289`). Cheap run, no code change.

**(1) The structural metrics flip sign depending which knob you turn.** Spearman vs
learned accuracy, per sweep family:

| metric       | rfgeom | rfsize | inhib2d | inhib-lr | theta |
| ------------ | ------ | ------ | ------- | -------- | ----- |
| dead_frac    | -0.20  | +0.40  | -0.53   | +0.80    | +1.00 |
| orient_coh   | +0.31  | +0.60  | +0.64   | -0.80    | +1.00 |
| rf_diversity | +0.14  | +0.20  | +0.64   | -0.80    | +0.80 |
| selectivity  | +0.60  | +0.40  | -0.57   | +0.50    | -0.40 |

Nothing holds its sign. **You cannot tune for any of these and expect accuracy to
follow** — grouped eta2 above is the exception. Answers Domantas' open question from
the 05.08 meeting, but is a metrics audit rather than a claim about SNNs, so it
belongs in a footnote justifying the panel, **not** in Results (decided 2026-08-10:
the correlation tables confuse more than they earn).

**(2) Within-run correlation against accuracy measures TIME, not quality.** 60k run,
60 checkpoints — everything correlates with readout accuracy simply because accuracy
rises monotonically as the readout learns. Against the refit probe it collapses:

| metric              | start -> end   | vs readout acc | vs refit probe |
| ------------------- | -------------- | -------------- | -------------- |
| participation ratio | 32.05 -> 33.70 | +0.681         | -0.266         |
| orient coherence    | 0.667 -> 0.492 | -0.758         | +0.247         |
| rf_diversity        | 0.090 -> 0.137 | +0.753         | -0.248         |
| per-neuron eta2     | 0.169 -> 0.161 | -0.604         | +0.280         |
| **selectivity**     | 0.318 -> 0.293 | -0.536         | **-0.000**     |

Selectivity at exactly 0.000 against decodability confirms the firing-rate confound
outright. **Rule: report representation metrics against `refit_acc`, never against
readout accuracy.**

**(3) Representation metrics predict the CEILING, not the achieved accuracy.** Across
the 6 rfgeom cells: `pr` +0.257 vs learned but **+0.600 vs refit ceiling**;
`corr_within`/`corr_all` -0.600 vs ceiling; `orient_coh` +0.657. The `dom` cell is the
vivid case — **highest ceiling of all six (0.867) and the lowest achieved accuracy
(0.683)**. This is the "prior sets the ceiling, rule + readout climb to it" thesis,
now measured across designs rather than within one run. Vindicates Andreas' PR
intuition, but about the ceiling.

Only **9 of 60 runs** log >=5 of `pr / pr_cov / eta2 / corr_within / corr_all /
dead_frac / orient_coh` (the 60k run, spiking_15k, run15k, and the 6 rfgeom cells).
Everything else predates the instrumentation — which is why the pooled between-design
analysis had to fall back on the older keys.

### Spiking readout — keep it, as one sentence

`results/rstdp_spiking/spiking_15k`: pool **0.725** -> spiking readout **0.811** ->
delta readout **0.915** (refit probe 0.837). Better than pooling by 8.6pp, worse than
the delta rule by 10.4pp, as remembered. The number that explains it:
`spiking_tie_frac = 0.126` — an eighth of trials tie, an intrinsic consequence of
decoding integer spike counts, and the real reason it trails. **Decision:** three
numbers in one sentence of Results + an appendix subsection, not a section. It
pre-empts the strongest attack on the paper ("you used gradient descent for the
classifier") for almost no space.

### Metric panel — FINAL, organised by claim (supersedes the 05.08 cut)

Metrics are assigned to the claim they answer, each with the question it is there to
answer. Nothing appears twice.

**Claim 1 — "does biological realism pay?" OUTCOME METRICS ONLY.**

| metric                | question it answers                                                                                                                                    |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| readout accuracy      | how well does the network's **own** online decision rule classify held-out items?                                                                      |
| linear-probe accuracy | how much class structure is linearly decodable **at all**, whether or not the readout finds it? (external control; log `n` — data-starved below ~2000) |

Two things to keep straight. The dense readout reads **all 1000 excitatory neurons**
(positive evidence for its class, negative against competitors) — *not* its own class
pool; that is the block-diagonal design we replaced, and the difference is the whole
0.79 -> 0.955 ladder, so "extracted from the class pools" would be wrong. And neither
number alone is the claim: **the metric for Claim 1 is the RF-minus-random delta at
matched density** — accuracy is the measurement, the contrast is the claim.
**No representation metric here** — see the placement note below.

**Claim 2 — "the prior fixes the representation, but only under the right rule."**
Ordered as a chain (capacity -> health -> redundancy -> prior survival -> usable class
structure -> did it actually help), not a list:

| metric                                           | question it answers                                                                                                                                                        |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| participation ratio vs the K-1 threshold         | is there enough dimensionality? (need >=9 for 10-way; isotropic sits at 4.3, oriented+margin at 11.8)                                                                      |
| dead fraction                                    | are the units alive to carry it? (0.43 -> 0.00)                                                                                                                            |
| correlation pair, within vs overall as a **gap** | are the live units carrying different signals, or the same one? (0.121 vs 0.113 = group membership buys almost nothing, which is *why* pooling gets 0.756 and dense 0.955) |
| orientation coherence, **BOTH rules**            | did the prior survive the rule? (trace 0.71->0.39 erosion vs R-STDP 0.667->0.492 bending)                                                                                  |
| grouped eta2                                     | did any of that become class structure the readout can use? (0.178 -> 0.275 under R-STDP)                                                                                  |
| drift, fixed vs refit probe                      | does the code keep changing — and does any of that change make it more **decodable**? (fixed 0.797->0.597 while refit holds flat ~0.83; answer: no)                        |

Two notes on this panel. Orientation coherence must appear for **both** rules — the
result *is* the contrast, and reporting only the trace-STDP side leaves half a
comparison and no evidence that rule choice is what matters. And drift must be reported
as the **pair**: the frozen-probe decay alone reads as a stability complaint, whereas
frozen-decays-while-refit-holds-flat is the evidence for "the prior set the ceiling and
the rule climbs to it", which is what Claim 2 ultimately argues.

**Claim 3 — "a local learner that knows when it doesn't know."**
Also a chain, not a list — the three are not parallel (a statistic, a way of evaluating
a statistic, and an operational payoff):

| metric                                    | question it answers                                                                                   |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| entropy (perplexity as the readable unit) | how confident is the network, and how many classes remain in contention?                              |
| AUROC of entropy and margin               | do these statistics separate correct from incorrect items **at all**, before any threshold is chosen? |
| risk-coverage                             | what does that buy operationally — how much error is removed per unit of coverage given up?           |

**Terminology to get right: risk is the ERROR rate on the accepted set** (1 - selective
accuracy), not accuracy — and the curve *characterises* the trade-off rather than
locating an optimum, since there is no optimum without pricing an abstention.
**Perplexity is a monotone rescaling of entropy**, so the two give *identical* abstention
decisions: report entropy as the statistic and perplexity as the interpretable unit, not
as two pieces of evidence. **Margin beats entropy empirically** (0.726 vs 0.637 on the
network-native readout, 0.888 vs 0.873 on the probe) — entropy is the one to *report*
(Ellingsen baseline, Hubin's ask), margin is the one to *use*.
Ship the caveat with the number: only *shape* statistics work
(entropy/perplexity/margin/maxp ~0.94) while `total_rate` and `topk_sum` sit at
0.526-0.528, indistinguishable from chance.

**PLACEMENT NOTE — why grouped eta2 sits in Claim 2, not Claim 1.** It is measured on
the representation but in the *readout's* coordinate system, which is why it tracks
achieved accuracy (+0.657 between designs) and not the ceiling (+0.029). Per Domantas'
own argument on 05.08 — if clustering tracks accuracy closely it tells us little that
accuracy did not — putting it beside accuracy in Claim 1 reads as saying the same thing
twice. In Claim 2 it is the terminal link that connects the representation story back to
the accuracy story, which is the join Claim 2 needs in order not to read as a detour.

**Cut:** `selectivity` (confounded, deprecated in code), `rf_diversity` as a swept
quantity (falls mechanically as RFs shrink), Gini (not implemented; duplicates
dead_frac + w_floor_frac), `winner_entropy`, `frac_ever_winner`, `w_floor_frac`,
`share_ce`, `brier`, `cur_*`. Also cut the metric-vs-accuracy correlation tables from
Results — they are a metrics audit, not a claim about SNNs, and belong in a footnote
justifying the panel.

### Draft errata found while reading

- R-STDP subsection says "we train the readout" but indexes j as the excitatory unit
  — R-STDP trains `W_se`, the readout is the delta rule. The whole
  biological-plausibility argument turns on this split.
- Oriented-RF exponentials still missing their minus signs; the rotation equation now
  defines only `x'` and has **lost `y'` entirely** (regressed vs the previous version).
- Phase-2 text says Vogels iSTDP is a ladder rung; Figure 3 shows "Pool-layout" in that
  slot. Text and figure disagree. (Agreed to drop the Vogels *method* subsection but
  keep the finding as one sentence + appendix — it is step 4 of the agreed red thread.)
- `rfgeom` learned accuracies disagree between `summary.json` (base 0.739 /
  thin_margin 0.819) and trajectory final checkpoint (0.717 / 0.785). **Resolve which
  is canonical before quoting either.**

**Open / next:**
1. **Regenerate the notMNIST cache and complete the 5-dataset predictor test** —
   the one item standing between this and a Section 3 that makes a real claim.
2. Get the Phase-2 raw outputs onto the repo (currently figure-only).
3. Run the 5-dataset sweep on the 4-layer supervised model (the older sweep is the
   3-layer trace-STDP net, so the scope claim currently rests on a different model).
4. Recurrence on/off under STDP for the mechanistic erosion explanation (05.08 task).
5. **Re-run trace-STDP with `--grouped`** so grouped eta2 exists on both sides of the
   erosion-vs-bending contrast (currently R-STDP only — see the naming trap above).
6. Fix the draft errata above.
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

| dataset  | oriented    | random      | delta    | eta2 (val, final ckpt) |
| -------- | ----------- | ----------- | -------- | ---------------------- |
| mnist    | **0.8656**  | 0.8215      | **+4.4** | 0.167 / 0.177          |
| kmnist   | **0.7465**  | 0.7044      | **+4.2** | 0.117 / 0.101          |
| fmnist   | 0.7194      | **0.7363**  | -1.7     | 0.295 / 0.299          |
| notmnist | 0.8177      | **0.8363**  | -1.9     | 0.224 / 0.169          |
| svhn     | 0.2658      | *running*   | —        | 0.013                  |
| cifar10  | *see below* | *see below* | —        | 0.052 / 0.038          |

**SINGLE SEED — do not quote yet.** Seed 1 agrees on mnist so far (oriented 0.8788;
random still running, val 0.807 vs oriented's 0.890). Read provisionally: the oriented
prior helps on the digit-like sets and slightly *hurts* on the two harder ones. If that
survives seeds 1-2 it is a real and reportable boundary on the prior's usefulness — not
the clean win, but a more honest one. eta2 tracks accuracy across datasets in the
expected direction (fmnist 0.29 down to svhn 0.013).

### THE BUG — merged-pool splits silently produced an EMPTY test set

`ImageDataStreamer` **merges** torchvision's train and test splits into one pool and
carves train/val/test from it, so the budget is `len_train + len_test`, NOT `len_train`:

| dataset                 | pool                | requested 59000/1000/10000 |
| ----------------------- | ------------------- | -------------------------- |
| mnist / fmnist / kmnist | 70000               | fits **exactly**           |
| svhn                    | 99289               | fits                       |
| **cifar10**             | **60000** (50k+10k) | **test = 0**               |

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

| prior    | epochs 2-5 buy | range          |
| -------- | -------------- | -------------- |
| oriented | **+0.03pp**    | -0.57 to +0.70 |
| random   | **+0.46pp**    | +0.18 to +0.66 |

Both inside the noise floor. **~80% of every 23h cell buys nothing measurable in
accuracy.** Consistent with 07-25's "refit probe is FLAT — decodability did not improve";
this is the same result seen from the accuracy side. Note training is still *doing*
something (eta2 moves monotonically below, and drift was still widening at epoch 5) — it
just does not cash out.

### eta2 moves in OPPOSITE directions for the two priors (the good finding)

Matched on dataset AND seed (n=4 pairs: fmnist/kmnist/mnist/notmnist, all s0), eta2 from
epoch 1 -> epoch 5:

| dataset  | oriented                    | random                      |
| -------- | --------------------------- | --------------------------- |
| fmnist   | 0.312 -> 0.295 (**-0.018**) | 0.269 -> 0.299 (**+0.030**) |
| kmnist   | 0.119 -> 0.117 (-0.003)     | 0.087 -> 0.101 (+0.014)     |
| mnist    | 0.171 -> 0.167 (-0.004)     | 0.146 -> 0.177 (+0.031)     |
| notmnist | 0.232 -> 0.224 (-0.008)     | 0.159 -> 0.168 (+0.010)     |
| **mean** | **-0.008**                  | **+0.021**                  |

**4/4 negative and 4/4 positive — perfect separation.** The oriented prior *front-loads*
per-neuron selectivity and training erodes it; random starts lower, builds it up, and
**overtakes** (fmnist 0.299 vs 0.295, mnist 0.177 vs 0.167). This is the **random control
for the "reward-STDP bends the prior" claim** from 07-25 — the counterfactual says the
erosion is specific to having a structured prior to erode. Seed check: mnist_oriented has
two seeds and the delta reproduces at -0.005 / -0.004, so seed noise on this quantity is
~0.001, well clear of the +-0.02-0.03 contrast.

**The accuracy gap is front-loaded too, and shrinks** (oriented - random, pp):

| dataset  | 0.2ep     | 1ep   | 3ep   | 5ep       |
| -------- | --------- | ----- | ----- | --------- |
| kmnist   | +8.70     | +5.90 | +6.50 | **+2.90** |
| mnist    | +8.30     | +5.40 | +5.40 | **+5.50** |
| fmnist   | +0.00     | -0.50 | -1.70 | **-2.00** |
| notmnist | -1.93     | -4.20 | -2.93 | **-1.87** |
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

| readout               | test acc             |
| --------------------- | -------------------- |
| signs free            | **0.9397 +- 0.0033** |
| non-negative (w >= 0) | **0.9336 +- 0.0043** |
| gap                   | **0.61 points**      |

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

| decoder                             | test acc   | what it is                                                     |
| ----------------------------------- | ---------- | -------------------------------------------------------------- |
| **learned readout (dense, online)** | **0.9551** | the model's own answer — 1000x10, softmax delta, 300k samples  |
| linear probe (`test_lin_acc`)       | 0.8811     | L1-LR fit on **1000** val images — **data-starved, see below** |
| `pca_lr` evaluator (`test_acc`)     | 0.8647     | harness Evaluator, scaler+PCA+LR                               |
| uniform pool                        | 0.7557     | fixed block-diagonal pooling                                   |
| online train decisions              | 0.9546     | matches the readout — no train/test gap                        |

### The 7-point learned-vs-linear gap is a CONTROL artefact, not a representation fact

`interp_harness.fit_clf` fits ~10k parameters on **700–1000 samples in 1000-D** (p ~= n).
Refitting the *byte-identical* probe on the frozen final features, varying only n:

| n_train   | 700   | 1000  | 2000  | 4000  | **7000**   |
| --------- | ----- | ----- | ----- | ----- | ---------- |
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

| metric                         | start    | end      | delta                                  |
| ------------------------------ | -------- | -------- | -------------------------------------- |
| orientation coherence          | 0.667    | 0.492    | **-26%**                               |
| within-group RF diversity      | 0.090    | 0.137    | **+52%**                               |
| w_floor_frac (pruned synapses) | 0.024    | 0.148    | +14.8pp                                |
| rf_mean_cosine                 | 0.097    | 0.125    | +29%                                   |
| per-neuron eta2                | 0.169    | 0.161    | -5%                                    |
| grouped eta2 (val_phi)         | 0.178    | 0.275    | **+55%**                               |
| pop_sparseness                 | 0.485    | 0.552    | +14%                                   |
| participation ratio            | 32.1     | 33.7     | +5%                                    |
| w_se_mean                      | 0.330175 | 0.330175 | **exactly constant** (L1 norm holding) |

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
| config       | refit-LR ceiling | learned | uniform pool | dead | win_ent |
| ------------ | ---------------- | ------- | ------------ | ---- | ------- |
| baseline_3.0 | 0.805            | 0.788   | 0.686        | 0.56 | 0.36    |
| local_1.5    | 0.797            | 0.778   | 0.575        | 0.37 | 0.40    |
| tight_1.0    | 0.800            | 0.747   | 0.471        | 0.31 | 0.41    |
| hetero_2.0ln | 0.825            | 0.772   | 0.344        | 0.78 | 0.13    |
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
| vogels_lr    | learned | uniform | dead | win_ent | ceiling |
| ------------ | ------- | ------- | ---- | ------- | ------- |
| off (static) | 0.783   | 0.690   | 0.56 | 0.356   | 0.812   |
| 0.005        | 0.790   | 0.677   | 0.55 | 0.357   | 0.802   |
| 0.02         | 0.778   | 0.676   | 0.53 | 0.367   | 0.803   |
| 0.05         | 0.770   | 0.689   | 0.52 | 0.391   | 0.805   |
| 0.1          | 0.760   | 0.656   | 0.50 | 0.402   | 0.798   |
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
