# Project Diary

Running log of work sessions, **newest first**. Purpose: fast re-entry across
devices/sessions — what was done, why, what landed (commits), and what's next.
Complements git history (the detailed "what") and any tooling notes.

**Convention:** add a `## YYYY-MM-DD — <short title>` section at the TOP for each
session. Capture: focus, key decisions, commits (hashes), and open items / next steps.
Keep it scannable — a few bullets, not a transcript.

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
