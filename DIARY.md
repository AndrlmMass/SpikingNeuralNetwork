# Project Diary

Running log of work sessions, **newest first**. Purpose: fast re-entry across
devices/sessions — what was done, why, what landed (commits), and what's next.
Complements git history (the detailed "what") and any tooling notes.

**Convention:** add a `## YYYY-MM-DD — <short title>` section at the TOP for each
session. Capture: focus, key decisions, commits (hashes), and open items / next steps.
Keep it scannable — a few bullets, not a transcript.

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
