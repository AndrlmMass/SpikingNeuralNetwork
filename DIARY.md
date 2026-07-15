# Project Diary

Running log of work sessions, **newest first**. Purpose: fast re-entry across
devices/sessions — what was done, why, what landed (commits), and what's next.
Complements git history (the detailed "what") and any tooling notes.

**Convention:** add a `## YYYY-MM-DD — <short title>` section at the TOP for each
session. Capture: focus, key decisions, commits (hashes), and open items / next steps.
Keep it scannable — a few bullets, not a transcript.

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

**Open / next:**
- Full interp sweep run (results/interp/) comparing R1 vs B1/B2; **reward-lr (2e-5) needs
  tuning** (`tune_reward_lr.py`).
- Milestone 5 sanity controls: label-shuffle (learning must vanish) + baseline on/off.
- Then variants: V2 (reward + Vogels — makes the I->E panel dynamic), V3/V4 (reward on
  inhibition).
- `random` RF shape under tiling is NOT wired (isotropic/oriented only).
- Watch the inhibitory-activity plot panel on a longer run (was sparse in a tiny untrained
  frame).
