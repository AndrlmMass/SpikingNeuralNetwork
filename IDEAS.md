# Ideas / Backlog

Parking lot for pursuits we're **considering but haven't committed to** — so we don't
forget them. Distinct from `DIARY.md` (what we *did*) and git history (the *how*).
Newest ideas at the top of each section. Mark status: `idea` / `testing` / `adopted` /
`rejected` (with a one-line reason).

---

## Neurodynamics audit vs Diehl & Cook (2015)

Diehl & Cook reach ~95% on MNIST **unsupervised** (STDP + homeostasis), scaling with
neuron count: 100->82.9%, 400->87%, 1600->91.9%, 6400->95%. Our 1000-neuron setup at
~0.80-0.83 UNDERPERFORMS their comparable size -> diagnosable headroom. Source code:
peter-u-diehl/stdp-mnist (Brian). Divergences found (status: audit, fixes not yet made):

| param | Diehl | Ours | impact |
|---|---|---|---|
| synapse model | **conductance** g*(V_rev-V), shunting/self-limiting | **current-based** R*I_syn, V-independent | HIGH |
| membrane tau (exc) | 100 ms | 20 ms | MED-HIGH |
| refractory (exc) | 5 ms | **none** | MED-HIGH (winners fire every step) |
| inhibition | conductance, reversal -100mV (divisive) | current, peak -2 | MED |
| reset | to rest (-65) | -80 (below rest -70) | LOW-MED |
| adaptive theta | tau=1e7, delta=0.05, **frozen@test** | tau=200, delta=0.5, always-on, **reset@eval** | HIGH |
| threshold (exc) | -52 | -55 | LOW |
| dt | 0.5 ms | ~1 ms | LOW |
| input | Poisson ~64Hz max, re-present w/ higher intensity if <5 spikes | rate-coded, no re-present | MED |
| weight norm | sum->78 every iter | Normalize freq 1050 | check |

**Top 3 likely ceilings (independent of theta):** (1) current- vs conductance-based synapses
(no saturation -> runaway winners); (2) no refractory period; (3) fast membrane tau. These
INTERACT with theta: our neurons fire more per sample, so Diehl's delta=0.05 may need tuning.

**Adaptive threshold — same equation, opposite regime.** Both do
`a += -a/tau*dt; a += delta on spike; threshold = default + a` (ours neurons.py:179). Ours
tau=200/delta=0.5 -> decays ~83%/sample = transient within-trial FATIGUE (1-sample memory).
Diehl tau=1e7/delta=0.05 -> ~no decay = permanent LIFETIME accumulation -> equalizes firing
rates across the population -> every neuron a distinct used detector (the 95% ingredient).
Exposed via `--theta-tau` / `--theta-delta` (2026-07-16). TODO for full Diehl fidelity: carry
trained `a` into val/test and FREEZE it (currently resets to zeros + re-adapts, runner.py:569,657).

## Learning rule

### Causal / gated eligibility trace  — status: idea (not sold)
Replace the symmetric rate-product eligibility `e_ij = #pre_i × #post_j` with the
standard pair-STDP form: maintain a decaying pre-trace `x_i(t)` and accumulate it into
`e_ij` **only at the moments post j spikes** (credit the causal drivers, not every input
by its rate).
- **Helps:** sharper, more selective RFs for *active* neurons; more principled / the
  canonical three-factor form reviewers expect.
- **Does NOT help:** dead / undeveloped RFs — those neurons never spike, so they get zero
  eligibility under *both* rules (the update is already post-gated at
  `synapses.py:261`). Dead RFs are a competition/homeostasis problem, not eligibility.
- **Caveat:** with rate-coded MNIST (~350 steps, Poisson-ish) a pre spike is nearly as
  likely just after as just before a post spike, so the causal trace largely *recovers*
  the rate product → expect a sharpening, not a step change. Test, don't bank on it.
- Verify one-thing-at-a-time against a clean baseline.

### Adaptive per-neuron threshold / intrinsic homeostasis (NCG-style)  — status: idea (high leverage)
Per-neuron threshold that rises for frequent winners and falls for silent neurons, so
losers occasionally win. Mechanism from our own NCG reference (Goupy et al.,
arXiv:2410.17066, two-compartment adaptive threshold).
- **Does both open goals at once:** revives dead neurons (they finally win → develop RFs)
  AND forces within-cluster diversity (different neurons pushed to win different samples →
  specialize on different aspects). This is the "inherent, improve-the-WTA" decorrelation
  we wanted — *not* a frequent-winner penalty bolted on.
- Note: plastic inhibition (Vogels iSTDP) is a *related* homeostasis — rate-targeting via
  inhibition instead of thresholds. The inhibitory-LR sweep partly tests this idea.

---

## Readout

### Full / dense readout  — status: idea (motivated by RF-size sweep)
Each class output reads **all** neurons (N_exc×10 plastic, delta-rule), not just its own
block-diagonal cluster.
- **Why now:** the RF-size sweep showed local RFs keep the feature ceiling flat (~0.80,
  refit LR) while the pooled readout craters (uniform 0.69→0.47 as RF shrinks). The info
  is there; the block-diagonal readout can't *compose* parts. A dense readout should
  convert the ~0.80 ceiling into real accuracy with genuine part-detectors.
- Report the spectrum: uniform (~0.65) → block-diag learned (~0.76–0.79) → full learned
  (~0.83?) → external LR ceiling (~0.80–0.85). Optional sign-constrained middle (own
  cluster +, others −) for a cleaner bio story. ~10-line change to the readout learner.

### Unify the readout on R-STDP (drop the delta rule)  — status: idea
The readout currently learns by a **softmax delta rule** (exact SGD on cross-entropy),
while `W_se` learns by reward-STDP. Two different learning rules in one model. The
original commit (`e6b35750`) justifies adding a *plastic* readout ("features support
0.85 fitted but uniform pooling extracts only 0.65 → the readout is the bottleneck")
but records **no rationale for the rule choice** — it went in as the default for a
linear classifier, uncompared.
- **Why the delta rule fits anyway:** (a) the readout is not a spiking layer — it's
  `scores = er @ W` over accumulated counts, so there are no readout spike times and no
  STDP eligibility term exists to compute; (b) `p − onehot` is graded by how wrong the
  prediction was and self-anneals to zero, where R-STDP's ±1 reward is constant-magnitude
  and needs an EMA baseline to centre.
- **Why it's still a liability:** for a paper leaning on biological plausibility, "we use
  gradient descent for the classifier" is the obvious point of attack. State it in the
  methods either way; don't let a reviewer find it.
- **The unified alternative:** make the readout a spiking layer and use reward-modulated
  STDP from exc → readout neurons. One rule throughout. Cost: slower, noisier convergence
  and a weaker teacher signal. Worth an ablation if a reviewer pushes, not before.

---

## Regularization

### `reg_frequency` is dead config in every R-STDP run  — status: idea (test next)
`trainer.py:380` gates the `reg_frequency` timer behind `and not self.use_reward`, so it
**never fires** in reward runs. Instead `trainer.py:605` normalizes `W_se` inside the
sample-boundary block, and `Normalizer` has no internal frequency check of its own.
- **Net effect:** R-STDP normalizes once **per image**; `reg_frequency=1050` is ignored.
  That is 3x more often than configured (1050 / 350 steps = every 3 images).
- **Why it may matter a lot:** per-neuron normalization holds each neuron's total input
  weight constant. Applied after *every* image it directly opposes reward-STDP's attempt
  to grow some weights relative to others — a plausible cause of class selectivity sitting
  flat at ~0.37 regardless of what else we change. Explains the flatness better than
  anything on the inhibition axis.
- **Test:** sweep normalization frequency in the reward path (every 1 / 3 / 10 / 50 images,
  and off) against selectivity trend + accuracy. Queued behind the inhibition sweep.

---

## Architecture / receptive fields

### Smaller / heterogeneous RF sizes  — status: testing (sweep run 2026-07-16)
Structural RF footprint set by `sigma_se` (now exposed via `--sigma-se` /
`--sigma-se-lognormal`). Baseline 3.0 ≈ near-global whole-digit templates (tiling is
cosmetic); ~1.5 → 6px local patches; ~1.0 → 3.6px tight patches.
- **Sweep finding (single seed, `results/rstdp_rfsize/rfsize_main/`):** smaller RFs
  preserve the feature ceiling, improve health (dead 0.56→0.31, winners spread), and make
  RFs interpretable/compositional — but need a compositional readout to pay off. Pairs
  naturally with the dense readout above.
- Heterogeneous (lognormal) sizes = a mix of local-part + holistic detectors; defensible
  biological ablation.

### Non-uniform / structured lateral inhibition  — status: idea
Current intra-group I→E is static and uniform (inhibit all neighbors equally, peak −2).
Suspect suboptimal. Options: (a) plastic inhibition (Vogels — sweeping LR now), (b)
inhibition scaled by RF *similarity* (neurons suppress group-mates they resemble → pushes
them apart = inherent decorrelation), (c) distance/overlap-weighted inhibition.

---

## Evaluation / rigor

- Multi-seed runs for every ablation before any claim goes in the paper (all current
  findings are single-seed).
- The `rf_diversity` metric is **confounded** for the tiled arch — it's cosine similarity
  of raw RF columns, so it mechanically drops as RFs shrink (less spatial overlap)
  regardless of feature content. Read the visual RF grids + accuracy as the real signal.
