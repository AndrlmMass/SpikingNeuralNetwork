# Reward-modulated STDP — implementation & validation plan

Supervised extension of the RF-article work. Unsupervised STDP erodes the oriented
prior (see interp harness / mechanism results); reward-modulated STDP gives the
plasticity a *class target*. Template: Goupy et al. 2024 "Neuronal Competition Groups
with Supervised STDP" (arXiv:2410.17066), rate-adapted to our existing framework.

## 0. Scope of the first step (smoke test)

Minimal change to prove the rule does *something*:
- Keep the current 3-layer layout (input / exc / inh) and existing RFs.
- **Fixed** class assignment of existing exc neurons: neuron `j` -> class `j mod 10`
  (~100 neurons/class with N_exc=1024). No new layer, no regrouping.
- Readout = pool exc firing rate by class label, argmax. (Diehl-Cook post-hoc labeling,
  but assigned a priori.)
- Run it as a **new cell in the interp harness** so we reuse `class_selectivity`,
  `orient_coh`, and readout-drift instrumentation and compare apples-to-apples against
  the unsupervised erosion cells (B1/B2).

**First run = V1 ONLY.** No variant sweep, no group structure yet. Just get the reward
rule working end-to-end and compare to V0.

**Everything except SE is STATIC in V1.** Only the feedforward input->exc (SE) weights
change, via reward. EE (recurrent), IE/EI (inhibition) are all frozen — the inhibition
is a fixed WTA scaffold. This isolates the reward rule as the single variable; letting
other populations learn unsupervised at the same time would reintroduce the confound.

**Recurrent (EE) weights are NOT reward-trained** — deliberately, not an oversight.
Recurrence is negligible here, and reward-strengthening recurrent excitation would
synchronize a group's neurons (redundancy + worse monopolization), the opposite of the
decorrelated coverage we want. An "assembly" variant with plastic EE could be tested
later, but it is advised against as a default.

Only if V1 works but plateaus do we add group structure (intra-group inhibition,
longer-tau homeostatic threshold).

## 1. The rule

Per plastic synapse (pre `j` -> post `i`), accumulate an eligibility over the sample and
apply once at the sample boundary:

```
e_ij   = n_pre_spikes[j] · n_post_spikes[i]   over the sample   (count-product, ≥ 0)
                                              (reset e_ij = 0 at each sample boundary)

R_i    = +1  if class(i) == y   (target neuron for this sample's label y)
         -1  otherwise
         # no need to check "did i fire": if i is silent, e_ij = 0, so Δw = 0 anyway.

Δw_ij  = η_r · (R_i − R̄) · e_ij     applied at sample boundary, then clip + Normalize
R̄     ← running mean of applied reward (global to start; per-class later if needed)
```

- **e_ij** = coincidence memory (unsupervised). We use the **count-product** form —
  `#pre spikes × #post spikes` over the trial: a symmetric rate/correlation eligibility
  (reward-modulated Hebbian; Loewenstein & Seung, Legenstein et al.). Chosen over the
  STDP `pre_trace`-at-post-spike form because our input is rate-coded (timing carries no
  class info) and it's a cheap separable outer product. NON-decaying, reset per sample
  (justified: fixed-length trial, reward at boundary). Eligibility traces are the
  standard mechanism of three-factor learning (Izhikevich 2007; Frémaux & Gerstner 2016;
  Gerstner et al. 2018) — this is not a novel construct.
- **R_i** = supervised direction (the third factor): potentiate correct-class
  coincidences, depress wrong-class ones. This is what makes it supervised.
- **R̄** = baseline: learn from deviation-from-usual, not raw reward. Kills DC drift
  (so Normalize isn't fighting a constant push), reduces variance, "learn from surprise."

### Where it slots into the code
- `trace_STDP` in `neurosnn/_core/synapses.py` ALREADY computes the coincidence term
  (`spike_trace[j]` sampled when post i spikes). The reward kernel reuses that: instead
  of applying `Δw` immediately, accumulate the (non-negative) coincidence into an
  eligibility matrix `E[i,j]`, then at the sample boundary apply `η_r·(R_i − R̄)·E`.
- New njit kernel `reward_STDP` + dataclass wrapper mirroring `TraceSTDPKernel`.
- The target label `y` is already available: `spike_labels` flows into `trainer.step`
  (runner.py ~line 345). Add a fixed `neuron_class[]` array alongside.
- Cadence change: current `update_weights_freq=100` applies mid-sample (num_steps=350).
  Reward update must fire at the **sample boundary** so eligibility spans one clean
  stimulus and reward multiplies the whole trial at once.
- Keep `Normalize` + homeostatic `spike_adaption` threshold ON.

### Config knobs
- `η_r` (reward learning rate), baseline mode (off / global / per-class).
- Per-block plasticity flags to run the variant sweep (SE static/reward, IE
  static/Vogels/reward).
- `neuron_class` assignment (fixed).

## 2. Variant sweep — where should reward-STDP live?

| Cell | SE (ff exc) | IE (inhibition) | purpose |
|------|-------------|-----------------|---------|
| V0 | static | static | frozen control (already have) |
| V1 | reward | static | PRIMARY — build first |
| V2 | reward | Vogels (unsup.) | stable "both adapt" |
| V3 | reward | reward | full, coupled, hardest — last |
| V4 | static | reward | optional ablation: inhibition alone |

Note: "dynamic inhibition" splits into (a) unsupervised Vogels homeostasis [V2] vs
(b) reward-modulated inhibition [V3/V4]. (b)'s credit assignment is a research question
(an inhibitory synapse "helped" if it suppressed a wrong-class competitor) — do it last.

## 3. Checks / diagnostics (what we measure to learn from results)

Primary:
- **Accuracy vs frozen baseline (V0)** and vs the Phase-0 frozen-Gaussian + logistic
  ceiling. Success = measurably beats V0.
- **class_selectivity trajectory** — does reward BUILD selectivity where unsupervised
  trace-STDP eroded it? (same panel as B1/B2).
- **readout drift (refit vs fixed@init)** — gap should now shrink/not grow.

Health / coverage:
- **Reward curve**: mean applied reward and running accuracy over training — trending up?
  Track R̄.
- **Coverage / dead-neuron diagnostics** (the monopolization check): fraction of neurons
  that ever win; per-class count of active neurons; entropy of winner identity within a
  class. Watch for neurons dying.
- **Weight health**: `w_floor_frac`, weight-norm distribution (explosion check — should
  be fine with Normalize; if not, that's a separate failure from monopolization).
- **Confusion matrix** — which classes confuse; informs whether inhibition-shaping (V3)
  could help.

## 4. Sanity controls (prove it's the supervised signal, not an artifact)

- **Label shuffle**: randomize `y` per sample -> learning signal should DISAPPEAR
  (accuracy stays at baseline). If it still "learns," the effect is an artifact.
- **Baseline on/off**: compare R̄ off vs on -> validates variance/drift role of the
  baseline; expect noisier / upward-drifting weights with it off.
- **Reward off** (`η_r=0`) == V0 frozen control (consistency check).

## 5. Success & kill criteria

- SUCCESS (V1): beats V0 on accuracy AND class_selectivity rises over training.
- INCONCLUSIVE (not failure): V1 flat. Could be missing intra-group competition — a null
  is NOT decisive (only a positive is). Next: add group structure before abandoning.
- KILL a direction: weights explode despite Normalize (bug, not concept), or label-shuffle
  control also "learns" (measuring an artifact) — stop and debug before more runs.

## 6. Build order (milestones)

1. `reward_STDP` njit kernel + wrapper (reuse trace coincidence; eligibility buffer +
   boundary application). Unit-test the eligibility/reset on a toy 2-neuron case.
2. Wire `neuron_class` + label plumbing + boundary cadence into trainer/runner.
3. Pool-by-label readout.
4. New interp-harness cell (V1); run vs B1/B2; add reward/coverage diagnostics.
5. Sanity controls (label shuffle, baseline on/off).
6. If V1 works: V2, then V3/V4. If V1 flat: add group structure and retry.

## 7. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| Winner monopolization (few neurons win everything, rest die) | homeostatic threshold with LONGER `tau_adaption`; watch coverage diagnostics; two-compartment threshold as fallback only |
| Reward positive-feedback (rich-get-richer) inflates weights | baseline R̄ (zero-mean updates) + Normalize |
| Weight explosion | Normalize + soft/hard bounds (already have) — distinct from monopolization |
| Mid-sample updates smear credit | apply at sample boundary only |
| Inhibitory reward credit assignment unclear | do exc-only (V1) first; V3/V4 last |
| Null result misread as "impossible" | treat V1 as smoke test; add group structure before concluding |
