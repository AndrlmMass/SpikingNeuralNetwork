# Code Audit — why the network may be learning below its potential

Read-only audit of the core simulation + learning path (2026-07-17). Goal: find bugs,
neurodynamic weaknesses, and logic issues that could explain why accuracy caps ~0.80–0.83
(below raw-pixel LR ~0.87 at matched sample, and below Diehl & Cook's ~87% at comparable
neuron count). Scope read in full: `neurons.py`, `trainer.py` (timestep loop),
`synapses.py` (reward path), `regularization.py`, `init_weights.py` (WTA + RF),
`runner.py` (epoch/val flow), `get_data.py` (encoding).

**Bottom line:** no single smoking-gun bug. The code is largely *correct*. The ceiling is
an **accumulation of neurodynamic design gaps vs a working STDP-MNIST recipe (Diehl)**, plus
a few latent inconsistencies. Ranked by likely impact below. Severity = estimated effect on
the accuracy ceiling.

---

## HIGH severity — most likely to cap the representation

### H1. Soft / slow lateral inhibition → weak WTA → monopolization + dead neurons
- **Where:** competition path is E→I (1:1, `wta_ei_weights` = `eye*peak_ei`, init_weights.py:494)
  then I→E (block, `grouped_wta_ie_weights` peak_ie=−2, init_weights.py:584), integrated in
  `update_membrane_potential` (neurons.py:70–116) via `spikes_prev` (trainer.py:548).
- **Mechanism:** inhibition reaches a competitor **2–3 timesteps after** the winner spikes
  (exc→[+1]→inh fires→[+1]→I→E arrives; and inh needs ~2 exc spikes to cross threshold at
  peak_ei=50). Meanwhile I→E strength (−2) is small vs the winner's feedforward drive.
- **Why it hurts:** the WTA is *soft and laggy*, so a strong neuron fires several times before
  its group is suppressed, and weak neurons never get a turn → the monopolization we measure
  (dead_frac 0.56, half the population unused). Effectively ~440 useful neurons — *underperforming
  Diehl's 400*.
- **Contrast:** Diehl uses conductance inhibition with a −100 mV reversal (divisive shunting,
  instantaneous-feeling, self-limiting) — a much sharper WTA.
- **Verify:** log per-neuron firing-rate histogram over a batch; expect a few saturating winners
  + a long dead tail. Test: raise |peak_ie|, or shorten the E→I→E path, and watch dead_frac.

### H2. Current-based synapses (no conductance / no shunting)
- **Where:** neurons.py:77–86 — `I_syn += (−I_syn + Σw)·dt/τ`, `mp += (−(mp−rest) + R·I_syn)·dt/τ`.
  Input enters as a **voltage-independent current**.
- **Why it hurts:** no saturation. Diehl's `g·(V_rev − V)` weakens excitation as V rises and lets
  inhibition (−100 mV) *divisively* shunt. Our linear model lets a well-tuned neuron's drive grow
  without bound (clipped only at max_mp=40), amplifying the runaway-winner problem in H1 and making
  E/I balance fragile. This is the single largest neurodynamic divergence from a working recipe.
- **Verify:** compare winner membrane trajectories to the inhibition they receive; if inhibition
  barely bends V, shunting is missing.

### H3. Representation is lossy relative to pixels (capacity)
- **Evidence (measured this session):** refit-LR feature ceiling ~0.80–0.83, flat from batch 0;
  raw-pixel LR = 0.87 at the same sample size. The SNN encoding *discards* linearly-decodable
  class info. Driven by: only ~440 effective neurons (H1), near-global blob RFs (tiling cosmetic),
  and hard-WTA sparsification.
- **Why it hurts:** no readout (incl. the new dense one) can exceed a ceiling the representation
  doesn't contain. Raising it needs more *effective* neurons (fix H1) and/or richer RFs (oriented),
  not more training — the ceiling is flat over time.

---

## MEDIUM severity — plausibly limiting, worth fixing/testing

### M1. No refractory period, softened by a hyperpolarizing reset
- **Where:** `spike_and_reset` (neurons.py:171–173) has no refractory timer; reset = **−80 mV,
  below rest −70** (trainer.py defaults, harness LIF).
- **Two-edged:** the sub-rest reset acts as an *implicit* rate limiter (good), but it's
  drive-dependent — strongly-driven winners recover fast and re-fire, weakly-driven neurons take
  even longer or never (bad, *worsens* the dead-neuron tail). Diehl uses a fixed 5 ms refractory +
  reset-to-rest, which caps winners uniformly and is fairer to weak units.
- **Verify:** inter-spike-interval distribution of winners vs the theoretical cap.

### M2. No inter-image rest → state bleeds across images
- **Where:** images are concatenated back-to-back (get_data.py:577–585, num_steps each); nothing
  resets `mp`, `I_syn_exc/inh`, or `spike_trace` at the sample boundary (trainer.py loop). `a`
  (adaptation) also persists.
- **Why it hurts:** image N+1's first ~10–30 steps are contaminated by image N's decaying tail →
  blurs the per-image spike-count features used for both the reward eligibility and the readout.
  Diehl inserts a 150 ms blank (rates=0) between images to let everything settle.
- **Verify:** compare features when a blank/reset is inserted between images.

### M3. Adaptive threshold is transient, and train/eval mismatched
- **Where:** `a` update (neurons.py:179–184) with τ=200, δ=0.5 → decays ~83%/sample (within-trial
  fatigue, not lifetime homeostasis). At val/test `a` is **reset to zeros** (runner.py:569,657) and
  keeps adapting, while training carries a nonzero `a`.
- **Why it hurts:** provides none of the population rate-equalization that lets Diehl use all 6400
  neurons; and the train/eval mismatch means the competition state at scoring ≠ at training. (We
  confirmed this session that porting Diehl's persistent θ naively *breaks* selectivity in the
  supervised setup — so this is a "needs a different fix," not "just turn the knob.")

### M4. Weak, asymmetric, non-causal reward signal
- **Where:** `RewardLearner.step` / `reward_STDP` (synapses.py:463–517, 229–278).
- **Three issues:**
  1. **Label-driven, not performance-driven.** `reward_post` depends only on class assignment vs
     target (synapses.py:502), *not* on whether the neuron actually helped the readout. Every
     target-class neuron is potentiated regardless of quality — weak credit assignment.
  2. **Asymmetric magnitude.** baseline→−0.8 gives target neurons +1.8 but each non-target only
     −0.2 (synapses.py:503). Correct-class potentiation dominates; wrong-class evidence is barely
     suppressed.
  3. **Non-causal eligibility.** `e_ij = #pre·#post` (rate product), so a neuron drifts toward the
     class-average template rather than its causal drivers (relates to the whole-digit-RF finding).
- **Verify:** ablate each (performance-gated reward; symmetric baseline; causal trace) and compare.

### M5. Normalization runs every image in the reward path, ignoring `frequency`
- **Where:** reward branch (trainer.py:605–608) calls `norm_se.step` at **every** sample boundary
  with no `reg_frequency` gate (the non-reward branch *does* gate, trainer.py:383). `Normalizer.step`
  fully renormalizes on every call (regularization.py:85–110) — the `frequency=1050` set in the
  harness is silently ignored under reward.
- **Why it matters:** SE weights are renormalized ~1000×/batch. It preserves direction (so it
  doesn't erase selectivity), but it's far more aggressive than configured and couples oddly with
  the tiny reward_lr — worth confirming it isn't damping net learning per image. At minimum it's a
  **latent inconsistency** between intended and actual behavior.

---

## LOW severity / latent

- **L1. `mp` / `mp_new` alias after iteration 1** (MembranePotential holds one persistent `mp_new`;
  trainer sets `mp_prev = mp`, trainer.py:706). *Not a bug today* — `update_membrane_potential`
  reads `mp[i]` and writes `mp_new[i]` at the same index with drive from `spikes` (not `mp`), so
  there's no cross-neuron contamination. Fragile if anyone later reads neighbor `mp` in that loop.
- **L2. Reward path only normalizes SE, non-reward normalizes SE+EE** (trainer.py:606 vs 650–655).
  Irrelevant while W_ee=0, would matter if recurrence is enabled.
- **L3. `max_rate_hz=90`** (higher than Diehl's ~64). Fine, but interacts with H2 (no saturation) to
  push winners harder.

---

## Confirmed CORRECT (bounds the search — not where the bug is)

- Timestep flow: 1-step synaptic delay, `spikes_prev` updated correctly (trainer.py:707); membrane
  drive correctly includes inhibition via negative I→E weights.
- Sample boundary aligns with image (`time_per_item == num_steps`, runner.py:236); reward applied
  once/image; **spike counts reset each image** (synapses.py:516) — no cross-image eligibility leak.
- Baseline centering is correct (converges −0.8, zero-mean updates).
- WTA topology correct: winner suppresses group-mates but **not itself** (init_weights.py:583).
- Plastic set frozen at init structure; clipping/normalization touch only SE/EE, never the fixed
  WTA/inhibitory weights (trainer.py:222–226, 264–271); reward-STDP touches only structural pre
  (synapses.py:263), excluding inhibitory (init_weights.py:104).
- Dense readout learns the intended +own / −competitor structure (verified this session).

---

## Prioritized recommendations

1. **Attack the WTA (H1) first** — it gates H3 (effective neuron count). Options, cheapest first:
   stronger |peak_ie|; shorten/skip the E→I→E delay (direct lateral E→E inhibition or apply
   inhibition same-step); then the real fix, **conductance-based shunting inhibition (H2)**.
2. **Add a fixed refractory + reset-to-rest (M1)** and **an inter-image blank (M2)** — small, cheap,
   both align us with the working recipe and clean up the features.
3. **More effective neurons** — once H1 is fixed, scale N_exc (Diehl scaling) to lift the ceiling.
4. **Oriented RFs** — richer features (Andreas' 89% precedent) to raise H3.
5. **Reward-signal upgrades (M4)** — performance-gated and/or causal eligibility, as ablations.
6. Fix the normalization-frequency inconsistency (M5) or make it intentional.

**Honest framing:** don't expect one fix to jump us to 95%. H1+H2 (sharp, shunting WTA) is the
highest-leverage lever because it simultaneously revives dead neurons, enables neuron-count scaling,
and is the biggest divergence from a recipe that *does* reach 95%.
