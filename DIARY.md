# Project Diary — sleep-paper revision (`plots-dec08`)

Running log of work sessions, **newest first**. This worktree is branched from
`b5d2cc1` ("we make plots bby", 8 Dec 2025) and carries the revision work for
Neurocomputing manuscript **NEUCOM-D-26-00713**, *Sleep-Based Homeostatic
Regularization for Stabilizing STDP in Recurrent SNNs*. Deadline **13 Oct 2026**.

**Convention:** add a `## YYYY-MM-DD — <short title>` section at the TOP for each
session. Capture focus, findings, decisions, and open items. Keep it scannable.

---

## 2026-09-23 — Conventional-stabilization baselines built; five bugs found, one of which invalidates the published sleep-ratio sweep

**Focus:** Implement the three conventional baselines Reviewer 3 asks for in
Major Point 1 (continuous weight decay, layer-wise weight normalization,
per-neuron synaptic scaling), calibrate them, and prepare an HPC grid. The
implementation went quickly; auditing the sleep path did not.

### BLOCKING — `sleep_max_iters` silently flattened the published sleep-ratio sweep

Sleep duration is capped **twice** and only the first cap is in the paper:

```python
sleep_window = round(check_sleep_interval * sleep_ratio)   # train.py:434
break when sleep_iter >= sleep_max_iters OR >= sleep_window # train.py:745
```

Realized duration is `min(window, sleep_max_iters)`. With
`check_sleep_interval = 35000` (hardcoded in `main.py`) and `--sleep-max-iters`
defaulting to **10000**, the cap binds above 28.6%:

| requested | window | realized |
|---|---|---|
| 10% | 3500 | 10.00% |
| 20% | 7000 | 20.00% |
| 30% | 10500 | **28.57%** |
| 50% | 17500 | **28.57%** |
| 100% | 35000 | **28.57%** |

**This affects the published MNIST-family sweep.** All 161 committed
`results/results_*.json` record `sleep_max_iters: 10000`, including 17 runs with
the full `(0.0, 0.1, ... 1.0)` sweep. `git log -L` confirms
`check_sleep_interval=35000` was stable across the whole commit range those
results came from.

So Table 1's β₃₀–β₁₀₀ are **eight estimates of the same condition** (28.57%
sleep). The coefficients look exactly like that: 0.875, 0.853, 0.827, 0.804,
0.787, 0.833, 0.806, 0.778 — a ±6% band — while β₁₀ = 1.563 and β₂₀ = 1.592, the
only two faithful conditions, sit far above.

This is why Reviewer 3 (R3.7) noticed *"the claim that performance decreases
monotonically beyond the optimum is not strictly supported by the reported
coefficients."* There is no dose-response beyond 28.6% because there is no
manipulation beyond 28.6%.

**Decision: the main results must be re-run** with the cap raised so every
requested ratio is faithful. Not yet verified by re-running the sweep; the
`Virtual sleep (epoch): N iters (~X%)` log line prints the realized percentage
directly if any run logs survive.

### `--on-timeout give_up` makes sleep *anti*-regularizing

The convergence criterion is never met inside a sleep window, so the timeout
branch is the normal operating regime, not an edge case. Under `give_up` the
window spends its full budget on noise-driven STDP with input suppressed — which
potentiates — and then abandons the downscaling. Measured on MNIST, T=70k, total
excitatory |w| relative to initialization:

| λ | `on_timeout` | final |
|---|---|---|
| — | *no sleep* | 4.13× |
| 0.99997 | `give_up` | **8.23×** |
| 0.99997 | `scale_to_target` | 2.19× |
| 0.9997 | `give_up` | 4.76× |
| 0.9997 | `scale_to_target` | 1.80× |

`big_comb` defaults to `scale_to_target`; **`main.py`'s `--on-timeout` defaults
to `give_up`**, so anything driven through `main.py` got the broken setting. The
main.py default is left unchanged pending a decision — the grid passes its choice
explicitly.

Analytic check on pull strength: the power law contracts each weight's
log-distance to target by λ per iteration, so over a 3500-step window
λ=0.99997 leaves 90% of the gap and λ=0.9997 leaves 35%. Neither converges.

### `scale_to_target` **is** layer normalization — a confound with our own baseline

```python
scale_exc = target_exc / current_sum_exc
weights[:ex, st:ih] *= scale_exc
```

One uniform factor across the block, i.e. exactly the `norm_layer` baseline. So
under `scale_to_target` the sleep arm is "3500 noisy STDP iterations, then a
layer-norm," and comparing it against `norm_layer` would partly compare a thing
to itself. **Grid uses `give_up` with λ=0.997 instead**, strong enough that the
window does the downscaling on its own (2.32× vs `scale_to_target`'s 1.76×).

### Weight clipping was gated on the `sleep` flag

```python
# Clip weights only when any form of sleep is active
if sleep:
    weights = clip_weights(...)
```

Sleep arms were hard-bounded to `[min_weight, max_weight]` every timestep; every
other condition got only sign clamping, with no upper bound. Effect on the
unregularized reference: **23.4× W₀ unclipped vs 8.5× clipped**. This would have
invalidated the whole R3.1 comparison. Added `clip_always` (default `False`,
preserving the old path); the grid passes it for every arm.

### Termination criterion did not match the published Eq. 6

Paper: $\mathcal{W}_x(t) \le \alpha_{\text{base}} \mathcal{W}_x(0)$ — one-sided.
Code: `abs(current_sum - target) <= sleep_tol_frac * target`, a **two-sided
±0.1% band**. Since the power law pulls each weight toward `w_target` rather than
steering the sum, the band is routinely overshot and the criterion never fires.

Added `--sleep-termination {band,below_target}` (default `band`). Switching to
`below_target` changed nothing measurable, because α_base = 1.0 demands sleep
drive total weight back to its *initialization* value and noise-driven STDP
inside the window prevents that. Raising α_base makes it reachable, but the
criterion then **skips whole episodes rather than shortening them** (the check
sits at the top of the loop):

| α_base (`beta`) | ratio | sleep iters |
|---|---|---|
| 1.0 | 5.018 | 7000 / 7000 |
| 1.5 | 5.021 | 7000 / 7000 |
| 2.0 | 5.000 | **3500** / 7000 |
| 3.0 | 5.000 | **3500** / 7000 |

**Decision: keep α_base = 1.0 for the grid.** Every episode then fires, matching
the normalization arms' cadence exactly. With α_base > 1.0 the *number* of
regularization events becomes data-dependent, reintroducing an event-count
confound. Weight bounding comes from λ, not from the threshold (5.000 vs 5.018).

### α_trig and α_base exist in code under different names

R3.6 asks for α_trig to be defined. It is present, as `alpha`:

```python
max_sum_exc      = sum_weights_exc * alpha   # alpha = 1.1 -> alpha_trig
baseline_sum_exc = sum_weights_exc * beta    # beta  = 1.0 -> alpha_base
```

Inside `sleep_func` both are functional — `if sum_weights_exc > max_sum_exc`
triggers, `if sum_weights_exc2 <= baseline_sum_exc` terminates. **But the trigger
block sits behind `if not sleep_now_inh or not sleep_now_exc:`**, and the
scheduled hard-pause path calls `sleep_func` with both forced `True`. So α_trig
is present but **bypassed** under scheduled triggering. The honest fix for R3.6
is to define `alpha` and state that it is inactive, not to delete it.

Naming collision to clean up: the paper uses β for *sleep percentage*, the code
uses `beta` for the α_base multiplier.

### The power law does not do what the mechanism description claims

$w \leftarrow w_{\text{tgt}}(w/w_{\text{tgt}})^{\lambda}$ is a **uniform
contraction in log space** — every weight closes the same *fraction* of its
log-distance. Consequences, verified numerically:

- Stronger weights are dragged **faster**, not slower. At λ=0.9997, a weight at
  25× target loses 0.0965%/iteration against 0.0067% at 1.2× target.
- Rank is preserved (monotonic map), but the *ratio* between strong and weak
  contracts as $(w_1/w_2)^{\lambda}$.

So the intended "strong synapses retain an advantage while dreaming" is not the
mechanism. **The outcome still holds, by a different route:** measured over 6
sleep episodes, sd(log w) rises 0.333 → 0.67–0.97 and p90/p10 rises 2.0 → 6–13,
because STDP re-expands the distribution faster than the power law contracts it.

Useful for R3.5: at *matched* total-weight control, sleep produces a markedly
tighter distribution than the conventional methods (sd 0.67–0.97, p90/p10 6–13
vs sd 1.70–2.04, p90/p10 46–119). Two-sided framing — sleep increases
heterogeneity relative to initialization but decreases it relative to
unregularized STDP and to the baselines.

### Baselines implemented and calibrated

`weight_funcs.py` gained three `@njit` kernels, all acting on the same nonzero
index sets `sleep_func` uses, so every arm regularizes identical synapses:

- `continuous_decay` — `w *= (1-rate)` every timestep, no target, sign-preserving
- `norm_layer` — instantaneous rescale of the block to its initial total |w|
- `norm_neuron` — synaptic scaling, per postsynaptic neuron

Unit-tested: decay preserves signs and ratios; `norm_layer` restores the block
total exactly; `norm_neuron` restores every post-neuron to 0.00% deviation where
`norm_layer` leaves 0.06%.

**Decay rate calibration.** The unregularized network grows *exponentially*, not
linearly — STDP potentiation scales with activity, which scales with weight — so
`dW/dt = (c - λ)W` with c ≈ 2.0e-5 per timestep (doubling every ~32k timesteps,
~320 images). Continuous decay therefore has **no attractor**: it is marginally
stable only at λ = c, and any mismatch compounds exponentially. That is a
structural difference from sleep and normalization, which measure the current sum
and correct toward a target (closed loop) where decay cannot (open loop).
Corroborated by [arXiv:1910.00122], which reports slow weight decay "renders
weight distribution unimodal but hardly affects global stability."

`DECAY_RATE = 1.6e-5` ≈ 1.07c, landing at 2.07× W₀ — well matched to sleep's
1.76–2.32× and to norm_layer/norm_neuron's 2.89–2.93×. All four regularized arms
sit in 1.76–2.93×, so the comparison tests mechanism rather than amount.

### Smaller issues found

- **`vectorized_trace_func` (`weight_funcs.py:140`) has no `@njit`**, unlike every
  kernel around it. Off the default path (`vectorized_trace=False`), so harmless
  now, but enabling that flag would be very slow. Looks unintentional.
- **`spike_timing` is O(all synapses) per timestep** regardless of activity: it
  `prange`s every post neuron and all ~38.5k incoming connections, then does
  `if spikes[j]==0 and spikes[i]==0: continue` *inside* the inner loop. Compare
  `trace_STDP` directly above it, which loops only over post neurons that fired.
- **`main.py --help` was broken** — unescaped `%` in the `--geom-jitter-amount`
  help string raised `ValueError: unsupported format character ')'`. Pre-existing;
  fixed.
- **Single-cell runs enabled plotting.** `disable_plotting` is
  `runs>1 or len(sleep_rates)>1 or len(datasets)>1`, and one grid cell is exactly
  one of each. Added `--no-plots`.
- **No matplotlib backend was pinned anywhere.** Added `matplotlib.use("Agg")`
  before any pyplot import, plus `MPLBACKEND=Agg` in the runner and SLURM script.
- **Stray `print("Norm!")`** inside the training loop (`train.py`).
- **Performance:** runs were being launched under the VS Code debugger
  (`debugpy`, 75 threads on 24 cores, no threading env vars). Pinning BLAS to one
  thread halves per-timestep cost — the matvec is 250×1034, far too small to
  thread, and unpinned OpenBLAS spent most of its time on synchronization.
  Measured 1.00 ms → 0.48 ms per timestep.

### Paper / code hyperparameter mismatches

Do not bias the comparison (all arms share them) but the revision's runs will not
reproduce the published configuration:

| Parameter | Paper table | `main.py` |
|---|---|---|
| A₋ | 0.3 | 0.5 |
| τ₋ (`tau_LTD`) | 7.5 ms | 10 ms |
| membrane noise var | 3 | 2 |
| sleep λ | 0.9997 | 0.99997 |

### Asymmetries to disclose rather than fix

- **Membrane noise is sleep-only** (`noisy_potential and (sleep_now_inh or
  sleep_now_exc)`), so the sleep arm has stochastic dynamics and the others are
  deterministic. Sleep bundles downscaling + noise + continued STDP against
  baselines that supply one mechanism — which is what R3.2's component ablation
  is for.
- **The sleep arm performs ~59,500 extra plasticity updates** (17 episodes ×
  3500). Arms are matched on data seen but not on weight updates or compute.
  Bears directly on R3.7's challenge to the "negligible overhead" claim.
- **`min_weight_exc = 0.01`** means no arm can prune to zero.

### Committed results confirm the cap, decisively

Per-dataset means from the committed `results/results_*.json` show a cliff at the
`sleep_max_iters` boundary and then a flat band:

| rate | notMNIST | fmnist | MNIST |
|---|---|---|---|
| 0.10 | .6463 | .5645 | .7151 |
| 0.20 | .6371 | .5525 | .7329 |
| **0.30** | **.2831** | **.3534** | .7106 |
| 0.40-1.00 | .2707-.2894 | .3058-.3412 | .6850-.7106 |

notMNIST's rates 0.30-1.00 span **1.8 percentage points across eight
conditions**. Eight genuinely different sleep durations from 30% to 100% would
not do that.

### Three study drivers, one shared configuration

| driver | grid | purpose |
|---|---|---|
| `src/sweep.py` | 11 ratios x 4 datasets x 5 seeds = 220 | re-run the main result; find the true optimum |
| `src/experiment.py` | 5 methods x 4 datasets x 5 seeds = 100 | R3.1 conventional-stabilization baselines |
| `src/ablation.py` | (2^4 + 1) x MNIST x 5 seeds = 85 | R3.2 sleep component factorial |

Run order matters: **sweep first**. `experiment.py` and `ablation.py` both read
the sleep ratio from `results/sweep/` via `experiment.resolve_sleep_ratio()`,
excluding ratio 0, and warn loudly if the sweep has not been run. So the optimum
propagates automatically instead of being copied by hand.

lambda is **0.997 everywhere** — `sweep.py`, `experiment.py`, `main.py`'s call
site and argparse default, and `big_comb.py`'s signature default. Before this
there were four different values live (0.9999 / 0.99997 / 0.9997 / 0.997), so any
of them could silently override another.

The four sleep components are now independently switchable via
`--sleep-components downscale,noise,stdp,suppress` (name a component to keep it
ON; `none` ablates all four). Verified that each gate changes behaviour rather
than only parsing, over 9 sleep episodes at T=35000:

| condition | exc \|w\| ratio | sd(log w) |
|---|---|---|
| 1111 full | 1.209 | 0.213 |
| 0111 no downscale | 6.063 | 0.983 |
| 1011 no noise | 1.206 | 0.212 |
| 1101 no stdp | 1.077 | 0.345 |
| 1110 no suppress | 1.209 | 0.213 |
| 0000 all off | 1.727 | 0.601 |

Downscaling dominates the weight metric; noise and input suppression barely move
it. Whether they matter for *accuracy* is a different question — weight-level and
accuracy-level effects have already diverged elsewhere in this codebase — and is
exactly what the factorial exists to answer.

One semantic caveat: with `suppress` off, real time is frozen so there is no
fresh input to stream in; the last presented spike vector is *held* for the whole
window. The contrast is "no sensory drive" vs "one frame repeated".

### Landed in this session

`src/weight_funcs.py` (+84), `src/train.py` (+204/-…), `src/big_comb.py` (+40),
`src/main.py` (+152), new `src/experiment.py` (100-cell grid driver),
new `slurm/run_baselines.sh`, `.gitignore` extended (results, logs, caches, sif).
All new flags default to the historical behaviour, so the submitted code path is
unchanged unless explicitly opted out of.

Grid configuration:

```
methods   none, sleep, decay, norm_layer, norm_neuron
datasets  mnist, fmnist, kmnist, notmnist
seeds     42-46                                     -> 100 cells
sleep     ratio 0.1, lambda 0.997, give_up, below_target,
          max_iters 10000, alpha_base 1.0
decay     1.6e-5 per timestep
shared    reg interval 35000, clip_always, no plots
```

### CORRECTION — num_steps: the paper is right, the worktree default is not

An earlier note in this entry claimed the code "ran 10x longer per stimulus than
the paper reports". **That was wrong.** `git log -L` on the `num_steps` default:

    b05e88b  2025-01-13  added as 1000
    1367c00  2025-12-05  1000 -> 100
    cd0315a  2025-12-08  100 -> 1000     <- this worktree is branched from here

The MNIST-family results are dated 1 Nov - 7 Dec 2025, i.e. almost entirely
before the switch back to 1000. So the published runs used 100 ms, the paper's
methods are accurate, and the 1000 default is a late change this worktree
happens to sit on. All three drivers now pass `--num-steps 100` explicitly.

Measured cost, same cell (mnist / none / seed 42):

| num_steps | wall-clock | peak RSS | test acc |
|---|---|---|---|
| 1000 | 73 min | — | 0.6271 |
| 100 | **2 min 18 s** | 1.5 GB | 0.3034 |

The accuracy difference is not a 100 ms penalty: 0% sleep is the *worst*
condition in the committed data (MNIST 0.4253 at rate 0, 0.7151 at rate 0.1), and
this cell is the unregularized arm. `--num-steps` and `--check-sleep-interval`
are now both exposed, because the latter is measured in timesteps and must
co-scale or the episode count per image changes 10-fold.

### Open items

1. **Re-run the main sleep-ratio sweep** with `sleep_max_iters` raised so ratios
   above 28.6% are faithful. Decided: yes.
2. ~~`deeplake` missing from `environment_linux.yml`~~ — **done**, pinned to
   `3.9.52`. Both this repo's loader and `precache_datasets.py` call
   `deeplake.load` / `deeplake.deepcopy`, which are v3 API and were removed in
   4.x; 3.9.52 matches the main repo's `env_linux.yml` and the working local
   `noise_env`. (The `4.4.3` pin in `requirements.txt` carries the note "might
   be wrong. Had issue when importing" — do not use it.)
3. ~~notMNIST hardcoded the remote hub path~~ — **done**. `get_data.py` now
   prefers a local deeplake copy at `NOTMNIST_LOCAL` (default
   `data/datasets/notmnist_dl`) and falls back to the hub only when no populated
   cache exists. A failed *local* load raises rather than silently falling back
   to the network, which would reintroduce the download race.
4. Datasets still must be pre-cached on a login node before submitting any
   array; only MNIST is cached in this worktree. Adapt `precache_datasets.py`
   from the main repo — its `TORCH_ROOT` (`data/torchvision`) already matches
   `get_data.py:355`, so it is mostly a copy with SVHN dropped.
5. Decide `main.py`'s `--on-timeout` default (currently `give_up`).
6. Duplicate 64 MB `src/data/` tree, created by running scripts from `src/`
   before the cwd fix. Safe to delete.
7. Reconcile the four paper/code hyperparameter mismatches above.
8. Define `alpha` (α_trig) for R3.6 and state that it is bypassed under
   scheduled triggering; remove the paper's β/`beta` symbol collision.

## 2026-09-24 — Sleep-phase plasticity is a dead end: neither the sign of the STDP window nor the noise dose matters

### Anti-STDP (Thiele et al. 2017) at the operating point: null

Ratio 0.1 (the sweep optimum), MNIST, λ = 0.9999576013637494, 5 seeds, paired.
`--sleep-anti-stdp` negates the sleep-phase learning rates only; wake plasticity
is untouched.

| seed | normal | anti | diff |
|------|--------|------|------|
| 42 | 0.7657 | 0.7829 | +0.0171 |
| 43 | 0.7337 | 0.7554 | +0.0217 |
| 44 | 0.7740 | 0.7797 | +0.0056 |
| 45 | 0.7663 | 0.7609 | −0.0054 |
| 46 | 0.7880 | 0.7609 | −0.0272 |
| mean | 0.7656 ± 0.0200 | 0.7679 ± 0.0124 | +0.0024 |

Paired t = 0.27 on 4 df (p ≈ 0.80); the sign flips across seeds. Inverting the
STDP window neither rescues the component nor worsens it.

### The noise dose does not matter either, over a 17-fold range

`var_noise` was hardcoded at `main.py:187`; it is now `--sleep-noise-var`
(default 2.0, unchanged) and is recorded in each result's `args`. Grid:
σ ∈ {2, 4, 8, 16} × {normal, anti} × 5 seeds = 40 runs, σ = 2 reusing the
comparison above.

| σ | normal | anti | anti − normal | paired t |
|---|--------|------|---------------|----------|
| 2 | 0.7656 ± 0.0200 | 0.7679 ± 0.0124 | +0.0024 | 0.27 |
| 4 | 0.7724 ± 0.0216 | 0.7679 ± 0.0219 | −0.0045 | −0.84 |
| 8 | 0.7746 ± 0.0175 | 0.7734 ± 0.0231 | −0.0012 | −0.16 |
| 16 | 0.7710 ± 0.0134 | 0.7778 ± 0.0268 | +0.0068 | 0.54 |

`acc ~ log2(σ) * arm + (1|seed)`, Beta: σ slope +0.011 per doubling (p = 0.39),
arm −0.018 (p = 0.71), interaction +0.009 (p = 0.60). Dropping both arm terms
costs nothing: LRT χ² = 0.33 on 2 df, p = 0.85. No run collapsed anywhere in the
grid (min accuracy 0.7337), so σ = 16 does not even destabilize.

### CORRECTION — σ = 2 is not subthreshold, as I first assumed

I initially reasoned that σ = 2 mV against the 15 mV rest-to-threshold gap
(−70 → −55) could never evoke a spike, and that this explained the ablation's
null noise coefficient (−0.011, p = 0.85). That is wrong. The noise is injected
every timestep while the leak removes only `dt/tau_m` = 1/30 of the membrane
deviation, so the process is AR(1) with stationary sd `σ/sqrt(1-(1-dt/tau_m)^2)`
≈ 3.9 σ. Simulating the sleep-phase dynamics with sensory drive suppressed:

| σ | stationary sd | gap in sd | sleep spikes/neuron | rate/step |
|---|---------------|-----------|---------------------|-----------|
| 2 | 7.8 | 1.92 | 26 | 0.0026 |
| 4 | 15.6 | 0.96 | 105 | 0.0107 |
| 8 | 31.2 | 0.48 | 233 | 0.0238 |
| 16 | 62.5 | 0.24 | 428 | 0.0437 |

So the default already evokes spontaneous spiking, and the sweep spans a 17-fold
range in sleep-phase firing rate. (This simulation sets `I_syn = 0`, ignoring
recurrent drive and the adaptive threshold, so it is a lower bound on the rate.)

### What this establishes

The manipulation was effective and the outcome was flat, which makes this a
strong negative result rather than an inconclusive one: sleep-phase spontaneous
activity does not affect the outcome at **any** dose from silent to 17× the
default, under plasticity of **either** sign. Only the downscaling component
carries the effect (ablation: downscale +1.291; stdp −0.134 with downscaling
present).

The parsimonious reading is that unstructured activity carries no information to
consolidate, so the sign of the rule is irrelevant — you cannot fix the learning
rule when the problem is its input. One plausible contributing mechanism, not
tested: the adaptive threshold homeostatically absorbs the extra drive, which
would explain why even σ = 16 neither helps nor destabilizes.

This is the empirical case for compressed replay (feeding time-compressed recent
input during sleep instead of noise) as the next step, and it is a much stronger
case than the σ = 2 test alone, because it rules out "we did not drive enough
activity."

Artifacts: `results/noise_sweep_summary.csv`, `results/results_noise_s*.json`,
`results/results_anti_*.json`.
