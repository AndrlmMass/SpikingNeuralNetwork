"""Spiking readout layer trained by a local three-factor rule.

A biologically-motivated ALTERNATIVE to the softmax delta readout in
`RewardLearner` -- not a replacement. Both are meant to run in the same pass over
the same excitatory activity, so the comparison between them is paired
within-sample (same network, same spikes, same seed) rather than across runs.

One LIF neuron per class, each with its own membrane potential, threshold and
adaptation, driven by a plastic `W_ro` (N_exc, n_classes). The class is decoded
from the OUTPUT SPIKE COUNT over the trial -- no softmax anywhere.

Learning (error-modulated supervised STDP, after Goupy et al. 2024):

    per timestep    e_ij <- e_ij * decay + x_i(t) * y_j(t)   (pre trace x post spike)
    at boundary     err_j = A_j - a_j                        (desired minus actual count)
                    A_j   = mean(a) +/- margin               (+ for target, - otherwise)
                    dw_ij = lr * err_j * e_ij * soft_bound(w_ij)

`err_j` is defined relative to the layer mean, which (a) is the rate translation
of their "fire just before / just after the mean firing time" scheme, (b)
self-normalises against drift in overall output excitability, and (c) makes the
reward-baseline subtraction that three-factor rules require (Fremaux & Gerstner
2016) implicit -- err is already centred, so no separate EMA baseline is needed.

Weight signs are free: a negative w_ro stands for a disynaptic inhibitory path
(exc -> interneuron -> output), which is how a cortical circuit would implement
"this feature argues against class c". Without it the readout can only accumulate
evidence FOR a class, which is what limits the uniform pool to ~75%.
"""
from dataclasses import dataclass, field
from numba import njit
import numpy as np


@njit(cache=True)
def _readout_step(
    exc_spikes, W_ro, mp, I_syn, a, spike_threshold, pre_trace, elig, counts,
    mp_peak, out_spikes, prev_out_spikes,
    dt, tau_syn, tau_m, membrane_resistance, resting_potential, reset_potential,
    spike_threshold_default, min_mp, max_mp,
    decay_pre, decay_elig, spike_adaption, tau_adaption, delta_adaption,
    lateral_inh, noise_std, theta,
):
    """One timestep of the output layer. Mirrors the main LIF kernel's dynamics.

    Everything is in-place so the caller keeps ownership of the state arrays.
    Returns nothing; `out_spikes` holds this timestep's output spikes.
    """
    n_exc, n_out = W_ro.shape

    # pre-synaptic trace, updated BEFORE the post spike is resolved so that a
    # pre spike at t can take credit for a post spike at t (causal, inclusive).
    for i in range(n_exc):
        if exc_spikes[i] != 0:
            pre_trace[i] = pre_trace[i] * decay_pre + 1.0
        else:
            pre_trace[i] *= decay_pre

    # feedforward drive: only presynaptic neurons that actually spiked contribute
    nz = np.where(exc_spikes != 0)[0]
    prev_sum = 0.0
    for j in range(n_out):
        prev_sum += prev_out_spikes[j]

    for j in range(n_out):
        drive = 0.0
        for k in nz:
            drive += W_ro[k, j]
        # lateral inhibition from the OTHER outputs' previous-timestep spikes
        if lateral_inh != 0.0:
            drive -= lateral_inh * (prev_sum - prev_out_spikes[j])
        d_I = (-I_syn[j] + drive) * dt / tau_syn
        I_syn[j] += d_I
        d_mp = (
            (-(mp[j] - resting_potential) + membrane_resistance * I_syn[j])
            / tau_m * dt
        )
        mp[j] += d_mp
        if noise_std > 0.0:
            mp[j] += np.random.normal(0.0, noise_std)
        if mp[j] < min_mp:
            mp[j] = min_mp
        elif mp[j] > max_mp:
            mp[j] = max_mp
        # highest subthreshold potential reached this trial -- the tie-break used
        # when a trial produces no output spike at all
        if mp[j] > mp_peak[j]:
            mp_peak[j] = mp[j]

    for j in range(n_out):
        if mp[j] > spike_threshold[j]:
            out_spikes[j] = 1.0
            counts[j] += 1.0
            mp[j] = reset_potential
        else:
            out_spikes[j] = 0.0

    # eligibility: decay, then credit the active inputs of whichever output fired
    if decay_elig != 1.0:
        for i in range(n_exc):
            for j in range(n_out):
                elig[i, j] *= decay_elig
    for j in range(n_out):
        if out_spikes[j] != 0.0:
            for i in range(n_exc):
                if pre_trace[i] != 0.0:
                    elig[i, j] += pre_trace[i]

    # threshold = baseline + slow homeostatic offset + fast within-trial fatigue
    for j in range(n_out):
        if spike_adaption:
            a[j] += (-a[j] / tau_adaption) * dt
            if out_spikes[j] != 0.0:
                a[j] += delta_adaption
        spike_threshold[j] = spike_threshold_default + theta[j] + a[j]


@njit(cache=True)
def _apply_error(W_ro, elig, err, lr, w_min, w_max, mu_weight):
    """dw_ij = lr * err_j * elig_ij, with the soft bounds used by reward_STDP."""
    n_exc, n_out = W_ro.shape
    for j in range(n_out):
        e_j = err[j]
        if e_j == 0.0:
            continue
        for i in range(n_exc):
            g = elig[i, j]
            if g == 0.0:
                continue
            dw = lr * e_j * g
            w = W_ro[i, j]
            if dw >= 0.0:
                bound = (w_max - w) ** mu_weight if w_max > w else 0.0
            else:
                bound = (w - w_min) ** mu_weight if w > w_min else 0.0
            W_ro[i, j] = w + dw * bound
    return W_ro


def l1_renorm(W, target_l1):
    """Hold each output neuron's incoming |w| sum at its initial value.

    The heterosynaptic homeostasis Goupy et al. rely on, and the same idea as
    `regularization.post_norm` -- but on the L1 norm, NOT the signed sum.
    post_norm divides by the signed sum, which for a mixed-sign weight vector can
    pass through zero and blow the rescale up. Signed weights are the whole point
    of this readout, so it needs the L1 form.
    """
    cur = np.abs(W).sum(axis=0)
    scale = np.where(cur > 1e-12, target_l1 / np.maximum(cur, 1e-12), 1.0)
    return W * scale


@dataclass
class SpikingReadout:
    """LIF output layer, one neuron per class, trained by error-modulated STDP."""

    n_exc: int
    n_classes: int
    neuron_class: np.ndarray          # (n_exc,) exc -> class, for the warm start

    # --- learning ---
    lr: float = 0.005
    margin: float = 2.0               # desired count offset from the layer mean
    tau_elig: float = 0.0             # 0 = no decay within a trial (integrate all)
    tau_pre: float = 20.0
    # Non-negative by default. Measured 2026-07-27 by replaying the delta rule on
    # frozen run60k_5ep features, 3 seeds, identical but for the floor: signs free
    # 0.9397 +/- 0.0033 vs w>=0 0.9336 +/- 0.0043 -- negative weights are worth
    # 0.61 points, not the ~13 assumed. And w>=0 buys two things: the layer can no
    # longer be driven silent by its own weights (worst case is zero contribution,
    # never inhibitory drive), and L1 conservation then does what it should, since
    # weakening an adversarial synapse hands its budget to the useful ones -- which
    # it cannot do cleanly when a weight can consume budget by going negative.
    # Set w_min < 0 to recover the signed variant.
    w_min: float = 0.0
    w_max: float = 2.0
    mu_weight: float = 0.0            # 0 => hard bounds, as reward_STDP defaults
    renormalize: bool = True

    # --- dynamics (mirrors the main population's parameters) ---
    dt: float = 1.0
    tau_m: float = 20.0
    tau_syn: float = 5.0
    membrane_resistance: float = 15.0
    resting_potential: float = -70.0
    reset_potential: float = -80.0
    spike_threshold_default: float = -55.0
    min_mp: float = -100.0
    max_mp: float = 40.0
    spike_adaption: bool = True
    tau_adaption: float = 200.0
    delta_adaption: float = 0.5
    lateral_inh: float = 0.0          # off by default; competition is in the rule
    noise_std: float = 0.0            # exploration noise (Legenstein/Maass)
    # intrinsic homeostasis: keeps every output firing so eligibility keeps flowing
    homeo_lr: float = 0.02
    target_count: float = 6.0         # desired output spikes per trial
    theta_range: float = 15.0         # clamp on the per-output threshold offset

    seed: int = 0
    _rng: np.random.Generator = field(default=None, repr=False)

    def __post_init__(self):
        self.neuron_class = np.asarray(self.neuron_class)
        # Warm start at the block-diagonal one-hot, exactly as W_dense is, so the
        # readout BEGINS equivalent to uniform pooling and has to learn its way
        # up. Makes the learning curve interpretable and the comparison fair.
        self.W_ro = np.zeros((self.n_exc, self.n_classes), dtype=np.float64)
        self.W_ro[np.arange(self.n_exc), self.neuron_class] = 1.0
        self.target_l1 = np.abs(self.W_ro).sum(axis=0).copy()

        self.decay_pre = float(np.exp(-self.dt / self.tau_pre))
        self.decay_elig = (1.0 if self.tau_elig <= 0
                           else float(np.exp(-self.dt / self.tau_elig)))
        self._rng = np.random.default_rng(self.seed)

        self.mp = np.full(self.n_classes, self.resting_potential)
        self.I_syn = np.zeros(self.n_classes)
        self.a = np.zeros(self.n_classes)
        # per-output threshold offset, moved by the slow homeostat
        self.theta = np.zeros(self.n_classes)
        self.spike_threshold = np.full(self.n_classes, self.spike_threshold_default)
        self.pre_trace = np.zeros(self.n_exc)
        self.elig = np.zeros((self.n_exc, self.n_classes))
        self.counts = np.zeros(self.n_classes)
        self.mp_peak = np.full(self.n_classes, -np.inf)
        self.out_spikes = np.zeros(self.n_classes)
        self._prev_out = np.zeros(self.n_classes)

        # diagnostics accumulated across a checkpoint window
        self._n = 0
        self._correct = 0
        self._silent = 0
        self._tie = 0
        self.reset_trial()

    # ------------------------------------------------------------------ trial
    def reset_trial(self):
        """Clear per-sample state. Call at every sample boundary."""
        self.mp[:] = self.resting_potential
        self.I_syn[:] = 0.0
        self.pre_trace[:] = 0.0
        self.elig[:] = 0.0
        self.counts[:] = 0.0
        self.mp_peak[:] = -np.inf
        self.out_spikes[:] = 0.0
        self._prev_out[:] = 0.0

    def step_dynamics(self, exc_spikes):
        """Advance one timestep. `exc_spikes` is the (N_exc,) slice for this t."""
        self._prev_out[:] = self.out_spikes
        _readout_step(
            np.ascontiguousarray(exc_spikes, dtype=np.float64), self.W_ro,
            self.mp, self.I_syn, self.a, self.spike_threshold, self.pre_trace,
            self.elig, self.counts, self.mp_peak, self.out_spikes, self._prev_out,
            self.dt, self.tau_syn, self.tau_m, self.membrane_resistance,
            self.resting_potential, self.reset_potential,
            self.spike_threshold_default, self.min_mp, self.max_mp,
            self.decay_pre, self.decay_elig, self.spike_adaption,
            self.tau_adaption, self.delta_adaption, self.lateral_inh,
            self.noise_std, self.theta,
        )

    # ---------------------------------------------------------------- decoding
    def predict(self):
        """(class, was_silent, was_tie) from this trial's output spike counts.

        A rate-coded readout can simply fail to fire, which a softmax never does.
        Silent trials fall back to the highest subthreshold membrane potential
        reached -- the closest thing to "which output came nearest to firing".
        """
        c = self.counts
        top = c.max()
        if top <= 0.0:
            return int(np.argmax(self.mp_peak)), True, False
        tie = int((c == top).sum()) > 1
        if tie:
            # break by peak potential among the tied outputs, not by array order
            masked = np.where(c == top, self.mp_peak, -np.inf)
            return int(np.argmax(masked)), False, True
        return int(np.argmax(c)), False, False

    # ---------------------------------------------------------------- learning
    def apply_reward(self, target_label, train=True):
        """Error-modulated update at the sample boundary. Returns the prediction."""
        pred, silent, tie = self.predict()
        self._n += 1
        self._correct += int(pred == target_label)
        self._silent += int(silent)
        self._tie += int(tie)

        if train and self.lr > 0.0:
            # MARGIN error, not target-rate error.
            #
            # Goupy et al. rank firing TIMES -- "target fires earlier than the
            # others" has no ceiling. Translating that into a desired COUNT
            # (a_bar +/- margin) invents one: a target correctly firing 20 spikes
            # against a mean of 4 is then judged 14 spikes too high and gets
            # DEPRESSED. Measured, not theorised: with target-rate error the
            # frozen readout scored 1.00 on a separable task and learning drove it
            # to 0.40 with 87% of the layer silent.
            #
            # A classifier only needs an ORDERING: target above every competitor
            # by `margin`. Below, error is nonzero only while that is violated,
            # and is zero-sum by construction (+1 on the target, -1 shared among
            # the offenders), so no DC term can accumulate.
            tgt = int(target_label)
            err = np.zeros(self.n_classes)
            viol = np.flatnonzero(self.counts > self.counts[tgt] - self.margin)
            viol = viol[viol != tgt]
            if viol.size:
                err[tgt] = 1.0
                err[viol] = -1.0 / viol.size
            self.W_ro = _apply_error(self.W_ro, self.elig, err, self.lr,
                                     self.w_min, self.w_max, self.mu_weight)
            np.clip(self.W_ro, self.w_min, self.w_max, out=self.W_ro)
            if self.renormalize:
                self.W_ro = l1_renorm(self.W_ro, self.target_l1)

            # Intrinsic homeostasis. Eligibility is gated on POST spikes, so an
            # output that stops firing stops accumulating eligibility and can
            # never be potentiated back -- the layer is then permanently dead.
            # Nudging each output's own threshold toward a target firing count
            # keeps every output in play. Same idea as the theta adaptation on the
            # main population, but on the slow (across-trial) timescale.
            if self.homeo_lr > 0.0:
                self.theta += self.homeo_lr * (self.counts - self.target_count)
                np.clip(self.theta, -self.theta_range, self.theta_range,
                        out=self.theta)
        return pred

    def pop_stats(self):
        """Online accuracy / silent / tie rates since the last call; resets."""
        n = self._n
        out = dict(
            spiking_online_acc=(self._correct / n) if n else float("nan"),
            spiking_silent_frac=(self._silent / n) if n else float("nan"),
            spiking_tie_frac=(self._tie / n) if n else float("nan"),
            w_ro_mean=float(self.W_ro.mean()),
            w_ro_neg_frac=float((self.W_ro < 0).mean()),
        )
        self._n = self._correct = self._silent = self._tie = 0
        return out
