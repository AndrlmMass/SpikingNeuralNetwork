from numba.typed import List
from numba import njit
from tqdm import tqdm
import numpy as np
from .plasticity import sleep_func, spike_timing


def report_numba_status():
    """Print a minimal Numba status summary (no large IR dumps)."""
    try:
        funcs = [
            ("clip_weights", clip_weights),
            ("sleep_func", sleep_func),
            ("spike_timing", spike_timing),
        ]
        msgs = []
        for name, func in funcs:
            sigs = getattr(func, "signatures", []) or []
            status = "compiled" if len(sigs) > 0 else "pending"
            msgs.append(f"{name}:{status}")
        print("Numba status — " + ", ".join(msgs))
    except Exception:
        pass


@njit(cache=True)
def clip_weights(
    weights,
    nz_cols_exc,
    nz_cols_inh,
    nz_rows_exc,
    nz_rows_inh,
):
    # Only enforce sign boundaries: exc >= 0, inh <= 0
    # This prevents weights from crossing over (exc->inh or inh->exc)
    for i_ in range(nz_rows_exc.shape[0]):
        i, j = nz_rows_exc[i_], nz_cols_exc[i_]
        if weights[i, j] < 0.0:
            weights[i, j] = 0.0
    for i_ in range(nz_rows_inh.shape[0]):
        i, j = nz_rows_inh[i_], nz_cols_inh[i_]
        if weights[i, j] > 0.0:
            weights[i, j] = 0.0
    return weights


def update_weights(
    weights,
    spike_times,
    nonzero_pre_idx,
    noisy_weights,
    weight_mean_noise,
    weight_var_noise,
    spikes,
    N_inh,
    sleep,
    sleep_now_inh,
    sleep_now_exc,
    N_x,
    nz_cols_exc,
    nz_cols_inh,
    nz_rows_exc,
    nz_rows_inh,
    A_plus,
    A_minus,
    learning_rate_exc,
    learning_rate_inh,
    tau_LTP,
    tau_LTD,
):
    """
    Apply the STDP rule to update synaptic weights using a fully vectorized approach.

    Parameters:
    - weights: Matrix of weights (2D or 1D, depending on connections).
    - spike_times: Array of time since the last spike for each neuron (0 indicates a spike at the current timestep).
    - min_weight_exc, max_weight_exc: Min and max weights for excitatory synapses.
    - min_weight_inh, max_weight_inh: Min and max weights for inhibitory synapses.
    - N_inh: Number of inhibitory neurons.
    - learning_rate_exc, learning_rate_inh: Learning rates for excitatory and inhibitory weights.
    - tau_pre, tau_post: Time constants for pre- and post-synaptic STDP components.

    Returns:
    - Updated weights.
    """
    # Clip weights only when any form of sleep is active
    if sleep:
        weights = clip_weights(
            weights=weights,
            nz_cols_exc=nz_cols_exc,
            nz_cols_inh=nz_cols_inh,
            nz_rows_exc=nz_rows_exc,
            nz_rows_inh=nz_rows_inh,
        )

    # Update weights using spike timing
    weights = spike_timing(
        spike_times=spike_times,
        A_plus=A_plus,
        A_minus=A_minus,
        tau_LTP=tau_LTP,
        tau_LTD=tau_LTD,
        learning_rate_exc=learning_rate_exc,
        learning_rate_inh=learning_rate_inh,
        N_inh=N_inh,
        weights=weights,
        N_x=N_x,
        spikes=spikes,
        nonzero_pre_idx=nonzero_pre_idx,
    )

    # add noise to weights if desired
    if noisy_weights:
        delta_weight_noise = np.random.normal(
            loc=weight_mean_noise, scale=weight_var_noise, size=weights.shape
        )
        weights += delta_weight_noise

    return weights, sleep_now_inh, sleep_now_exc


def update_membrane_potential(
    mp,
    weights,
    spikes,
    noisy_potential,
    resting_potential,
    membrane_resistance,
    tau_m,
    dt,
    I_syn,
    tau_syn,
    mean_noise,
    var_noise,
    sleep_now_inh,
    sleep_now_exc,
    weights_T=None,  # Pre-transposed weights for speed
):
    # Use pre-transposed weights if available
    if weights_T is not None:
        I_syn_new = I_syn + (-I_syn + np.dot(weights_T, spikes)) * dt / tau_syn
    else:
        I_syn_new = I_syn + (-I_syn + np.dot(weights.T, spikes)) * dt / tau_syn

    # Compute membrane potential update
    mp_new = mp + (
        (-(mp - resting_potential) + membrane_resistance * I_syn_new) / tau_m * dt
    )

    # Add noise only during sleep
    if noisy_potential and (sleep_now_inh or sleep_now_exc):
        mp_new += np.random.normal(loc=mean_noise, scale=var_noise, size=mp.shape)

    return mp_new, I_syn_new


def update_spikes(
    st,
    ih,
    mp,
    dt,
    a,
    spikes,
    spike_times,
    max_mp,
    min_mp,
    spike_adaption,
    tau_adaption,
    delta_adaption,
    noisy_threshold,
    spike_slope,
    spike_intercept,
    spike_threshold,
    spike_threshold_default,
    reset_potential,
):
    # Clip membrane potential
    np.clip(mp, min_mp, max_mp, out=mp)

    # Threshold crossing - vectorized
    spiked = mp > spike_threshold
    spikes[st:ih][spiked] = 1

    if noisy_threshold:
        delta_potential = spike_threshold - mp
        p_fire = np.exp(spike_slope * delta_potential + spike_intercept) / (
            1 + np.exp(spike_slope * delta_potential + spike_intercept)
        )
        additional_spikes = np.random.binomial(n=1, p=p_fire)
        spikes[st:ih] = spikes[st:ih] | additional_spikes

    if spike_adaption:
        a += (-a / tau_adaption) * dt
        a[spikes[st:ih] == 1] += delta_adaption
        spike_threshold = spike_threshold_default + a
        np.clip(spike_threshold, -90, 0, out=spike_threshold)

    # Reset spiked neurons
    mp[spikes[st:ih] == 1] = reset_potential # where does ih go? max or to ex?

    # Update spike times - use in-place where possible
    spike_times[spikes == 1] = 0
    spike_times[spikes == 0] += 1

    return mp, spikes, spike_times, spike_threshold, a


def train_network(
    weights,
    mp,
    spikes,
    noisy_potential,
    resting_potential,
    membrane_resistance,
    spike_times,
    train_weights,
    weight_decay_rate_exc,
    weight_decay_rate_inh,
    sleep_synchronized,
    baseline_sum,
    max_sum,
    N_inh,
    N_exc,
    learning_rate_exc,
    learning_rate_inh,
    tau_LTP,
    tau_LTD,
    max_mp,
    min_mp,
    w_target_exc,
    w_target_inh,
    baseline_sum_exc,
    baseline_sum_inh,
    max_sum_exc,
    max_sum_inh,
    dt,
    N,
    A_plus,
    A_minus,
    tau_m,
    sleep,
    spike_adaption,
    tau_adaption,
    delta_adaption,
    spike_threshold_default,
    reset_potential,
    spike_slope,
    spike_intercept,
    noisy_threshold,
    noisy_weights,
    spike_labels,
    weight_mean_noise,
    weight_var_noise,
    N_x,
    T,
    tau_syn,
    beta,
    mean_noise,
    var_noise,
    I_syn,
    a,
    spike_threshold,
    sleep_ratio=0.0,
    normalize_weights=False,
    initial_sum_exc=None,
    initial_sum_inh=None,
    # Hard-pause sleep settings
    sleep_hard_pause: bool = True,
    sleep_epsilon: float = 1e-8,
    sleep_tol_frac: float = 1e-3,
    sleep_max_iters: int = 5000,
    on_timeout: str = "scale_to_target",  # one of {"scale_to_target","extend","give_up"}
    sleep_mode: str = "static",  # one of {"static","group","post"}
    track_weights: bool = False,  # Enable weight tracking (adds overhead)
    weight_track_samples_exc=8,
    weight_track_samples_inh=8,
    train_snapshot_interval=None,
    sleep_snapshot_interval=None,
):

    st = N_x  # stimulation
    ex = st + N_exc  # excitatory
    ih = ex + N_inh  # inhibitory

    # Disable training-time plotting snapshots for performance
    weights_4_plotting_exc = np.empty((0, 0, 0))
    weights_4_plotting_inh = np.empty((0, 0, 0))

    delta_w = np.zeros(shape=weights.shape)

    nz_rows_inh, nz_cols_inh = np.nonzero(weights[ex:ih, st:ex])
    nz_rows_exc, nz_cols_exc = np.nonzero(weights[:ex, :ih])
    nz_rows_inh += ex
    nz_cols_inh += st
    nz_rows, nz_cols = np.nonzero(weights)
    sleep_now_inh = False
    sleep_now_exc = False

    # Pre-transpose weights for faster matrix multiply in membrane potential update
    weights_T_cache = weights[:, st:ih].T.copy()

    # Precompute scheduled sleep window per batch
    # e.g., sleep_ratio=0.1 means 10% of each batch is sleep
    if sleep and sleep_ratio is not None and sleep_ratio > 0.0:
        sleep_window = max(1, int(round(T * sleep_ratio)))
    else:
        sleep_window = 0

    # Print sleep configuration
    if sleep and sleep_window > 0:
        expected_sleep_pct = (sleep_window / T) * 100
        print(
            f"Sleep scheduled: {sleep_window}/{T} timesteps per batch ({expected_sleep_pct:.1f}%)"
        )

    # Suppose weights is your initial 2D numpy array of weights.
    # Here, we assume that the columns correspond to post-neurons.
    if train_weights:
        desc = "Training network:"
    else:
        desc = "Testing network:"

    # Compute for neurons N_x to N_post-1
    nonzero_pre_idx = List()
    for i in range(st, ih):
        pre_idx = np.nonzero(weights[:ih, i])[0]
        nonzero_pre_idx.append(pre_idx.astype(np.int64))

    sleep_amount = 0
    virtual_sleep_iters_epoch = 0

    # Maintain previous-step state explicitly so we can evolve during hard-pause sleep
    mp_prev = mp[0].copy()
    spikes_prev = spikes[0].copy()
    t_virtual = 0  # virtual time used only inside sleep loop

    # Lightweight weight tracking focused around sleep periods (plus sparse training samples)
    # Only initialize if track_weights is enabled to avoid overhead
    weight_tracking_sleep = None
    exc_pairs = []
    inh_pairs = []
    plot_time = 0
    train_record_every = 1
    sleep_record_every = 1

    if track_weights:
        weight_tracking_sleep = {
            "times": [],  # monotonically increasing plot-time (snapshot index)
            "exc_mean": [],
            "exc_std": [],
            "exc_min": [],
            "exc_max": [],
            "exc_samples": [],  # list of [K_exc] values (tracked weights)
            "inh_mean": [],
            "inh_std": [],
            "inh_min": [],
            "inh_max": [],
            "inh_samples": [],  # list of [K_inh] values (tracked weights)
            "sleep_segments": [],  # list of (t_start, t_end) in the same plot-time reference
        }
        # Select a small number of active synapses to track (prefer non-zero)
        rng = np.random.default_rng(42)
        try:
            K_exc = max(1, int(weight_track_samples_exc))
        except Exception:
            K_exc = 8
        try:
            K_inh = max(1, int(weight_track_samples_inh))
        except Exception:
            K_inh = 8
        # Candidates within the decayed submatrices
        exc_cand = np.argwhere(weights[:ex, st:ih] != 0)
        inh_cand = np.argwhere(weights[ex:ih, st:ex] != 0)
        # Fall back to random positions if there are too few non-zero entries
        if exc_cand.shape[0] >= K_exc:
            choose_exc = rng.choice(exc_cand.shape[0], size=K_exc, replace=False)
            exc_pairs = [
                (int(exc_cand[i, 0]), int(st + exc_cand[i, 1])) for i in choose_exc
            ]
        else:
            # Sample uniformly within the slice
            if (ex > 0) and (ih - st > 0):
                exc_pairs = [
                    (int(rng.integers(0, ex)), int(st + rng.integers(0, ih - st)))
                    for _ in range(K_exc)
                ]
            else:
                exc_pairs = []
        if inh_cand.shape[0] >= K_inh:
            choose_inh = rng.choice(inh_cand.shape[0], size=K_inh, replace=False)
            inh_pairs = [
                (int(ex + inh_cand[i, 0]), int(st + inh_cand[i, 1])) for i in choose_inh
            ]
        else:
            if (ih - ex > 0) and (ex - st > 0):
                inh_pairs = [
                    (
                        int(ex + rng.integers(0, ih - ex)),
                        int(st + rng.integers(0, ex - st)),
                    )
                    for _ in range(K_inh)
                ]
            else:
                inh_pairs = []

        # Snapshot cadence
        custom_train_interval = None
        try:
            if (
                train_snapshot_interval is not None
                and float(train_snapshot_interval) > 0
            ):
                custom_train_interval = max(1, int(train_snapshot_interval))
        except Exception:
            custom_train_interval = None
        custom_sleep_interval = None
        try:
            if (
                sleep_snapshot_interval is not None
                and float(sleep_snapshot_interval) > 0
            ):
                custom_sleep_interval = max(1, int(sleep_snapshot_interval))
        except Exception:
            custom_sleep_interval = None
        if sleep_window > 0:
            default_sleep_interval = max(1, sleep_window // 100)
        else:
            default_sleep_interval = 5
        sleep_record_every = custom_sleep_interval or default_sleep_interval

    def _record_snapshot():
        nonlocal plot_time
        if not track_weights or weight_tracking_sleep is None:
            return False
        # Compute group stats over signed weights (preserve inhibitory sign)
        try:
            W_exc = weights[:ex, st:ih][weights[:ex, st:ih] != 0]
            W_inh = weights[ex:ih, st:ex][weights[ex:ih, st:ex] != 0]

            exc_vals = (
                [float(weights[i, j]) for (i, j) in exc_pairs] if exc_pairs else []
            )
            inh_vals = (
                [float(weights[i, j]) for (i, j) in inh_pairs] if inh_pairs else []
            )
            weight_tracking_sleep["times"].append(plot_time)
            # Exc stats
            if W_exc.size > 0:
                weight_tracking_sleep["exc_mean"].append(float(np.mean(W_exc)))
                weight_tracking_sleep["exc_std"].append(float(np.std(W_exc)))
                weight_tracking_sleep["exc_min"].append(float(np.min(W_exc)))
                weight_tracking_sleep["exc_max"].append(float(np.max(W_exc)))
            else:
                weight_tracking_sleep["exc_mean"].append(0.0)
                weight_tracking_sleep["exc_std"].append(0.0)
                weight_tracking_sleep["exc_min"].append(0.0)
                weight_tracking_sleep["exc_max"].append(0.0)
            # Inh stats
            if W_inh.size > 0:
                weight_tracking_sleep["inh_mean"].append(float(np.mean(W_inh)))
                weight_tracking_sleep["inh_std"].append(float(np.std(W_inh)))
                weight_tracking_sleep["inh_min"].append(float(np.min(W_inh)))
                weight_tracking_sleep["inh_max"].append(float(np.max(W_inh)))
            else:
                weight_tracking_sleep["inh_mean"].append(0.0)
                weight_tracking_sleep["inh_std"].append(0.0)
                weight_tracking_sleep["inh_min"].append(0.0)
                weight_tracking_sleep["inh_max"].append(0.0)
            # Samples
            weight_tracking_sleep["exc_samples"].append(exc_vals)
            weight_tracking_sleep["inh_samples"].append(inh_vals)
            return True
        except Exception:
            return False

    # Validate T parameter
    if T is None or T <= 0:
        raise ValueError(f"T (total timesteps) must be a positive integer, got: {T}")

    # Create progress bar wrapping the timestep range
    # Note: range(1, T) gives timesteps 1 to T-1 (T-1 total iterations)
    pbar = tqdm(
        range(1, T),
        desc=desc,
        leave=False,
        unit="step",
    )
    last_sample = 0
    last_sleep_flag = -1  # unknown
    last_stats_update_t = -1000
    # Initial snapshot (only if tracking enabled)
    if track_weights:
        _record_snapshot()
        plot_time += 1
    for t in range(1, T):
        # Update progress bar only when sample count changes (every 1000 timesteps)
        current_sample = t // 1000
        if current_sample > last_sample:
            pbar.update(current_sample - last_sample)
            last_sample = current_sample

        # Reset sleep flags for this timestep
        sleep_now_inh = False
        sleep_now_exc = False

        # Trigger hard-pause only at the start of a batch (sleep window at batch start)
        is_window_start = (
            sleep and sleep_window > 0 and ((t % T) == 0)
        )

        # Hard-pause sleep: run inner loop without advancing real time t
        slept_this_step = False
        if train_weights and sleep and sleep_hard_pause and is_window_start:
            # Mark current real timestep as sleep once
            if spike_labels is not None:
                spike_labels[t] = -2

            # Determine current sleep targets based on sleep_mode
            # Defaults: use scalars passed in
            use_post_targets = False
            w_target_exc_cur = w_target_exc
            w_target_inh_cur = w_target_inh
            # Always allocate vectors (Numba requires consistent types), will be unused unless post-mode
            w_target_exc_vec = np.zeros(N)
            w_target_inh_vec = np.zeros(N)
            eps = 1e-12

            if sleep_mode == "group":
                try:
                    # Mean of absolute weights for each group
                    if (ex > 0) and (ih - st > 0):
                        w_target_exc_cur = float(np.mean(np.abs(weights[:ex, st:ih])))
                        # ensure strictly positive
                        if w_target_exc_cur <= 0:
                            w_target_exc_cur = max(np.abs(w_target_exc), eps)
                    if (ih - ex > 0) and (ex - st > 0):
                        inh_mag = float(np.mean(np.abs(weights[ex:ih, st:ex])))
                        if inh_mag <= 0:
                            inh_mag = max(np.abs(w_target_inh), eps)
                        # inhibitory sign handled inside sleep_func; keep scalar for API
                        w_target_inh_cur = -inh_mag
                except Exception:
                    # Fallback to static if anything goes wrong
                    w_target_exc_cur = max(np.abs(w_target_exc), eps)
                    w_target_inh_cur = -max(np.abs(w_target_inh), eps)

            elif sleep_mode == "post":
                use_post_targets = True
                try:
                    # Per-post targets: mean(|w|) computed over NON-ZERO incoming weights only
                    if (ex > 0) and (ih - st > 0):
                        abs_exc = np.abs(weights[:ex, st:ih])
                        cnt_exc = np.count_nonzero(abs_exc, axis=0)
                        sum_exc = np.sum(abs_exc, axis=0)
                        means_exc = np.divide(
                            sum_exc,
                            cnt_exc,
                            out=np.zeros_like(sum_exc),
                            where=cnt_exc > 0,
                        )
                        w_target_exc_vec[st:ih] = means_exc
                        # Avoid zeros (division by zero risk) via scalar fallback
                        for j in range(st, ih):
                            if cnt_exc[j - st] <= 0 or w_target_exc_vec[j] <= 0:
                                w_target_exc_vec[j] = max(np.abs(w_target_exc), eps)
                    if (ih - ex > 0) and (ex - st > 0):
                        abs_inh = np.abs(weights[ex:ih, st:ex])
                        cnt_inh = np.count_nonzero(abs_inh, axis=0)
                        sum_inh = np.sum(abs_inh, axis=0)
                        means_inh = np.divide(
                            sum_inh,
                            cnt_inh,
                            out=np.zeros_like(sum_inh),
                            where=cnt_inh > 0,
                        )
                        w_target_inh_vec[st:ex] = means_inh
                        for j in range(st, ex):
                            if cnt_inh[j - st] <= 0 or w_target_inh_vec[j] <= 0:
                                w_target_inh_vec[j] = max(np.abs(w_target_inh), eps)
                    # Scalars remain as safe magnitudes (sleep_func uses vectors when use_post_targets=True)
                    w_target_exc_cur = max(np.abs(w_target_exc), eps)
                    w_target_inh_cur = -max(np.abs(w_target_inh), eps)
                except Exception:
                    # Fallback to static mode
                    use_post_targets = False
                    w_target_exc_cur = max(np.abs(w_target_exc), eps)
                    w_target_inh_cur = -max(np.abs(w_target_inh), eps)

            # Targets derived from baseline and beta
            target_exc = (
                baseline_sum_exc * beta if baseline_sum_exc is not None else None
            )
            target_inh = (
                baseline_sum_inh * beta if baseline_sum_inh is not None else None
            )

            sleep_iter = 0
            sleep_time_counter = 0

            while True:
                # Compute current sums
                current_sum_exc = np.sum(np.abs(weights[:ex, st:ih]))
                current_sum_inh = np.sum(np.abs(weights[ex:ih, st:ex]))

                # Check convergence to targets using fractional tolerance if targets exist
                eps_exc = (
                    sleep_tol_frac * target_exc
                    if (target_exc is not None)
                    else sleep_epsilon
                )
                eps_inh = (
                    sleep_tol_frac * target_inh
                    if (target_inh is not None)
                    else sleep_epsilon
                )
                reached_exc = (
                    target_exc is None
                    or np.abs(current_sum_exc - target_exc) <= eps_exc
                )
                reached_inh = (
                    target_inh is None
                    or np.abs(current_sum_inh - target_inh) <= eps_inh
                )
                if reached_exc and reached_inh:
                    break

                # Safety cap: handle non-convergence
                if sleep_iter >= sleep_max_iters or sleep_iter >= sleep_window:
                    if on_timeout == "scale_to_target":
                        # Scale weights to hit targets exactly (if defined)
                        if target_exc is not None and current_sum_exc > 1e-12:
                            scale_exc = target_exc / current_sum_exc
                            weights[:ex, st:ih] *= scale_exc
                        if target_inh is not None and current_sum_inh > 1e-12:
                            scale_inh = target_inh / current_sum_inh
                            weights[ex:ih, st:ex] *= scale_inh
                        # Clip to bounds after scaling
                        weights = clip_weights(
                            weights=weights,
                            nz_cols_exc=nz_cols_exc,
                            nz_cols_inh=nz_cols_inh,
                            nz_rows_exc=nz_rows_exc,
                            nz_rows_inh=nz_rows_inh,
                        )
                    elif on_timeout == "extend":
                        # Allow additional internal iterations up to sleep_max_iters
                        if sleep_iter < sleep_max_iters:
                            sleep_window = sleep_max_iters
                            # continue sleeping
                        else:
                            pass  # fall through to break
                    # Exit hard-pause sleep
                    break

                # Run one internal sleep iteration: evolve dynamics with no sensory input
                sleep_now_exc = True
                sleep_now_inh = True
                slept_this_step = True

                # Zero sensory input (do not consume data)
                spikes_prev[:st] = 0

                # Update membrane potentials using previous step spikes
                mp_prev, I_syn = update_membrane_potential(
                    mp=mp_prev,
                    weights=weights[:, st:ih],
                    spikes=spikes_prev,
                    resting_potential=resting_potential,
                    membrane_resistance=membrane_resistance,
                    tau_m=tau_m,
                    dt=dt,
                    noisy_potential=noisy_potential,
                    mean_noise=mean_noise,
                    var_noise=var_noise,
                    I_syn=I_syn,
                    tau_syn=tau_syn,
                    sleep_now_inh=sleep_now_inh,
                    sleep_now_exc=sleep_now_exc,
                    weights_T=weights_T_cache,
                )

                # Prepare current spikes vector: keep prior network activity but zero sensory inputs
                sleep_spikes_cur = spikes_prev.copy()
                sleep_spikes_cur[:st] = 0

                # Update spikes and thresholds
                (
                    mp_prev,
                    sleep_spikes_cur,
                    spike_times,
                    spike_threshold,
                    a,
                ) = update_spikes(
                    st=st,
                    ih=ih,
                    mp=mp_prev,
                    dt=dt,
                    a=a,
                    spikes=sleep_spikes_cur,
                    spike_times=spike_times,
                    spike_intercept=spike_intercept,
                    spike_slope=spike_slope,
                    noisy_threshold=noisy_threshold,
                    spike_adaption=spike_adaption,
                    tau_adaption=tau_adaption,
                    delta_adaption=delta_adaption,
                    max_mp=max_mp,
                    min_mp=min_mp,
                    spike_threshold=spike_threshold,
                    spike_threshold_default=spike_threshold_default,
                    reset_potential=reset_potential,
                )

                # Update weights once per internal sleep iteration (use previous spikes_prev)
                if train_weights:
                    # First apply the slow exponential decay toward targets
                    weights, sleep_now_inh, sleep_now_exc = sleep_func(
                        weights=weights,
                        max_sum=max_sum,
                        max_sum_exc=max_sum_exc,
                        max_sum_inh=max_sum_inh,
                        sleep_now_inh=True,
                        sleep_now_exc=True,
                        w_target_exc=w_target_exc_cur,
                        w_target_inh=w_target_inh_cur,
                        use_post_targets=use_post_targets,
                        w_target_exc_vec=w_target_exc_vec,
                        w_target_inh_vec=w_target_inh_vec,
                        weight_decay_rate_exc=weight_decay_rate_exc,
                        weight_decay_rate_inh=weight_decay_rate_inh,
                        baseline_sum_exc=baseline_sum_exc,
                        baseline_sum_inh=baseline_sum_inh,
                        sleep_synchronized=sleep_synchronized,
                        nz_rows=nz_rows,
                        nz_cols=nz_cols,
                        baseline_sum=baseline_sum,
                        nz_rows_exc=nz_rows_exc,
                        nz_rows_inh=nz_rows_inh,
                        nz_cols_exc=nz_cols_exc,
                        nz_cols_inh=nz_cols_inh,
                    )

                    # Then apply STDP/noisy updates to capture jitter after decay
                    weights, _, _ = update_weights(
                        spikes=spikes_prev,
                        weights=weights,
                        nonzero_pre_idx=nonzero_pre_idx,
                        noisy_weights=noisy_weights,
                        weight_mean_noise=weight_mean_noise,
                        weight_var_noise=weight_var_noise,
                        N_inh=N_inh,
                        sleep=True,
                        sleep_now_inh=True,
                        sleep_now_exc=True,
                        N_x=N_x,
                        nz_cols_exc=nz_cols_exc,
                        nz_cols_inh=nz_cols_inh,
                        nz_rows_exc=nz_rows_exc,
                        nz_rows_inh=nz_rows_inh,
                        A_plus=A_plus,
                        A_minus=A_minus,
                        learning_rate_exc=learning_rate_exc,
                        learning_rate_inh=learning_rate_inh,
                        tau_LTP=tau_LTP,
                        tau_LTD=tau_LTD,
                        spike_times=spike_times,
                    )

                # Optional normalization at every step if enabled
                if normalize_weights and initial_sum_exc is not None:
                    cur_exc = np.sum(np.abs(weights[:ex, st:ih]))
                    if cur_exc > 1e-10:
                        weights[:ex, st:ih] *= initial_sum_exc / cur_exc
                    cur_inh = np.sum(np.abs(weights[ex:ih, st:ex]))
                    if cur_inh > 1e-10:
                        weights[ex:ih, st:ex] *= initial_sum_inh / cur_inh

                # Record decimated snapshots during sleep (only if tracking enabled)
                if track_weights and (sleep_iter % sleep_record_every) == 0:
                    _record_snapshot()
                    plot_time += 1

                # Advance internal counters and previous-step states
                spikes_prev = sleep_spikes_cur
                sleep_iter += 1
                sleep_time_counter += 1
                virtual_sleep_iters_epoch += 1
                t_virtual += 1

            # End of hard-pause sleep loop — record one more snapshot and mark the segment
            if track_weights and weight_tracking_sleep is not None:
                try:
                    _record_snapshot()
                    plot_time += 1
                    # Sleep segment spans from the snapshot just before loop (plot_time - (sleep_time_counter + 1))
                    # to the snapshot we just took (plot_time - 1)
                    seg_end = plot_time - 1
                    seg_start = max(
                        0, seg_end - max(1, (sleep_time_counter // sleep_record_every))
                    )
                    weight_tracking_sleep["sleep_segments"].append((seg_start, seg_end))
                except Exception:
                    pass

            # End hard-pause sleep; do not advance real t here (the loop continues below)
            # Update cached transpose after sleep modified weights
            weights_T_cache = weights[:, st:ih].T.copy()

        # update membrane potential (use maintained previous state)
        mp[t], I_syn = update_membrane_potential(
            mp=mp_prev,
            weights=weights[:, st:ih],
            spikes=spikes_prev,
            resting_potential=resting_potential,
            membrane_resistance=membrane_resistance,
            tau_m=tau_m,
            dt=dt,
            noisy_potential=noisy_potential,
            mean_noise=mean_noise,
            var_noise=var_noise,
            I_syn=I_syn,
            tau_syn=tau_syn,
            sleep_now_inh=sleep_now_inh,
            sleep_now_exc=sleep_now_exc,
            weights_T=weights_T_cache,
        )

        # update spikes array
        (
            mp[t],
            spikes[t],
            spike_times,
            spike_threshold,
            a,
        ) = update_spikes(
            st=st,
            ih=ih,
            mp=mp[t],
            dt=dt,
            a=a,
            spikes=spikes[t],
            spike_times=spike_times,
            spike_intercept=spike_intercept,
            spike_slope=spike_slope,
            noisy_threshold=noisy_threshold,
            spike_adaption=spike_adaption,
            tau_adaption=tau_adaption,
            delta_adaption=delta_adaption,
            max_mp=max_mp,
            min_mp=min_mp,
            spike_threshold=spike_threshold,
            spike_threshold_default=spike_threshold_default,
            reset_potential=reset_potential,
        )

        # update weights
        if train_weights:
            weights, sleep_now_inh, sleep_now_exc = update_weights(
                weights=weights,
                spike_times=spike_times,
                nonzero_pre_idx=nonzero_pre_idx,
                noisy_weights=noisy_weights,
                weight_mean_noise=weight_mean_noise,
                weight_var_noise=weight_var_noise,
                spikes=spikes_prev,
                N_inh=N_inh,
                sleep=sleep,
                sleep_now_inh=sleep_now_inh,
                sleep_now_exc=sleep_now_exc,
                N_x=N_x,
                nz_cols_exc=nz_cols_exc,
                nz_cols_inh=nz_cols_inh,
                nz_rows_exc=nz_rows_exc,
                nz_rows_inh=nz_rows_inh,
                A_plus=A_plus,
                A_minus=A_minus,
                learning_rate_exc=learning_rate_exc,
                learning_rate_inh=learning_rate_inh,
                tau_LTP=tau_LTP,
                tau_LTD=tau_LTD,
            )

            if not sleep:
                # Prevent excitatory weights from becoming negative and inhibitory weights from becoming positive
                np.maximum(weights[:ex, st:ih], 0.0, out=weights[:ex, st:ih])
                np.minimum(weights[ex:ih, st:ex], 0.0, out=weights[ex:ih, st:ex])

            # Update weight transpose cache periodically (every 100 steps) for accuracy
            if t % 100 == 0:
                weights_T_cache = weights[:, st:ih].T.copy()

            # After first successful compilation, report numba status once
            if t == 1:
                report_numba_status()

            # snapshots disabled

        # Apply scheduled sleep flags (non-hard-pause mode only). For hard-pause we only mark at window start.
        # Sleep is applied at the start of each batch (first sleep_window timesteps of each batch)
        if sleep and sleep_window > 0 and not sleep_hard_pause:
            batch_timestep = t % T  # Position within current batch
            if batch_timestep < sleep_window:
                sleep_now_exc = True
                sleep_now_inh = True
                slept_this_step = True
            else:
                sleep_now_exc = False
                sleep_now_inh = False

        # Mark sleep status and update progress bar
        if sleep and (sleep_now_exc or sleep_now_inh or slept_this_step):

            if t < T - 1 and spike_labels is not None:
                spike_labels[t] = -2
                sleep_amount += 1
            sleep_flag = 1
        else:
            sleep_flag = 0

        # Update postfix: sleep state on change; lightweight stats every 1000 steps
        if sleep_flag != last_sleep_flag:
            try:
                pbar.set_postfix({"sleep": sleep_flag})
            except Exception:
                pass
            last_sleep_flag = sleep_flag

        if t - last_stats_update_t >= 1000:
            try:
                # Compute means on submatrices (abs) to avoid sign confusion
                mean_exc = float(np.mean(np.abs(weights[:ex, st:ih])))
                mean_inh = float(np.mean(np.abs(weights[ex:ih, st:ex])))
                pbar.set_postfix(
                    {
                        "sleep": sleep_flag,
                        "m_exc": f"{mean_exc:.3f}",
                        "m_inh": f"{mean_inh:.3f}",
                    }
                )
            except Exception:
                pass
            last_stats_update_t = t

        # Update maintained previous-step state for next iteration
        mp_prev = mp[t]
        spikes_prev = spikes[t]

        # Sparse training snapshots (outside sleep) for context
        if track_weights and (t % train_record_every) == 0:
            _record_snapshot()
            plot_time += 1

    pbar.close()
    sleep_percent = (sleep_amount / T) * 100

    # Report virtual sleep duration for this epoch
    try:
        virtual_pct = (virtual_sleep_iters_epoch / max(1, T)) * 100
        print(
            f"Virtual sleep (epoch): {virtual_sleep_iters_epoch} iters (~{virtual_pct:.2f}% of real steps)"
        )
    except Exception:
        pass

    return (
        weights,
        spikes,
        mp,
        weights_4_plotting_exc,
        weights_4_plotting_inh,
        spike_threshold,
        spike_labels,
        sleep_percent,
        I_syn,
        spike_times,
        a,
        weight_tracking_sleep,
    )
