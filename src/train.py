from numba.typed import List
from numba import njit
from tqdm import tqdm
import numpy as np
from weight_funcs import sleep_func, spike_timing, vectorized_trace_func, normalize_weights_per_column
from plot import heatmap_spike_response


def report_numba_status():
    """Print a minimal Numba status summary (no large IR dumps)."""
    try:
        funcs = [
            ("clip_weights", clip_weights),
            ("sleep_func", sleep_func),
            ("spike_timing", spike_timing),
            ("vectorized_trace_func", vectorized_trace_func),
        ]
        msgs = []
        for name, func in funcs:
            sigs = getattr(func, "signatures", []) or []
            status = "compiled" if len(sigs) > 0 else "pending"
            msgs.append(f"{name}:{status}")
        # print("Numba status — " + ", ".join(msgs))
    except Exception:
        pass


@njit
def clip_weights(
    weights,
    nz_cols_exc,
    nz_cols_inh,
    nz_rows_exc,
    nz_rows_inh,
    min_weight_exc,
    max_weight_exc,
    min_weight_inh,
    max_weight_inh,
):
    for i_ in range(nz_rows_exc.shape[0]):
        i, j = nz_rows_exc[i_], nz_cols_exc[i_]
        if weights[i, j] < min_weight_exc:
            weights[i, j] = min_weight_exc
        elif weights[i, j] > max_weight_exc:
            weights[i, j] = max_weight_exc
    for i_ in range(nz_rows_inh.shape[0]):
        i, j = nz_rows_inh[i_], nz_cols_inh[i_]
        if weights[i, j] < min_weight_inh:
            weights[i, j] = min_weight_inh
        elif weights[i, j] > max_weight_inh:
            weights[i, j] = max_weight_inh
    return weights


def update_weights(
    weights,
    spike_times,
    timing_update,
    trace_update,
    min_weight_exc,
    max_weight_exc,
    min_weight_inh,
    max_weight_inh,
    nonzero_pre_idx,
    weight_decay_rate_exc,
    weight_decay_rate_inh,
    noisy_weights,
    weight_mean_noise,
    weight_var_noise,
    w_target_exc,
    w_target_inh,
    max_sum_exc,
    max_sum_inh,
    vectorized_trace,
    baseline_sum_exc,
    baseline_sum_inh,
    check_sleep_interval,
    sleep_synchronized,
    baseline_sum,
    max_sum,
    spikes,
    N_inh,
    N,
    N_exc,
    sleep,
    sleep_now_inh,
    sleep_now_exc,
    delta_w,
    N_x,
    nz_cols_exc,
    nz_cols_inh,
    nz_rows_exc,
    nz_rows_inh,
    t,
    dt,
    A_plus,
    A_minus,
    learning_rate_exc,
    learning_rate_inh,
    tau_LTP,
    tau_LTD,
    st=None,
    ex=None,
    ih=None,
    normalize_per_column=False,
    normalize_per_column_interval=1000,
    initial_sum_st_ex=None,
    initial_sum_ex_ex=None,
    initial_sum_ex_ih=None,
    initial_sum_ih_ex=None,
    spike_trace=None,
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
    # Clip weights before sleep to prevent zero/negative weights
    weights = clip_weights(
        weights=weights,
        nz_cols_exc=nz_cols_exc,
        nz_cols_inh=nz_cols_inh,
        nz_rows_exc=nz_rows_exc,
        nz_rows_inh=nz_rows_inh,
        min_weight_exc=min_weight_exc,
        max_weight_exc=max_weight_exc,
        min_weight_inh=min_weight_inh,
        max_weight_inh=max_weight_inh,
    )

    # Old threshold-based sleep removed; now using schedule-based sleep only

    if timing_update:
        weights = spike_timing(
            learning_rate_exc=learning_rate_exc,
            learning_rate_inh=learning_rate_inh,
            N_inh=N_inh,
            weights=weights,
            N_x=N_x,
            spikes=spikes,
            nonzero_pre_idx=nonzero_pre_idx,
            spike_trace=spike_trace,
        )

    # add noise to weights if desired
    if noisy_weights:
        delta_weight_noise = np.random.normal(
            loc=weight_mean_noise, scale=weight_var_noise, size=weights.shape
        )
        weights += delta_weight_noise

    # Per-column normalization (if enabled and initial sums provided)
    # Only normalize every N timesteps for performance
    if (normalize_per_column and st is not None and ex is not None and ih is not None 
        and t % normalize_per_column_interval == 0):
        if initial_sum_st_ex is not None:
            weights = normalize_weights_per_column(
                weights, initial_sum_st_ex, 0, st, st, ex
            )
        if initial_sum_ex_ex is not None:
            weights = normalize_weights_per_column(
                weights, initial_sum_ex_ex, st, ex, st, ex
            )
        if initial_sum_ex_ih is not None:
            weights = normalize_weights_per_column(
                weights, initial_sum_ex_ih, st, ex, ex, ih
            )
        if initial_sum_ih_ex is not None:
            weights = normalize_weights_per_column(
                weights, initial_sum_ih_ex, ex, ih, st, ex
            )

    return weights, sleep_now_inh, sleep_now_exc


def update_membrane_potential(
    mp,
    weights,
    spikes,
    noisy_potential,
    resting_potential,
    membrane_resistance_exc,
    membrane_resistance_inh,
    dt,
    N_exc,
    N_inh,
    track_stats,
    I_syn_exc,
    I_syn_inh,
    tau_syn_exc,
    tau_syn_inh,
    tau_m_exc,
    tau_m_inh,
    mean_noise,
    var_noise,
    sleep_now_inh,
    sleep_now_exc,
):
    if noisy_potential and sleep_now_inh and sleep_now_exc:
        gaussian_noise = np.random.normal(
            loc=mean_noise, scale=var_noise, size=mp.shape
        )
    else:
        gaussian_noise = np.zeros(mp.shape)

    mp_new = mp.copy()
    exc = slice(0, N_exc)
    inh = slice(N_exc, N_exc+N_inh)

    drive_exc = weights[:, exc].T @ spikes   # -> (N_exc,)
    drive_inh = weights[:, inh].T @ spikes   # -> (N_inh,)
    delta_I_syn_exc = (-I_syn_exc + drive_exc) * dt / tau_syn_exc
    delta_I_syn_inh = (-I_syn_inh + drive_inh) * dt / tau_syn_inh
    # Update synaptic current (avoid in-place modification)
    I_syn_exc += delta_I_syn_exc
    I_syn_inh += delta_I_syn_inh

    mp_delta_exc = (
        (-(mp[:N_exc] - resting_potential) + membrane_resistance_exc * I_syn_exc) / tau_m_exc * dt
    )
    mp_delta_inh = (
        (-(mp[N_exc:N_exc+N_inh] - resting_potential) + membrane_resistance_inh * I_syn_inh) / tau_m_inh * dt
    )
    mp_new[:N_exc] += mp_delta_exc + gaussian_noise[:N_exc]
    mp_new[N_exc:N_exc+N_inh] += mp_delta_inh + gaussian_noise[N_exc:N_exc+N_inh]

    if track_stats:
        return mp_new, I_syn_exc, I_syn_inh, {
            "delta_mp_ex": mp_delta_exc,
            "delta_mp_ih": mp_delta_inh,
            "delta_I_syn_ex": delta_I_syn_exc,
            "delta_I_syn_inh": delta_I_syn_inh,
        }
    else:
        return mp_new, I_syn_exc, I_syn_inh, None


def update_spikes(
    st,
    ih,
    ex,
    N_exc,
    N_inh,
    mp,
    dt,
    a,
    track_stats,
    spike_trace,
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
    tau_trace,
):

    # update spikes array
    mp = np.clip(mp, a_min=min_mp, a_max=max_mp)
    spikes[st:ih][mp > spike_threshold] = 1

    # # Add Solve's noisy membrane potential
    # if noisy_threshold:
    #     delta_potential = spike_threshold - mp
    #     p_fire = np.exp(spike_slope * delta_potential + spike_intercept) / (
    #         1 + np.exp(spike_slope * delta_potential + spike_intercept)
    #     )
    #     additional_spikes = np.random.binomial(n=1, p=p_fire)
    #     spikes[st:ih] = spikes[st:ih] | additional_spikes

    # add spike adaption
    if spike_adaption:
        # --- each timestep ---
        a += (-a / tau_adaption) * dt
        a[spikes[st:ih] == 1] += delta_adaption

        spike_threshold = spike_threshold_default + a

    mp[spikes[st:ih] == 1] = reset_potential
    spike_times = np.where(spikes == 1, 0, spike_times + 1)

    # update trace
    decay = np.exp(-dt / tau_trace)
    spike_trace *= decay
    spike_trace[spikes == 1] += 1.0

    if track_stats:
        return mp, spikes, spike_times, spike_threshold, a, spike_trace, {
            "a_ex": a[:N_exc],
            "a_ih": a[N_exc:N_exc+N_inh],
            "spike_threshold_ex": spike_threshold[:N_exc],
            "spike_threshold_ih": spike_threshold[N_exc:N_exc+N_inh],
            "spike_trace_exc": spike_trace[:N_exc],
            "spike_trace_ih": spike_trace[N_exc:N_exc+N_inh],
        }
    else:
        return mp, spikes, spike_times, spike_threshold, a, spike_trace, None


def train_network(
    weights,
    mp,
    spikes,
    noisy_potential,
    resting_potential,
    membrane_resistance_exc,
    membrane_resistance_inh,
    spike_times,
    spike_trace,
    min_weight_exc,
    max_weight_exc,
    min_weight_inh,
    max_weight_inh,
    train_weights,
    weight_decay,
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
    check_sleep_interval,
    w_target_exc,
    w_target_inh,
    baseline_sum_exc,
    baseline_sum_inh,
    max_sum_exc,
    max_sum_inh,
    dt,
    N,
    run,
    A_plus,
    A_minus,
    tau_m_exc,
    tau_m_inh,
    sleep,
    spike_adaption,
    tau_adaption,
    delta_adaption,
    spike_threshold_default,
    save_plots,
    reset_potential,
    spike_slope,
    spike_intercept,
    noisy_threshold,
    noisy_weights,
    vectorized_trace,
    spike_labels,
    weight_mean_noise,
    weight_var_noise,
    dataset,
    timing_update,
    trace_update,
    interval,
    N_x,
    T,
    tau_trace,
    tau_syn_exc,
    tau_syn_inh,
    beta,
    mean_noise,
    var_noise,
    num_exc,
    num_inh,
    I_syn_exc,
    I_syn_inh,
    a,
    spike_threshold,
    sleep_ratio=0.0,
    normalize_weights=False,
    initial_sum_exc=None,
    initial_sum_inh=None,
    initial_sum_total=None,
    normalize_per_column=False,
    normalize_per_column_interval=1000,
    initial_sum_st_ex=None,
    initial_sum_ex_ex=None,
    initial_sum_ex_ih=None,
    initial_sum_ih_ex=None,
    track_stats=False,
    # Hard-pause sleep settings
    sleep_hard_pause: bool = True,
    sleep_epsilon: float = 1e-8,
    sleep_tol_frac: float = 1e-3,
    sleep_max_iters: int = 5000,
    on_timeout: str = "scale_to_target",  # one of {"scale_to_target","extend","give_up"}
    sleep_mode: str = "static",  # one of {"static","group","post"}
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

    if track_stats:
        # track mp vars
        delta_mp_ex = np.zeros((N_exc,T))
        delta_mp_ih = np.zeros((N_inh,T))

        # track synaptic current
        delta_I_syn_ex = np.zeros((N_exc,T))
        delta_I_syn_ih = np.zeros((N_inh,T))
        I_syn_ex = np.zeros((N_exc,T))
        I_syn_ih = np.zeros((N_inh,T))

        # track adaptive spiking threshold
        a_ex = np.zeros((N_exc,T))
        a_ih = np.zeros((N_inh,T))
        spike_threshold_ex = np.zeros((N_exc,T))
        spike_threshold_ih = np.zeros((N_inh,T))
        spike_trace_ex = np.zeros((N_exc,T))
        spike_trace_ih = np.zeros((N_inh,T))

        # # track STDP update size
        # x_pre = np.zeros((N,T))
        # x_post = np.zeros((N-N_x,T))
        # delta_w_exc = np.zeros((T))
        # delta_w_inh = np.zeros((T))
        # weights_track_ex = np.zeros((T))
        # weights_track_ih = np.zeros((T))

    # Precompute scheduled sleep window per interval
    # e.g., sleep_ratio=0.1 means 10% of each interval is sleep
    if sleep and sleep_ratio is not None and sleep_ratio > 0.0:
        sleep_window = max(1, int(round(check_sleep_interval * sleep_ratio)))
    else:
        sleep_window = 0

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
    # - tracks a small, fixed set of non-zero synapses for both E and I pathways
    # - records snapshots just before sleep, during sleep (decimated), just after sleep, and sparsely during training
    weight_tracking_sleep = {
        "st_ex_mean": [],
        "ex_ex_mean": [],
        "ex_ih_mean": [],
        "ih_ex_mean": [],
        "st_ex_min": [],
        "ex_ex_min": [],
        "ex_ih_min": [],
        "ih_ex_min": [],
        "st_ex_max": [],
        "ex_ex_max": [],
        "ex_ih_max": [],
        "ih_ex_max": [],
        "st_ex_samples": [],
        "ex_ex_samples": [],
        "ex_ih_samples": [],
        "ih_ex_samples": [],
    }
    # Select a small number of active synapses to track (prefer non-zero)
    rng = np.random.default_rng(42)
    K_st_ex = 10
    K_ex_ex = 10
    K_ex_ih = 10
    K_ih_ex = 10
    # Candidates within the decayed submatrices
    st_ex_cand = np.argwhere(weights[:st, st:ex] != 0)
    ex_ex_cand = np.argwhere(weights[st:ex, st:ex] != 0)
    ex_ih_cand = np.argwhere(weights[st:ex, ex:ih] != 0)
    ih_ex_cand = np.argwhere(weights[ex:ih, st:ex] != 0)
    # Select random indices from candidates
    st_ex_idx = rng.choice(len(st_ex_cand), size=K_st_ex, replace=False)
    ex_ex_idx = rng.choice(len(ex_ex_cand), size=K_ex_ex, replace=False)
    ex_ih_idx = rng.choice(len(ex_ih_cand), size=K_ex_ih, replace=False)
    ih_ex_idx = rng.choice(len(ih_ex_cand), size=K_ih_ex, replace=False)
    # Get selected indices
    st_ex_sel = st_ex_cand[st_ex_idx]
    ex_ex_sel = ex_ex_cand[ex_ex_idx]
    ex_ih_sel = ex_ih_cand[ex_ih_idx]
    ih_ex_sel = ih_ex_cand[ih_ex_idx]   
    # Snapshot cadence
    plot_time = 0  # increases by 1 per recorded snapshot (training or sleep)
    sleep_record_every = (
        max(1, sleep_window // 50) if sleep_window > 0 else 10
    )  # up to ~50 points per sleep

    def _record_snapshot():
        # Compute group stats over absolute weights (avoid sign confusion)
        try:
            W_st_ex = weights[:st, st:ex]
            W_ex_ex = weights[st:ex, st:ex]
            W_ex_ih = weights[st:ex, ex:ih]
            W_ih_ex = weights[ex:ih, st:ex]

            W_st_ex_sp = W_st_ex[st_ex_sel[:, 0], st_ex_sel[:, 1]]
            W_ex_ex_sp = W_ex_ex[ex_ex_sel[:, 0], ex_ex_sel[:, 1]]
            W_ex_ih_sp = W_ex_ih[ex_ih_sel[:, 0], ex_ih_sel[:, 1]]
            W_ih_ex_sp = W_ih_ex[ih_ex_sel[:, 0], ih_ex_sel[:, 1]]

            # remove nonzero weights
            W_st_ex = W_st_ex[W_st_ex != 0]
            W_ex_ex = W_ex_ex[W_ex_ex != 0]
            W_ex_ih = W_ex_ih[W_ex_ih != 0]
            W_ih_ex = W_ih_ex[W_ih_ex != 0]


            # st-ex stats
            if W_st_ex.size > 0:
                weight_tracking_sleep["st_ex_mean"].append(float(W_st_ex.mean()))
                weight_tracking_sleep["st_ex_min"].append(float(W_st_ex.min()))
                weight_tracking_sleep["st_ex_max"].append(float(W_st_ex.max()))
                weight_tracking_sleep["st_ex_samples"].append(W_st_ex_sp.tolist())
            else:
                weight_tracking_sleep["st_ex_mean"].append(0.0)
                weight_tracking_sleep["st_ex_min"].append(0.0)
                weight_tracking_sleep["st_ex_max"].append(0.0)
                weight_tracking_sleep["st_ex_samples"].append([0.0] * K_st_ex)
            # ex-ex stats
            if W_ex_ex.size > 0:
                weight_tracking_sleep["ex_ex_mean"].append(float(W_ex_ex.mean()))
                weight_tracking_sleep["ex_ex_min"].append(float(W_ex_ex.min()))
                weight_tracking_sleep["ex_ex_max"].append(float(W_ex_ex.max()))
                weight_tracking_sleep["ex_ex_samples"].append(W_ex_ex_sp.tolist())
            else:
                weight_tracking_sleep["ex_ex_mean"].append(0.0)
                weight_tracking_sleep["ex_ex_min"].append(0.0)
                weight_tracking_sleep["ex_ex_max"].append(0.0)
                weight_tracking_sleep["ex_ex_samples"].append([0.0] * K_ex_ex)
            # ex-ih stats
            if W_ex_ih.size > 0:
                weight_tracking_sleep["ex_ih_mean"].append(float(W_ex_ih.mean()))
                weight_tracking_sleep["ex_ih_min"].append(float(W_ex_ih.min()))
                weight_tracking_sleep["ex_ih_max"].append(float(W_ex_ih.max()))
                weight_tracking_sleep["ex_ih_samples"].append(W_ex_ih_sp.tolist())
            else:
                weight_tracking_sleep["ex_ih_mean"].append(0.0)
                weight_tracking_sleep["ex_ih_min"].append(0.0)
                weight_tracking_sleep["ex_ih_max"].append(0.0)
                weight_tracking_sleep["ex_ih_samples"].append([0.0] * K_ex_ih)
            # ih-ex stats
            if W_ih_ex.size > 0:
                weight_tracking_sleep["ih_ex_mean"].append(float(W_ih_ex.mean()))
                weight_tracking_sleep["ih_ex_min"].append(float(W_ih_ex.min()))
                weight_tracking_sleep["ih_ex_max"].append(float(W_ih_ex.max()))
                weight_tracking_sleep["ih_ex_samples"].append(W_ih_ex_sp.tolist())
            else:
                weight_tracking_sleep["ih_ex_mean"].append(0.0)
                weight_tracking_sleep["ih_ex_min"].append(0.0)
                weight_tracking_sleep["ih_ex_max"].append(0.0)
                weight_tracking_sleep["ih_ex_samples"].append([0.0] * K_ih_ex)

            return True
        except Exception:
            return False

    pbar = tqdm(range(1, T), desc=desc, leave=False)
    last_sleep_flag = -1  # unknown
    last_stats_update_t = -1000
    # Initial snapshot
    _record_snapshot()
    plot_time += 1
    num_steps = int(T/10)
    iterations = 100
    _track_stats = False
    num = 0
    for t in pbar:
        if t % num_steps == 0 and save_plots:
            if track_stats:
                _track_stats = True
            heatmap_spike_response(
                spikes[t - iterations - 1 : t - 1, st:ex],
                spikes[t - iterations - 1 : t - 1, :st],
                spikes[t - iterations - 1 : t - 1, ex:],
                spike_labels[t - 1],
                dataset=dataset,
                run=run,
                num=num,
                weight_tracking_sleep=weight_tracking_sleep,
                weights_st_ex=weights[:st, st:ex],
                weights_ex_ex=weights[st:ex, st:ex],
                weights_ex_ih=weights[st:ex, ex:ih],
                weights_ih_ex=weights[ex:ih, st:ex],
            )
            # update num
            num += 1
        # Reset sleep flags for this timestep
        sleep_now_inh = False
        sleep_now_exc = False

        # Trigger hard-pause only at the start of a window
        is_window_start = (
            sleep and sleep_window > 0 and ((t % check_sleep_interval) == 0)
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
                            min_weight_exc=min_weight_exc,
                            max_weight_exc=max_weight_exc,
                            min_weight_inh=min_weight_inh,
                            max_weight_inh=max_weight_inh,
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
                mp_prev, I_syn_exc, I_syn_inh, tracking_package = update_membrane_potential(
                    mp=mp_prev,
                    weights=weights[:, st:ih],
                    spikes=spikes_prev,
                    resting_potential=resting_potential,
                    membrane_resistance_exc=membrane_resistance_exc,
                    membrane_resistance_inh=membrane_resistance_inh,
                    tau_m_exc=tau_m_exc,
                    tau_m_inh=tau_m_inh,
                    dt=dt,
                    N_exc=N_exc,
                    N_inh=N_inh,
                    track_stats=track_stats,
                    noisy_potential=noisy_potential,
                    mean_noise=mean_noise,
                    var_noise=var_noise,
                    I_syn_exc=I_syn_exc,
                    I_syn_inh=I_syn_inh,
                    tau_syn_exc=tau_syn_exc,
                    tau_syn_inh=tau_syn_inh,
                    sleep_now_inh=sleep_now_inh,
                    sleep_now_exc=sleep_now_exc,
                )
                if track_stats:
                    # track mp vars
                    delta_mp_ex[:, t] = tracking_package["delta_mp_ex"]
                    delta_mp_ih[:, t] = tracking_package["delta_mp_ih"]

                    # track synaptic current
                    delta_I_syn_ex[:, t] = tracking_package["delta_I_syn_ex"]
                    delta_I_syn_ih[:, t] = tracking_package["delta_I_syn_ih"]

                    I_syn_ex[:,t] = I_syn_exc
                    I_syn_ih[:,t] = I_syn_inh

                # Prepare current spikes vector (no sensory spikes during sleep)
                sleep_spikes_cur = np.zeros_like(spikes_prev)

                # Update spikes and thresholds
                (
                    mp_prev,
                    sleep_spikes_cur,
                    spike_times,
                    spike_threshold,
                    a,
                    spike_trace,
                    tracking_package,
                ) = update_spikes(
                    N_exc=N_exc,
                    N_inh=N_inh,
                    st=st,
                    ih=ih,
                    ex=ex,
                    mp=mp_prev,
                    dt=dt,
                    a=a,
                    track_stats=track_stats,
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
                    spike_trace=spike_trace,
                    spike_threshold=spike_threshold,
                    spike_threshold_default=spike_threshold_default,
                    reset_potential=reset_potential,
                    tau_trace=tau_trace,
                )
                if track_stats:
                    # track adaptive spiking threshold
                    a_ex[:, t] = tracking_package["a_ex"]
                    a_ih[:, t] = tracking_package["a_ih"]
                    spike_threshold_ex[:, t] = tracking_package["spike_threshold_ex"]
                    spike_threshold_ih[:, t] = tracking_package["spike_threshold_ih"]
                    spike_trace_ex[:, t] = tracking_package["spike_trace_exc"]
                    spike_trace_ih[:, t] = tracking_package["spike_trace_ih"]

                # Update weights once per internal sleep iteration (use previous spikes_prev)
                if train_weights:
                    weights, _, _  = update_weights(
                        spikes=spikes_prev,
                        weights=weights,
                        max_sum_exc=max_sum_exc,
                        max_sum_inh=max_sum_inh,
                        baseline_sum_exc=baseline_sum_exc,
                        baseline_sum_inh=baseline_sum_inh,
                        check_sleep_interval=check_sleep_interval,
                        sleep=True,
                        nonzero_pre_idx=nonzero_pre_idx,
                        N_x=N_x,
                        vectorized_trace=vectorized_trace,
                        delta_w=delta_w,
                        N_exc=N_exc,
                        timing_update=timing_update,
                        trace_update=trace_update,
                        spike_times=spike_times,
                        spike_trace=spike_trace,
                        weight_decay_rate_exc=weight_decay_rate_exc,
                        weight_decay_rate_inh=weight_decay_rate_inh,
                        min_weight_exc=min_weight_exc,
                        max_weight_exc=max_weight_exc,
                        min_weight_inh=min_weight_inh,
                        max_weight_inh=max_weight_inh,
                        noisy_weights=noisy_weights,
                        weight_mean_noise=weight_mean_noise,
                        weight_var_noise=weight_var_noise,
                        w_target_exc=w_target_exc_cur,
                        w_target_inh=w_target_inh_cur,
                        sleep_now_inh=True,
                        sleep_now_exc=True,
                        t=t_virtual,
                        N=N,
                        sleep_synchronized=sleep_synchronized,
                        baseline_sum=baseline_sum,
                        max_sum=max_sum,
                        nz_cols_exc=nz_cols_exc,
                        nz_cols_inh=nz_cols_inh,
                        nz_rows_exc=nz_rows_exc,
                        nz_rows_inh=nz_rows_inh,
                        N_inh=N_inh,
                        A_plus=A_plus,
                        A_minus=A_minus,
                        learning_rate_exc=learning_rate_exc,
                        learning_rate_inh=learning_rate_inh,
                        tau_LTP=tau_LTP,
                        tau_LTD=tau_LTD,
                        dt=dt,
                        st=st,
                        ex=ex,
                        ih=ih,
                        normalize_per_column=normalize_per_column,
                        normalize_per_column_interval=normalize_per_column_interval,
                        initial_sum_st_ex=initial_sum_st_ex,
                        initial_sum_ex_ex=initial_sum_ex_ex,
                        initial_sum_ex_ih=initial_sum_ex_ih,
                        initial_sum_ih_ex=initial_sum_ih_ex,
            )

                    # Apply exponential decay toward targets during sleep (combine with STDP)
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
                        baseline_sum=baseline_sum,
                        nz_rows_exc=nz_rows_exc,
                        nz_rows_inh=nz_rows_inh,
                        nz_cols_exc=nz_cols_exc,
                        nz_cols_inh=nz_cols_inh,
                        nz_rows=nz_rows,
                        nz_cols=nz_cols,
                        st=st,
                        ex=ex,
                    )

                # Record decimated snapshots during sleep
                if (sleep_iter % sleep_record_every) == 0:
                    _record_snapshot()
                    plot_time += 1

                # Advance internal counters and previous-step states
                spikes_prev = sleep_spikes_cur
                sleep_iter += 1
                sleep_time_counter += 1
                virtual_sleep_iters_epoch += 1
                t_virtual += 1

            # End of hard-pause sleep loop — record one more snapshot and mark the segment
            try:
                _record_snapshot()
                plot_time += 1
                # Sleep segment spans from the snapshot just before loop (plot_time - (sleep_time_counter + 1))
                # to the snapshot we just took (plot_time - 1)
                seg_end = plot_time - 1
                seg_start = max(
                    0, seg_end - max(1, (sleep_time_counter // sleep_record_every))
                )
                # weight_tracking_sleep["sleep_segments"].append((seg_start, seg_end))
            except Exception:
                pass

            # End hard-pause sleep; do not advance real t here (the loop continues below)
        # update membrane potential (use maintained previous state)
        mp[t], I_syn_exc, I_syn_inh, tracking_package = update_membrane_potential(
            mp=mp_prev,
            weights=weights[:, st:ih],
            spikes=spikes_prev,
            resting_potential=resting_potential,
            membrane_resistance_exc=membrane_resistance_exc,
            membrane_resistance_inh=membrane_resistance_inh,
            tau_m_exc=tau_m_exc,
            tau_m_inh=tau_m_inh,
            dt=dt,
            N_exc=N_exc,
            N_inh=N_inh,
            track_stats=track_stats,
            noisy_potential=noisy_potential,
            mean_noise=mean_noise,
            var_noise=var_noise,
            I_syn_exc=I_syn_exc,
            I_syn_inh=I_syn_inh,
            tau_syn_exc=tau_syn_exc,
            tau_syn_inh=tau_syn_inh,
            sleep_now_inh=sleep_now_inh,
            sleep_now_exc=sleep_now_exc,
        )
        if track_stats:
            # track synaptic current
            delta_I_syn_ex[:, t] = tracking_package["delta_I_syn_ex"]
            delta_I_syn_ih[:, t] = tracking_package["delta_I_syn_ih"]

            I_syn_ex[:,t] = I_syn_exc
            I_syn_ih[:,t] = I_syn_inh

        # update spikes array
        (
            mp[t],
            spikes[t],
            spike_times,
            spike_threshold,
            a,
            spike_trace,
            tracking_package,
        ) = update_spikes(
            N_exc=N_exc,
            N_inh=N_inh,
            st=st,
            ih=ih,
            ex=ex,  
            mp=mp[t],
            dt=dt,  
            a=a,
            spikes=spikes[t],
            spike_times=spike_times,
            track_stats=track_stats,
            spike_intercept=spike_intercept,
            spike_slope=spike_slope,
            noisy_threshold=noisy_threshold,
            spike_adaption=spike_adaption,
            tau_adaption=tau_adaption,
            delta_adaption=delta_adaption,
            spike_trace=spike_trace,
            max_mp=max_mp,
            min_mp=min_mp,
            spike_threshold=spike_threshold,
            spike_threshold_default=spike_threshold_default,
            reset_potential=reset_potential,
            tau_trace=tau_trace,
        )
        if track_stats:
            # track adaptive spiking threshold
            a_ex[:, t] = tracking_package["a_ex"]
            a_ih[:, t] = tracking_package["a_ih"]
            spike_threshold_ex[:, t] = tracking_package["spike_threshold_ex"]
            spike_threshold_ih[:, t] = tracking_package["spike_threshold_ih"]
            spike_trace_ex[:, t] = tracking_package["spike_trace_exc"]
            spike_trace_ih[:, t] = tracking_package["spike_trace_ih"]

        # update weights
        if train_weights:
            weights, sleep_now_inh, sleep_now_exc = update_weights(
                spikes=spikes_prev,
                weights=weights,
                max_sum_exc=max_sum_exc,
                max_sum_inh=max_sum_inh,
                baseline_sum_exc=baseline_sum_exc,
                baseline_sum_inh=baseline_sum_inh,
                check_sleep_interval=check_sleep_interval,
                sleep=sleep,
                nonzero_pre_idx=nonzero_pre_idx,
                N_x=N_x,
                vectorized_trace=vectorized_trace,
                delta_w=delta_w,
                N_exc=N_exc,
                timing_update=timing_update,
                trace_update=trace_update,
                spike_times=spike_times,
                spike_trace=spike_trace,
                weight_decay_rate_exc=weight_decay_rate_exc,
                weight_decay_rate_inh=weight_decay_rate_inh,
                min_weight_exc=min_weight_exc,
                max_weight_exc=max_weight_exc,
                min_weight_inh=min_weight_inh,
                max_weight_inh=max_weight_inh,
                noisy_weights=noisy_weights,
                weight_mean_noise=weight_mean_noise,
                weight_var_noise=weight_var_noise,
                w_target_exc=w_target_exc,
                w_target_inh=w_target_inh,
                sleep_now_inh=sleep_now_inh,
                sleep_now_exc=sleep_now_exc,
                t=t,
                N=N,
                sleep_synchronized=sleep_synchronized,
                baseline_sum=baseline_sum,
                max_sum=max_sum,
                nz_cols_exc=nz_cols_exc,
                nz_cols_inh=nz_cols_inh,
                nz_rows_exc=nz_rows_exc,
                nz_rows_inh=nz_rows_inh,
                N_inh=N_inh,
                A_plus=A_plus,
                A_minus=A_minus,
                learning_rate_exc=learning_rate_exc,
                learning_rate_inh=learning_rate_inh,
                tau_LTP=tau_LTP,
                tau_LTD=tau_LTD,
                dt=dt,
                st=st,
                ex=ex,
                ih=ih,
                normalize_per_column=normalize_per_column,
                normalize_per_column_interval=normalize_per_column_interval,
                initial_sum_st_ex=initial_sum_st_ex,
                initial_sum_ex_ex=initial_sum_ex_ex,
                initial_sum_ex_ih=initial_sum_ex_ih,
                initial_sum_ih_ex=initial_sum_ih_ex,
            )

        # Apply scheduled sleep flags (non-hard-pause mode only). For hard-pause we only mark at window start.
        if sleep and sleep_window > 0 and not sleep_hard_pause:
            if (t % check_sleep_interval) < sleep_window:
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
        if _track_stats == True:
            _record_snapshot()
            plot_time += 1

    sleep_percent = (sleep_amount / T) * 100

    # Report virtual sleep duration for this epoch
    try:
        virtual_pct = (virtual_sleep_iters_epoch / max(1, T)) * 100
    except Exception:
        pass
    
    if track_stats:
        # compute the mean of the trackers
        mean_delta_mp_ex = np.mean(delta_mp_ex, axis=1)
        mean_delta_mp_ih = np.mean(delta_mp_ih, axis=1)
        mean_delta_I_syn_ex = np.mean(delta_I_syn_ex, axis=1)
        mean_delta_I_syn_ih = np.mean(delta_I_syn_ih, axis=1)
        mean_a_ex = np.mean(a_ex, axis=1)
        mean_a_ih = np.mean(a_ih, axis=1)
        mean_spike_threshold_ex = np.mean(spike_threshold_ex, axis=1)
        mean_spike_threshold_ih = np.mean(spike_threshold_ih, axis=1)
        mean_spike_trace_ex = np.mean(spike_trace_ex, axis=1)
        mean_spike_trace_ih = np.mean(spike_trace_ih, axis=1)
        spikes_st = spikes[:,:st].mean(axis=0)
        spikes_ex = spikes[:,st:ex].mean(axis=0)
        spikes_ih = spikes[:,ex:ih].mean(axis=0)
        print(f"Mean delta mp ex: {mean_delta_mp_ex.mean()}")
        print(f"Mean delta mp ih: {mean_delta_mp_ih.mean()}")
        print(f"Mean delta I syn ex: {mean_delta_I_syn_ex.mean()}")
        print(f"Mean delta I syn ih: {mean_delta_I_syn_ih.mean()}")
        print(f"Mean membrane potential ex: {mp[t][:N_exc].mean()}")
        print(f"Mean membrane potential ih: {mp[t][N_exc:].mean()}")
        print(f"Mean I syn ex: {I_syn_ex[:,t].mean()}")
        print(f"Mean I syn ih: {I_syn_ih[:,t].mean()}")
        print(f"Mean a ex: {mean_a_ex.mean()}")
        print(f"Mean a ih: {mean_a_ih.mean()}")
        print(f"Mean spike threshold ex: {mean_spike_threshold_ex.mean()}")
        print(f"Mean spike threshold ih: {mean_spike_threshold_ih.mean()}")
        print(f"Mean spikes st: {spikes_st.mean()}")
        print(f"Mean spikes ih: {spikes_ih.mean()}")
        print(f"Mean spikes ex: {spikes_ex.mean()}")
        print(f"Mean spike trace ex: {mean_spike_trace_ex.mean()}")
        print(f"Mean spike trace ih: {mean_spike_trace_ih.mean()}")
    return (
        weights,
        spikes,
        mp,
        weights_4_plotting_exc,
        weights_4_plotting_inh,
        spike_threshold,
        max_sum_inh,
        max_sum_exc,
        spike_labels,
        sleep_percent,
        I_syn_exc,
        I_syn_inh,
        spike_times,
        a,
        weight_tracking_sleep,
        spike_trace,
    )
