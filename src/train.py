from numba.typed import List
from numba import njit
from tqdm import tqdm
from numba import njit, prange
import threading
import numpy as np

from weight_funcs import (
    trace_STDP,
    normalize_weights_per_column,
)
from plot import heatmap_spike_response


@njit(cache=True, parallel=True)
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
    w_max,
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
    mu_weight,
    A_plus,
    A_minus,
    learning_rate_exc,
    learning_rate_inh,
    tau_LTP,
    tau_LTD,
    track_weights,
    normalize_now,
    update_weights_now,
    x_tar_se,
    x_tar_ex,
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

    weights, m_x_pre, m_first_term, m_delta_w = trace_STDP(
        learning_rate_exc=learning_rate_exc,
        learning_rate_inh=learning_rate_inh,
        N_inh=N_inh,
        weights=weights,
        N_x=N_x,
        w_max=w_max,
        mu_weight=mu_weight,
        spikes=spikes,
        x_tar_se=x_tar_se,
        x_tar_ex=x_tar_ex,
        nonzero_pre_idx=nonzero_pre_idx,
        spike_trace=spike_trace,
        track_weights=track_weights,
    )

    # Per-column normalization (if enabled and initial sums provided)
    # Only normalize every N timesteps for performance
    if normalize_per_column and normalize_now:
        if initial_sum_st_ex is not None:
            weights = normalize_weights_per_column(
                weights, initial_sum_st_ex, 0, st, st, ex
            )
        if initial_sum_ex_ex is not None:
            weights = normalize_weights_per_column(
                weights, initial_sum_ex_ex, st, ex, st, ex
            )

    if track_weights:
        return weights, sleep_now_inh, sleep_now_exc, m_x_pre, m_first_term, m_delta_w

    return weights, sleep_now_inh, sleep_now_exc, None, None, None


@njit(parallel=True, cache=True)
def update_membrane_potential(
    mp,
    mp_new,
    weights_exc,
    weights_inh,
    spikes,
    noisy_potential,
    resting_potential,
    membrane_resistance_exc,
    membrane_resistance_inh,
    dt,
    st,
    ex,
    track_stats,
    N_exc,
    N_inh,
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
    # Pre-allocate tracking arrays
    if track_stats:
        delta_mp_ex = np.zeros(N_exc)
        delta_mp_ih = np.zeros(N_inh)
        delta_I_syn_ex = np.zeros(N_exc)
        delta_I_syn_ih = np.zeros(N_inh)
    else:
        delta_mp_ex = np.empty(0)
        delta_mp_ih = np.empty(0)
        delta_I_syn_ex = np.empty(0)
        delta_I_syn_ih = np.empty(0)

    for i in prange(N_exc):  # postsynaptic neurons
        drive = 0.0
        for j in range(spikes.shape[0]):  # presynaptic neurons
            if spikes[j] != 0:
                drive += weights_exc[i, j]
        d_I = (-I_syn_exc[i] + drive) * dt / tau_syn_exc
        I_syn_exc[i] += d_I
        d_mp = (
            (-(mp[i] - resting_potential) + membrane_resistance_exc * I_syn_exc[i])
            / tau_m_exc
            * dt
        )
        mp_new[i] = mp[i] + d_mp
        if noisy_potential and sleep_now_inh and sleep_now_exc:
            mp_new[i] += np.random.normal(mean_noise, var_noise)
        if track_stats:
            delta_mp_ex[i] = d_mp
            delta_I_syn_ex[i] = d_I

    for i in prange(N_inh):
        drive = 0.0
        for j in range(st, ex):
            if spikes[j] != 0:
                drive += weights_inh[i, j]
        d_I = (-I_syn_inh[i] + drive) * dt / tau_syn_inh
        I_syn_inh[i] += d_I
        idx = N_exc + i
        d_mp = (
            (-(mp[idx] - resting_potential) + membrane_resistance_inh * I_syn_inh[i])
            / tau_m_inh
            * dt
        )
        mp_new[idx] = mp[idx] + d_mp
        if noisy_potential and sleep_now_inh and sleep_now_exc:
            mp_new[idx] += np.random.normal(mean_noise, var_noise)
        if track_stats:
            delta_mp_ih[i] = d_mp
            delta_I_syn_ih[i] = d_I

    return (
        mp_new,
        I_syn_exc,
        I_syn_inh,
        delta_mp_ex,
        delta_mp_ih,
        delta_I_syn_ex,
        delta_I_syn_ih,
    )


@njit(parallel=True, cache=True)
def update_spikes(
    st,
    ih,
    N_exc,
    N_inh,
    mp,
    dt,
    a,
    track_stats,
    spike_trace,
    spikes,
    max_mp,
    min_mp,
    spike_adaption,
    tau_adaption,
    delta_adaption,
    spike_threshold,
    spike_threshold_default,
    reset_potential,
    tau_trace,
):
    decay = np.exp(-dt / tau_trace)
    n_total = ih - st

    for j in prange(n_total):
        if mp[j] < min_mp:
            mp[j] = min_mp
        elif mp[j] > max_mp:
            mp[j] = max_mp
        if mp[j] > spike_threshold[j]:
            spikes[st + j] = 1
            mp[j] = reset_potential

    if spike_adaption:
        for j in range(n_total):
            a[j] += (-a[j] / tau_adaption) * dt
            if spikes[st + j] == 1:
                a[j] += delta_adaption
            spike_threshold[j] = spike_threshold_default[j] + a[j]

    for j in prange(N_exc + st):
        idx = j
        if spikes[idx] == 1:
            spike_trace[idx] = spike_trace[idx] * decay + 1.0
        else:
            spike_trace[idx] *= decay

    return (
        mp,
        spikes,
        spike_threshold,
        a,
        spike_trace,
    )


def update_x_tar(spike_trace, N_x):
    x_tar_se = np.mean(spike_trace[:N_x], axis=0)
    x_tar_ex = np.mean(spike_trace[N_x:], axis=0)
    return x_tar_se, x_tar_ex


def _plot_background(kwargs):
    heatmap_spike_response(**kwargs)


def train_network(
    weights,
    mp,
    spikes,
    noisy_potential,
    resting_potential,
    membrane_resistance_exc,
    membrane_resistance_inh,
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
    w_max,
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
    track_weights,
    beta,
    mean_noise,
    var_noise,
    num_exc,
    num_inh,
    mu_weight,
    I_syn_exc,
    I_syn_inh,
    a,
    time_per_item,
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
    x_tar_se=None,
    x_tar_ex=None,
    # Hard-pause sleep settings
    sleep_hard_pause: bool = True,
    sleep_epsilon: float = 1e-8,
    sleep_tol_frac: float = 1e-3,
    sleep_max_iters: int = 5000,
    on_timeout: str = "scale_to_target",  # one of {"scale_to_target","extend","give_up"}
    sleep_mode: str = "static",  # one of {"static","group","post"}
):

    # Enforce consistent dtypes for Numba cache stability
    weights = np.asarray(weights, dtype=np.float64)
    spike_trace = np.asarray(spike_trace, dtype=np.float64)
    mp = np.asarray(mp, dtype=np.float64)
    I_syn_exc = np.asarray(I_syn_exc, dtype=np.float64)
    I_syn_inh = np.asarray(I_syn_inh, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    spike_threshold = np.asarray(spike_threshold, dtype=np.float64)
    N_x = np.int64(N_x)
    N_exc = np.int64(N_exc)
    N_inh = np.int64(N_inh)
    spike_threshold_default_arr = np.full(
        N_exc + N_inh, fill_value=spike_threshold_default, dtype=np.float64
    )

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
    sleep_now_inh = False
    sleep_now_exc = False

    if track_stats:
        delta_mp_ex = 0.0
        delta_mp_ih = 0.0
        mp_ex = 0.0
        mp_ih = 0.0
        delta_I_syn_ex = 0.0
        delta_I_syn_ih = 0.0
        I_syn_ex = 0.0
        I_syn_ih = 0.0
        a_ex = 0.0
        spike_threshold_ex = 0.0
        spike_trace_ex = 0.0
        track_count = 0
        x_tar_count = 0
        x_tar_sum_se = 0
        x_tar_sum_ex = 0
    if track_weights:
        first_term_sum = 0
        delta_w_sum = 0
        x_pre_sum = 0

    # Update membrane potentials using previous step spikes
    mp_new = np.zeros_like(mp)
    weights_exc = np.ascontiguousarray(weights[:, st:ex].T)
    weights_inh = np.ascontiguousarray(weights[:, ex:ih].T)

    # Suppose weights is your initial 2D numpy array of weights.
    # Here, we assume that the columns correspond to post-neurons.
    if train_weights:
        desc = "Training network:"
    else:
        desc = "Testing network:"

    # Compute for neurons N_x to N_post-1
    nonzero_pre_idx = List()
    for i in range(st, ih):
        pre_idx = np.nonzero(weights[:, i])[0]
        nonzero_pre_idx.append(pre_idx.astype(np.int64))

    nonzero_pre_idx_exc = List()
    for i in range(st, ex):
        pre_idx = np.nonzero(weights[:ex, i])[0]
        nonzero_pre_idx_exc.append(pre_idx.astype(np.int64))

    # Maintain previous-step state explicitly so we can evolve during hard-pause sleep
    mp_prev = mp.copy()
    spikes_prev = spikes[0].copy()

    # Get selected indices
    # Snapshot cadence
    plot_time = 0  # increases by 1 per recorded snapshot (training or sleep)

    pbar = tqdm(range(1, T), desc=desc, leave=False, mininterval=1.0)
    # Initial snapshot
    plot_time += 1
    num_steps = max(1, int((T * 100) // time_per_item))
    update_weight_freq = 100  # max(1, int(T // (time_per_item)))
    normalize_freq = int(update_weight_freq * 100)
    iterations = 100
    plot_threads = []
    _track_stats = False
    num = 0
    import psutil, os

    print(f"The network will update its weights every {update_weight_freq} steps")

    process = psutil.Process(os.getpid())
    print(f"Memory before training: {process.memory_info().rss / 1024**2:.0f} MB")
    update_weights_now = False
    normalize_now = False
    if x_tar_se is None or x_tar_ex is None:
        # precompute x_tar_se and x_tar_ex
        x_tar_se, x_tar_ex = update_x_tar(
            spike_trace=spike_trace,
            N_x=N_x,
        )

    # create heatmap spike plot before training to verify that it starts off as wrong
    for t in pbar:
        if t % update_weight_freq == 0 and train_weights:
            update_weights_now = True
            # update x_tar as often as we update weights
            x_tar_se, x_tar_ex = update_x_tar(
                spike_trace=spike_trace,
                N_x=N_x,
            )
            # update x_tar tracker
            if track_stats:
                x_tar_sum_se += x_tar_se.mean()
                x_tar_sum_ex += x_tar_ex.mean()
                x_tar_count += 1
        if t % normalize_freq == 0:
            normalize_now = True
        if t % num_steps == 0:
            if track_stats:
                _track_stats = True
            if save_plots:
                plot_kwargs = dict(
                    spikes_exc=spikes[t - iterations - 1 : t - 1, st:ex].copy(),
                    spikes_in=spikes[t - iterations - 1 : t - 1, :st].copy(),
                    spikes_ih=spikes[t - iterations - 1 : t - 1, ex:].copy(),
                    label=spike_labels[t - 1],
                    spike_trace=spike_trace.copy(),
                    dataset=dataset,
                    run=run,
                    num=num,
                    st=st,
                    ex=ex,
                    x_target_se=x_tar_se,
                    x_target_ex=x_tar_ex,
                    weights_st_ex=weights[:st, st:ex].copy(),
                    weights_ex_ex=weights[st:ex, st:ex].copy(),
                    weights_ex_ih=weights[st:ex, ex:ih].copy(),
                    weights_ih_ex=weights[ex:ih, st:ex].copy(),
                )
                plot_thread = threading.Thread(
                    target=_plot_background, args=(plot_kwargs,), daemon=True
                )
                plot_thread.start()
                plot_threads.append(plot_thread)
                # update num
                num += 1
        (
            mp,
            I_syn_exc,
            I_syn_inh,
            delta_mp_ex_,
            delta_mp_ih_,
            delta_I_syn_ex_,
            delta_I_syn_ih_,
        ) = update_membrane_potential(
            mp=mp_prev,
            mp_new=mp_new,
            weights_exc=weights_exc,
            weights_inh=weights_inh,
            spikes=spikes_prev,
            resting_potential=resting_potential,
            membrane_resistance_exc=membrane_resistance_exc,
            membrane_resistance_inh=membrane_resistance_inh,
            tau_m_exc=tau_m_exc,
            tau_m_inh=tau_m_inh,
            track_stats=_track_stats,
            dt=dt,
            st=st,
            ex=ex,
            N_exc=N_exc,
            N_inh=N_inh,
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

        if _track_stats:
            # track synaptic current
            delta_I_syn_ex += delta_I_syn_ex_.mean()
            delta_I_syn_ih += delta_I_syn_ih_.mean()

            I_syn_ex += I_syn_exc.mean()
            I_syn_ih += I_syn_inh.mean()

            # track mp vars
            delta_mp_ex += delta_mp_ex_.mean()
            delta_mp_ih += delta_mp_ih_.mean()
            mp_ex += mp[:N_exc].mean()
            mp_ih += mp[N_exc:].mean()

        # update spikes array
        (
            mp,
            spikes[t],
            spike_threshold,
            a,
            spike_trace,
        ) = update_spikes(
            N_exc=N_exc,
            N_inh=N_inh,
            st=st,
            ih=ih,
            mp=mp,
            dt=dt,
            a=a,
            track_stats=_track_stats,
            spikes=spikes[t],
            spike_adaption=spike_adaption,
            tau_adaption=tau_adaption,
            delta_adaption=delta_adaption,
            spike_trace=spike_trace,
            max_mp=max_mp,
            min_mp=min_mp,
            spike_threshold=spike_threshold,
            spike_threshold_default=spike_threshold_default_arr,
            reset_potential=reset_potential,
            tau_trace=tau_trace,
        )
        if _track_stats:
            # track adaptive spiking threshold
            a_ex += a.mean()
            spike_threshold_ex += spike_threshold.mean()
            spike_trace_ex += spike_trace.mean()
            track_count += 1

        # update weights
        if train_weights and t % update_weight_freq == 0:
            weights, sleep_now_inh, sleep_now_exc, m_x_pre, m_first_term, m_delta_w = (
                update_weights(
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
                    mu_weight=mu_weight,
                    delta_w=delta_w,
                    N_exc=N_exc,
                    timing_update=timing_update,
                    trace_update=trace_update,
                    spike_trace=spike_trace,
                    w_max=w_max,
                    x_tar_se=x_tar_se,
                    x_tar_ex=x_tar_ex,
                    normalize_now=normalize_now,
                    update_weights_now=update_weights_now,
                    track_weights=track_weights,
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
            )

            weights_exc = np.ascontiguousarray(weights[:, st:ex].T)
            weights_inh = np.ascontiguousarray(weights[:, ex:ih].T)

            if track_weights:
                x_pre_sum += m_x_pre
                first_term_sum += m_first_term
                # second_term_sum += m_second_term
                delta_w_sum += m_delta_w

        # Update maintained previous-step state for next iteration
        mp_prev = mp
        spikes_prev = spikes[t]

    # After the loop, before return:
    for pt in plot_threads:
        pt.join(timeout=0)  # don't block, daemon threads finish on their own

    if track_stats and train_weights:
        # compute the mean of the trackers
        mean_delta_mp_ex = delta_mp_ex / max(1, track_count)
        mean_delta_mp_ih = delta_mp_ih / max(1, track_count)
        mean_mp_ex = mp_ex / max(1, track_count)
        mean_mp_ih = mp_ih / max(1, track_count)
        mean_delta_I_syn_ex = delta_I_syn_ex / max(1, track_count)
        mean_delta_I_syn_ih = delta_I_syn_ih / max(1, track_count)
        mean_a_ex = a_ex / max(1, track_count)
        mean_x_tar_se = x_tar_sum_se / max(1, x_tar_count)
        mean_x_tar_ex = x_tar_sum_ex / max(1, x_tar_count)
        # mean_a_ih = a_ih.mean()
        mean_spike_threshold_ex = spike_threshold_ex / max(1, track_count)
        # mean_spike_threshold_ih = spike_threshold_ih.mean()
        mean_spike_trace_ex = spike_trace_ex / max(1, track_count)
        spikes_st = spikes[:, :st].mean(axis=0)
        spikes_ex = spikes[:, st:ex].mean(axis=0)
        spikes_ih = spikes[:, ex:ih].mean(axis=0)
        print(f"Mean delta mp ex: {mean_delta_mp_ex}")
        print(f"Mean delta mp ih: {mean_delta_mp_ih}")
        print(f"Mean delta I syn ex: {mean_delta_I_syn_ex}")
        print(f"Mean delta I syn ih: {mean_delta_I_syn_ih}")
        print(f"Mean membrane potential ex: {mean_mp_ex}")
        print(f"Mean membrane potential ih: {mean_mp_ih}")
        print(f"Mean I syn ex: {I_syn_ex/max(1,track_count)}")
        print(f"Mean I syn ih: {I_syn_ih/max(1,track_count)}")
        print(f"Mean a ex: {mean_a_ex}")
        print(f"Mean x_tar se: {mean_x_tar_se}")
        print(f"Mean x_tar ex: {mean_x_tar_ex}")
        # print(f"Mean a ih: {mean_a_ih}")
        print(f"Mean spike threshold ex: {mean_spike_threshold_ex}")
        # print(f"Mean spike threshold ih: {mean_spike_threshold_ih}")
        print(f"Mean spikes st: {spikes_st.mean()}")
        print(f"Mean spikes ih: {spikes_ih.mean()}")
        print(f"Mean spikes ex: {spikes_ex.mean()}")
        print(f"Mean spike trace ex: {mean_spike_trace_ex}")
        print(f"weights st->ex: ", weights[:st, st:ex][weights[:st, st:ex] != 0].mean())
        print(
            f"weights ex->ex: ",
            weights[st:ex, st:ex][weights[st:ex, st:ex] != 0].mean(),
        )
        print(
            f"weights ex->ih: ",
            weights[st:ex, ex:ih][weights[st:ex, ex:ih] != 0].mean(),
        )
        print(
            f"weights ih->ex: ",
            weights[ex:ih, st:ex][weights[ex:ih, st:ex] != 0].mean(),
        )
        # After each batch, print these:
        print("std ex->ih:", weights[st:ex, ex:ih][weights[st:ex, ex:ih] != 0].std())
        print("std ex->ex:", weights[st:ex, st:ex][weights[st:ex, st:ex] != 0].std())
        print("std st->ex:", weights[:st, st:ex][weights[:st, st:ex] != 0].std())
    if track_weights:
        mean_x_pre = x_pre_sum / max(1, x_tar_count)
        mean_first_term = first_term_sum / max(1, x_tar_count)
        # mean_second_term = second_term_sum / max(1, x_tar_count)
        mean_delta_w = delta_w_sum / max(1, x_tar_count)
        print(f"Mean x_pre: {mean_x_pre}")
        print(f"Mean first_term: {mean_first_term}")
        # print(f"Mean second_term: {mean_second_term}")
        print(f"Mean delta_w: {mean_delta_w}")
    print(np.mean(weights[st:ex, st:ex], axis=1)[32:].mean())
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
        0,
        I_syn_exc,
        I_syn_inh,
        a,
        0,
        spike_trace,
        x_tar_se,
        x_tar_ex,
    )
