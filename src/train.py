from numba.typed import List
from numba import njit
from tqdm import tqdm
import numpy as np
from weight_funcs import sleep_func, spike_timing, vectorized_trace_func


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
        print("Numba status — " + ", ".join(msgs))
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
    nz_rows,
    nz_cols,
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
            spike_times=spike_times,
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

    if vectorized_trace:
        weights = vectorized_trace_func(
            check_sleep_interval=check_sleep_interval,
            spikes=spikes,
            N_x=N_x,
            nz_rows=nz_rows,
            nz_cols=nz_cols,
            delta_w=delta_w,
            A_plus=A_plus,
            dt=dt,
            A_minus=A_minus,
            N_inh=N_inh,
        )

    # add noise to weights if desired
    if noisy_weights:
        delta_weight_noise = np.random.normal(
            loc=weight_mean_noise, scale=weight_var_noise, size=weights.shape
        )
        weights += delta_weight_noise

        # Update weights
        # print(
        #     f"\rexc weight change:{np.round(np.mean(delta_weights_exc),6)}, inh weight change:{np.round(np.mean(delta_weights_inh),6)}",
        #     end="",
        # )

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
    sleep_exc,
    sleep_inh,
):
    if noisy_potential and (sleep_exc or sleep_inh):
        gaussian_noise = np.random.normal(
            loc=mean_noise, scale=var_noise, size=mp.shape
        )
    else:
        gaussian_noise = 0

    mp_new = mp.copy()
    # Update synaptic current (avoid in-place modification)
    I_syn_new = I_syn + (-I_syn + np.dot(weights.T, spikes)) * dt / tau_syn
    mp_delta = (
        (-(mp - resting_potential) + membrane_resistance * I_syn_new) / tau_m * dt
    )
    mp_new += mp_delta + gaussian_noise
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

    # update spikes array
    mp = np.clip(mp, a_min=min_mp, a_max=max_mp)
    spikes[st:ih][mp > spike_threshold] = 1

    # Add Solve's noisy membrane potential
    if noisy_threshold:
        delta_potential = spike_threshold - mp
        p_fire = np.exp(spike_slope * delta_potential + spike_intercept) / (
            1 + np.exp(spike_slope * delta_potential + spike_intercept)
        )
        additional_spikes = np.random.binomial(n=1, p=p_fire)
        spikes[st:ih] = spikes[st:ih] | additional_spikes

    # add spike adaption
    if spike_adaption:
        # --- each timestep ---
        a += (-a / tau_adaption) * dt
        a[spikes[st:ih] == 1] += delta_adaption

        spike_threshold = spike_threshold_default + a
        spike_threshold = np.clip(spike_threshold, -90, 0)

    mp[spikes[st:ih] == 1] = reset_potential
    spike_times = np.where(spikes == 1, 0, spike_times + 1)

    return mp, spikes, spike_times, spike_threshold, a


def train_network(
    weights,
    mp,
    spikes,
    noisy_potential,
    resting_potential,
    membrane_resistance,
    spike_times,
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
    vectorized_trace,
    spike_labels,
    weight_mean_noise,
    weight_var_noise,
    timing_update,
    trace_update,
    interval,
    N_x,
    T,
    tau_syn,
    beta,
    mean_noise,
    var_noise,
    num_exc,
    num_inh,
    I_syn,
    a,
    spike_threshold,
    sleep_ratio=0.0,
    normalize_weights=False,
    initial_sum_exc=None,
    initial_sum_inh=None,
    initial_sum_total=None,
    # Hard-pause sleep settings
    sleep_hard_pause: bool = True,
    sleep_epsilon: float = 1e-8,
    sleep_tol_frac: float = 1e-3,
    sleep_max_iters: int = 5000,
    on_timeout: str = "scale_to_target",  # one of {"scale_to_target","extend","give_up"}
):

    st = N_x  # stimulation
    ex = st + N_exc  # excitatory
    ih = ex + N_inh  # inhibitory
    exc_interval = np.arange(st, ex)
    inh_interval = np.arange(ex, ih)
    idx_exc = np.random.choice(exc_interval, size=num_exc, replace=False)
    idx_inh = np.random.choice(inh_interval, size=num_inh, replace=False)

    # Disable training-time plotting snapshots for performance
    plot_positions = np.array([1])
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

    # Precompute scheduled sleep window per interval
    # e.g., sleep_ratio=0.1 means 10% of each interval is sleep
    if sleep and sleep_ratio is not None and sleep_ratio > 0.0:
        sleep_window = max(1, int(round(check_sleep_interval * sleep_ratio)))
    else:
        sleep_window = 0

    # Print sleep configuration
    if sleep and sleep_window > 0:
        expected_sleep_pct = (sleep_window / check_sleep_interval) * 100
        print(
            f"Sleep scheduled: {sleep_window}/{check_sleep_interval} timesteps per interval ({expected_sleep_pct:.1f}%)"
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

    idx = 0
    sleep_amount = 0
    virtual_sleep_iters_epoch = 0

    # Maintain previous-step state explicitly so we can evolve during hard-pause sleep
    mp_prev = mp[0].copy()
    spikes_prev = spikes[0].copy()
    t_virtual = 0  # virtual time used only inside sleep loop

    # Disable weight tracking during training for performance; keep empty shell for API compatibility
    weight_tracking_sleep = {
        "exc_mean": [],
        "exc_std": [],
        "exc_min": [],
        "exc_max": [],
        "exc_samples": [],
        "inh_mean": [],
        "inh_std": [],
        "inh_min": [],
        "inh_max": [],
        "inh_samples": [],
        "times": [],
    }

    pbar = tqdm(range(1, T), desc=desc, leave=False)
    last_sleep_flag = -1  # unknown
    last_stats_update_t = -1000
    for t in pbar:
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
                    sleep_exc=sleep_now_exc,
                    sleep_inh=sleep_now_inh,
                )

                # Prepare current spikes vector (no sensory spikes during sleep)
                sleep_spikes_cur = np.zeros_like(spikes_prev)

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
                    weights, _, _ = update_weights(
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
                        sleep_now_inh=True,
                        sleep_now_exc=True,
                        t=t_virtual,
                        N=N,
                        sleep_synchronized=sleep_synchronized,
                        baseline_sum=baseline_sum,
                        nz_rows=nz_rows,
                        nz_cols=nz_cols,
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
                    )

                    # Apply exponential decay toward targets during sleep (combine with STDP)
                    weights, sleep_now_inh, sleep_now_exc = sleep_func(
                        weights=weights,
                        max_sum=max_sum,
                        max_sum_exc=max_sum_exc,
                        max_sum_inh=max_sum_inh,
                        sleep_now_inh=True,
                        sleep_now_exc=True,
                        w_target_exc=w_target_exc,
                        w_target_inh=w_target_inh,
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

                # Optional normalization at every step if enabled
                if normalize_weights and initial_sum_exc is not None:
                    cur_exc = np.sum(np.abs(weights[:ex, st:ih]))
                    if cur_exc > 1e-10:
                        weights[:ex, st:ih] *= initial_sum_exc / cur_exc
                    cur_inh = np.sum(np.abs(weights[ex:ih, st:ex]))
                    if cur_inh > 1e-10:
                        weights[ex:ih, st:ex] *= initial_sum_inh / cur_inh

                # Tracking disabled

                # Advance internal counters and previous-step states
                spikes_prev = sleep_spikes_cur
                sleep_iter += 1
                sleep_time_counter += 1
                virtual_sleep_iters_epoch += 1
                t_virtual += 1

            # End hard-pause sleep; do not advance real t here (the loop continues below)
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
            sleep_exc=sleep_now_exc,
            sleep_inh=sleep_now_inh,
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
                nz_rows=nz_rows,
                nz_cols=nz_cols,
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
            )

            # Removed per-step weight normalization due to performance impact

            # After first successful compilation, report numba status once
            if t == 1:
                report_numba_status()

            # snapshots disabled

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
        max_sum_inh,
        max_sum_exc,
        spike_labels,
        sleep_percent,
        I_syn,
        spike_times,
        a,
        weight_tracking_sleep,
    )
