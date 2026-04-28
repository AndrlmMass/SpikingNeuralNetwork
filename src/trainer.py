# import external packages
from numba.typed import List
from numba import njit
from tqdm import tqdm
import threading
import numpy as np

# import internal packages
from src.regularization import Sleep, Normalizer
from src.neurons import NeuronState, MembranePotential, update_x_tar
from src.synapses import Learner, Clipper
from plot import heatmap_spike_response


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
    training_mode,
    weight_decay_rate,
    sleep_synchronized,
    baseline_sum,
    max_sum,
    N_inh,
    N_exc,
    learning_rate,
    tau_LTP,
    tau_LTD,
    max_mp,
    min_mp,
    w_max,
    check_sleep_interval,
    w_target,
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
    noisy_weights,
    vectorized_trace,
    spike_labels,
    weight_mean_noise,
    weight_var_noise,
    dataset,
    timing_update,
    trace_update,
    N_x,
    T,
    tau_trace,
    tau_syn_exc,
    tau_syn_inh,
    track_weights,
    mean_noise,
    var_noise,
    mu_weight,
    I_syn_exc,
    I_syn_inh,
    a,
    st,
    ex,
    ih,
    time_per_item,
    spike_threshold,
    normalize_weights=False,
    normalize_per_column=False,
    normalize_per_column_interval=1000,
    initial_sum_st_ex=None,
    initial_sum_ex_ex=None,
    initial_sum_ex_ih=None,
    initial_sum_ih_ex=None,
    track_stats=False,
    x_tar_se=None,
    x_tar_ex=None,
    reg_frequency=None,
    sleep_duration: int = 5000,
    reg_mode: str = "static",  # one of {"static","group","post"}
):

    # Enforce consistent dtypes for Numba cache stability
    """
    Are these necessary?
    """
    # weights = np.asarray(weights, dtype=np.float64)
    # spike_trace = np.asarray(spike_trace, dtype=np.float64)
    # mp = np.asarray(mp, dtype=np.float64)
    # I_syn_exc = np.asarray(I_syn_exc, dtype=np.float64)
    # I_syn_inh = np.asarray(I_syn_inh, dtype=np.float64)
    # a = np.asarray(a, dtype=np.float64)
    # spike_threshold = np.asarray(spike_threshold, dtype=np.float64)
    # N_x = np.int64(N_x)
    # N_exc = np.int64(N_exc)
    # N_inh = np.int64(N_inh)
    # spike_threshold_default_arr = np.full(
    #     N_exc + N_inh, fill_value=spike_threshold_default, dtype=np.float64
    # )

    # Update membrane potentials using previous step spikes
    mp_new = np.zeros_like(mp)
    weights_exc = np.ascontiguousarray(weights[:, st:ex].T)
    weights_inh = np.ascontiguousarray(weights[:, ex:ih].T)

    nz_rows_inh, nz_cols_inh = np.nonzero(weights[ex:ih, st:ex])
    nz_rows_exc, nz_cols_exc = np.nonzero(weights[:ex, :ih])
    nz_rows_inh += ex
    nz_cols_inh += st
    sleep_now = False

    # Compute for neurons N_x to N_post-1
    nonzero_pre_idx = List()
    for i in range(st, ih):
        pre_idx = np.nonzero(weights[:, i])[0]
        nonzero_pre_idx.append(pre_idx.astype(np.int64))

    # initiate neuron class
    neuron = NeuronState(
        st=st,
        ih=ih,
        N_exc=N_exc,
        dt=dt,
        max_mp=max_mp,
        min_mp=min_mp,
        spike_adaption=spike_adaption,
        tau_adaption=tau_adaption,
        delta_adaption=delta_adaption,
        spike_threshold_default=spike_threshold_default,
        reset_potential=reset_potential,
        tau_trace=tau_trace,
    )
    # initiate membrane class
    membrane = MembranePotential(
        mp_new=mp_new,
        resting_potential=resting_potential,
        membrane_resistance_exc=membrane_resistance_exc,
        memebrane_resistance_inh=membrane_resistance_inh,
        dt=dt,
        st=st,
        ex=ex,
        track_stats=track_stats,
        N_exc=N_exc,
        N_inh=N_inh,
        tau_syn_exc=tau_syn_exc,
        tau_syn_inh=tau_syn_inh,
        tau_m_exc=tau_m_exc,
        tau_m_inh=tau_m_inh,
        mean_noise=mean_noise,
        var_noise=var_noise,
    )
    # initiate learner class
    learner = Learner(
        learning_rate=learning_rate,
        N_x=N_x,
        nonzero_pre_idx=nonzero_pre_idx,
        track_weights=track_weights,
        w_max=w_max,
        mu_weight=mu_weight,
    )
    # initiate clipper
    clipper = Clipper(
        nz_cols_exc=nz_cols_exc,
        nz_cols_inh=nz_cols_inh,
        nz_rows_exc=nz_rows_exc,
        nz_rows_inh=nz_rows_inh,
        min_weight_exc=min_weight_exc,
        max_weight_exc=max_weight_exc,
        min_weight_inh=min_weight_inh,
        max_weight_inh=max_weight_inh,
    )

    # define sleep regularizer
    if sleep or normalize_weights:
        nz_rows_se, nz_cols_se = np.nonzero(weights[:st, st:ex])
        nz_rows_ee, nz_cols_ee = np.nonzero(weights[st:ex, st:ex])

        if reg_mode == "post":
            initial_sums_se = weights[:st, st:ex].sum(axis=0)
            initial_sums_ee = weights[st:ex, st:ex].sum(axis=0)
        elif reg_mode == "layer":
            initial_sums_se = np.full(nz_rows_se.size, weights[:st, st:ex].sum())
            initial_sums_ee = np.full(nz_rows_ee.size, weights[st:ex, st:ex].sum())
        else:
            initial_sums_se = 0
            initial_sums_ee = 0

        if sleep:
            sleep_reg_se = Sleep(
                mode=reg_mode,
                duration=sleep_duration,
                w_target=w_target,
                initial_sums=initial_sums_se,
                nz_rows=nz_rows_se,
                nz_cols=nz_cols_se,
            )
            sleep_reg_ee = Sleep(
                mode=reg_mode,
                duration=sleep_duration,
                w_target=w_target,
                initial_sums=initial_sums_ee,
                nz_rows=nz_rows_ee,
                nz_cols=nz_cols_ee,
            )
        elif normalize_weights:
            norm_reg_se = Normalizer(
                mode=reg_mode,
                initial_sum=initial_sums_se,
                target=w_target,
                nz_rows=nz_rows_se,
                nz_cols=nz_cols_se,
            )
            norm_reg_ee = Normalizer(
                mode=reg_mode,
                initial_sum=initial_sums_ee,
                target=w_target,
                nz_rows=nz_rows_ee,
                nz_cols=nz_cols_ee,
            )

    if track_stats:  # Add sleep tracking parameters here when this is ready
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

    if training_mode == "train":
        desc = "Training network:"
    elif training_mode == "val":
        desc = "Validating network"
    else:
        desc = "Testing network:"

    nonzero_pre_idx_exc = List()
    for i in range(st, ex):
        pre_idx = np.nonzero(weights[:ex, i])[0]
        nonzero_pre_idx_exc.append(pre_idx.astype(np.int64))

    # Get selected indices
    # Snapshot cadence
    plot_time = 0  # increases by 1 per recorded snapshot (training or sleep)

    pbar = tqdm(range(1, T), desc=desc, leave=False, mininterval=1.0)
    # Initial snapshot
    plot_time += 1
    num_steps = max(1, int((T * 100) // time_per_item))
    update_weight_freq = 100  # max(1, int(T // (time_per_item)))
    iterations = 100
    plot_threads = []
    _track_stats = False
    noisy_potential_now = False
    num = 0
    normalize_now = False
    sleep_now = False
    empty_spikes = np.zeros(spikes.shape[0])

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
        if t % update_weight_freq == 0 and training_mode == "train":
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
        if t % reg_frequency:
            if normalize_weights:
                normalize_now = True
            elif sleep:
                sleep_now = True
                noisy_potential_now = True
                sleep_remaining = sleep_duration
                sleep_reg_se.onset(weights[:st, st:ex])
                sleep_reg_ee.onset(weights[st:ex, st:ex])

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
        # Activate napping babyyy
        while sleep_remaining > 0:
            # remove input data
            spikes_buf = spikes_prev.copy()
            spikes_buf[:st] = 0

            (
                mp,
                I_syn_exc,
                I_syn_inh,
                delta_mp_ex_,
                delta_mp_ih_,
                delta_I_syn_ex_,
                delta_I_syn_ih_,
            ) = membrane.step(
                mp=mp_prev,
                weights_exc=weights_exc,
                weights_inh=weights_inh,
                spikes=spikes_buf,
                I_syn_exc=I_syn_exc,
                I_syn_inh=I_syn_inh,
                noisy_potential_now=noisy_potential_now,
            )

            # update spikes array
            (
                mp,
                spikes_buf,
                spike_threshold,
                a,
                spike_trace,
            ) = neuron.step(
                mp=mp,
                a=a,
                spikes=empty_spikes,
                spike_trace=spike_trace,
                spike_threshold=spike_threshold,
            )

            # synapse updates
            # clip weights
            weights = clipper.step(weights=weights)
            # perform learning
            weights, m_x_pre, m_first_term, m_delta_w = learner.step(
                spike_trace=spike_trace,
                weights=weights,
                spikes=spikes_buf,
                x_tar_se=x_tar_se,
                x_tar_ex=x_tar_ex,
            )
            # regularize weights
            weights[:st, st:ex] = sleep_reg_se.step(weights[:st, st:ex])
            weights[st:ex, st:ex] = sleep_reg_ee.step(weights[st:ex, st:ex])

            np.copyto(weights_exc, weights[:, st:ex].T)
            np.copyto(weights_inh, weights[:, ex:ih].T)

            sleep_remaining -= 1

        (
            mp,
            I_syn_exc,
            I_syn_inh,
            delta_mp_ex_,
            delta_mp_ih_,
            delta_I_syn_ex_,
            delta_I_syn_ih_,
        ) = membrane.step(
            mp=mp_prev,
            weights_exc=weights_exc,
            weights_inh=weights_inh,
            spikes=spikes_prev,
            I_syn_exc=I_syn_exc,
            I_syn_inh=I_syn_inh,
            noisy_potential_now=noisy_potential_now,
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
        ) = neuron.step(
            mp=mp,
            a=a,
            spikes=spikes[t],
            spike_trace=spike_trace,
            spike_threshold=spike_threshold,
        )

        if _track_stats:
            # track adaptive spiking threshold
            a_ex += a.mean()
            spike_threshold_ex += spike_threshold.mean()
            spike_trace_ex += spike_trace.mean()
            track_count += 1

        # synapse updates
        if update_weights_now:
            # clip weights
            weights = clipper.step(weights=weights)
            # perform learning
            weights, m_x_pre, m_first_term, m_delta_w = learner.step(
                spike_trace=spike_trace,
                weights=weights,
                spikes=spikes,
                x_tar_se=x_tar_se,
                x_tar_ex=x_tar_ex,
            )
            # regularize weights
            if normalize_weights and normalize_now:
                weights[:st, st:ex] = norm_reg_se.step(weights[:st, st:ex])
                weights[st:ex, st:ex] = norm_reg_ee.step(weights[st:ex, st:ex])

            np.copyto(weights_exc, weights[:, st:ex].T)
            np.copyto(weights_inh, weights[:, ex:ih].T)

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

    if track_stats and training_mode == "train":
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


@njit(parallel=True, cache=True)
def run_sleep(
    weights,
    mp,
    I_syn_exc,
    I_syn_inh,
    spike_trace,
    a,
    spike_threshold,
    sleep_duration,
    scale_se,
    scale_ee,  # precomputed by onset()
    nz_rows_se,
    nz_cols_se,
    nz_rows_ee,
    nz_cols_ee,
    # all static neuron/membrane params...
):
    spikes_buf = np.zeros(mp.shape[0])
    for _ in range(sleep_duration):
        mp, I_syn_exc, I_syn_inh = _update_membrane(mp, spikes_buf, ...)  # njit helper
        mp, spikes_buf, spike_threshold, a, spike_trace = _update_spikes(
            mp, ...
        )  # njit helper
        weights = _apply_scale(weights, scale_se, scale_ee, ...)  # njit helper
        spikes_buf[:] = 0.0
    return weights, mp, I_syn_exc, I_syn_inh, spike_trace, a, spike_threshold
