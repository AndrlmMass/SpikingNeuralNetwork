# import external packages
from dataclasses import dataclass
from tqdm import tqdm
import threading
import numpy as np

# import internal packages
from src.regularization import Sleep, Normalizer
from src.neurons import NeuronState, MembranePotential, update_x_tar
from src.synapses import Learner, Clipper
from src.trackers import TrainTracker
from plot import heatmap_spike_response


def _plot_background(kwargs):
    heatmap_spike_response(**kwargs)


def report_RAM_usage():
    import psutil, os

    process = psutil.Process(os.getpid())
    print(f"Memory before training: {process.memory_info().rss / 1024**2:.0f} MB")


def thread_plotting(
    self,
    t,
    spikes,
    spike_trace,
    spike_labels,
    weights,
    x_tar_se,
    x_tar_ee,
    num,
    iterations,
):
    if not self.save_plots:
        return num, None
    plot_kwargs = dict(
        spikes_exc=spikes[t - iterations - 1 : t - 1, self.st : self.ex].copy(),
        spikes_in=spikes[t - iterations - 1 : t - 1, : self.st].copy(),
        spikes_ih=spikes[t - iterations - 1 : t - 1, self.ex :].copy(),
        label=spike_labels[t - 1],
        spike_trace=spike_trace.copy(),
        dataset=self.dataset,
        run=self.run,
        num=num,
        st=self.st,
        ex=self.ex,
        x_target_se=x_tar_se,
        x_target_ex=x_tar_ee,
        weights_st_ex=weights[: self.st, self.st : self.ex].copy(),
        weights_ex_ex=weights[self.st : self.ex, self.st : self.ex].copy(),
        weights_ex_ih=weights[self.st : self.ex, self.ex : self.ih].copy(),
        weights_ih_ex=weights[self.ex : self.ih, self.st : self.ex].copy(),
    )
    thread = threading.Thread(target=_plot_background, args=(plot_kwargs,), daemon=True)
    thread.start()
    return num + 1, thread


@dataclass
class Trainer:
    resting_potential: float
    membrane_resistance_exc: float | int
    membrane_resistance_inh: float | int
    min_weight_exc: float
    max_weight_exc: float
    min_weight_inh: float
    max_weight_inh: float
    training_mode: str
    N_inh: int
    N_exc: int
    learning_rate: float
    tau_LTP: int | float
    tau_LTD: int | float
    max_mp: float
    min_mp: float
    w_max: float
    w_target: float
    max_sum_exc: int
    max_sum_inh: int
    dt: int | float
    run: str
    A_plus: float
    A_minus: float
    tau_m_exc: int | float
    tau_m_inh: int | float
    sleep: bool
    spike_adaption: bool
    tau_adaption: int | float
    delta_adaption: int | float
    spike_threshold_default: int | float
    save_plots: bool
    reset_potential: int | float
    initial_sums_se: np.ndarray
    initial_sums_ee: np.ndarray
    dataset: str
    N_x: int
    T: int
    tau_trace: int | float
    tau_syn_exc: int | float
    tau_syn_inh: int | float
    mean_noise: int | float
    var_noise: int | float
    mu_weight: int | float
    st: int
    ex: int
    ih: int
    mp_new: np.ndarray
    time_per_item: int
    normalize_weights: bool
    nonzero_pre_idx: list
    track_stats: bool
    x_tar_se: np.ndarray  # not sure if this should be passed here
    x_tar_ee: np.ndarray  # not sure if this should be passed here
    update_weights_freq: int
    reg_frequency: int
    sleep_duration: int
    reg_mode: str
    nz_rows_exc: list  # why do we have so many of these?
    nz_cols_exc: list  # why do we have so many of these?
    nz_rows_inh: list  # why do we have so many of these?
    nz_cols_inh: list  # why do we have so many of these?
    nz_cols_se: list  # why do we have so many of these?
    nz_rows_se: list  # why do we have so many of these?
    nz_cols_ee: list  # why do we have so many of these?
    nz_rows_ee: list  # why do we have so many of these?

    def __post_init__(self):
        # initiate neuron class
        self.neuron = NeuronState(
            st=self.st,
            ih=self.ih,
            N_exc=self.N_exc,
            dt=self.dt,
            max_mp=self.max_mp,
            min_mp=self.min_mp,
            spike_adaption=self.spike_adaption,
            tau_adaption=self.tau_adaption,
            delta_adaption=self.delta_adaption,
            spike_threshold_default=self.spike_threshold_default,
            reset_potential=self.reset_potential,
            tau_trace=self.tau_trace,
        )
        # initiate membrane class
        self.membrane = MembranePotential(
            mp_new=self.mp_new,
            resting_potential=self.resting_potential,
            membrane_resistance_exc=self.membrane_resistance_exc,
            memebrane_resistance_inh=self.membrane_resistance_inh,
            dt=self.dt,
            st=self.st,
            ex=self.ex,
            track_stats=self.track_stats,
            N_exc=self.N_exc,
            N_inh=self.N_inh,
            tau_syn_exc=self.tau_syn_exc,
            tau_syn_inh=self.tau_syn_inh,
            tau_m_exc=self.tau_m_exc,
            tau_m_inh=self.tau_m_inh,
            mean_noise=self.mean_noise,
            var_noise=self.var_noise,
        )
        # initiate learner class
        self.learner = Learner(
            learning_rate=self.learning_rate,
            N_x=self.N_x,
            nonzero_pre_idx=self.nonzero_pre_idx,
            w_max=self.w_max,
            mu_weight=self.mu_weight,
        )
        # initiate clipper
        self.clipper = Clipper(
            nz_cols_exc=self.nz_cols_exc,
            nz_cols_inh=self.nz_cols_inh,
            nz_rows_exc=self.nz_rows_exc,
            nz_rows_inh=self.nz_rows_inh,
            min_weight_exc=self.min_weight_exc,
            max_weight_exc=self.max_weight_exc,
            min_weight_inh=self.min_weight_inh,
            max_weight_inh=self.max_weight_inh,
        )
        # initiate tracker
        self.tracker = TrainTracker(
            N_exc=self.N_exc,
            track_stats=self.track_stats,
            track_weights=self.track_weights,
        )
        if self.sleep and self.normalize_weights:
            raise ValueError("sleep and normalize_weights cannot both be true.")

        # define regularizer
        if self.sleep:
            self.sleep_se = Sleep(
                mode=self.reg_mode,
                duration=self.sleep_duration,
                w_target=self.w_target,
                initial_sums=self.initial_sums_se,
                nz_rows=self.nz_rows_se,
                nz_cols=self.nz_cols_se,
            )
            self.sleep_ee = Sleep(
                mode=self.reg_mode,
                duration=self.sleep_duration,
                w_target=self.w_target,
                initial_sums=self.initial_sums_ee,
                nz_rows=self.nz_rows_ee,
                nz_cols=self.nz_cols_ee,
            )
        elif self.normalize_weights:
            self.norm_se = Normalizer(
                mode=self.reg_mode,
                initial_sum=self.initial_sums_se,
                target=self.w_target,
                nz_rows=self.nz_rows_se,
                nz_cols=self.nz_cols_se,
            )
            self.norm_ee = Normalizer(
                mode=self.reg_mode,
                initial_sum=self.initial_sums_ee,
                target=self.w_target,
                nz_rows=self.nz_rows_ee,
                nz_cols=self.nz_cols_ee,
            )

    def run(
        self,
        weights,
        mp,
        spikes,
        spike_labels,
        spike_trace,
        training_mode,
        spike_threshold,
        track_weights,
        I_syn_exc,
        I_syn_inh,
        a,
        x_tar_se,
        x_tar_ee,
    ):
        # define desc and stat-collection
        if training_mode == "train":
            desc = "Training network:"
            self.tracker.reset()
        elif training_mode == "val":
            desc = "Validating network"
            self.track_weights = False
            self.track_stats = False
        else:
            desc = "Testing network:"
            self.track_weights = False
            self.track_stats = False

        # reset guards
        num = 0
        sleep_remaining = 0
        _track_stats = False
        normalize_now = False
        update_weights_now = False
        noisy_potential_now = False

        # set variables
        mp_prev = mp.copy()
        spikes_prev = spikes[0].copy()
        weights_exc = np.ascontiguousarray(weights[:, self.st : self.ex].T)
        weights_inh = np.ascontiguousarray(weights[:, self.ex : self.ih].T)

        pbar = tqdm(range(1, self.T), desc=desc, leave=False, mininterval=1.0)
        num_steps = max(1, int((self.T * 100) // self.time_per_item))
        update_weight_freq = 100  # max(1, int(T // (time_per_item)))
        iterations = 100
        plot_threads = []

        # report ram before training
        report_RAM_usage()

        if x_tar_se is None or x_tar_ee is None:
            # precompute x_tar_se and x_tar_ee
            x_tar_se, x_tar_ee = update_x_tar(
                spike_trace=spike_trace,
                N_x=self.N_x,
            )

        # create heatmap spike plot before training to verify that it starts off as wrong
        for t in pbar:
            if t % update_weight_freq == 0 and training_mode == "train":
                update_weights_now = True
                # update x_tar as often as we update weights
                x_tar_se, x_tar_ee = update_x_tar(
                    spike_trace=spike_trace,
                    N_x=self.N_x,
                )
                # update x_tar tracker
                if self.track_stats:
                    x_tar_sum_se += x_tar_se.mean()
                    x_tar_sum_ee += x_tar_ee.mean()
                    x_tar_count += 1
            if t % self.reg_frequency == 0:
                if self.normalize_weights:
                    normalize_now = True
                elif self.sleep:
                    noisy_potential_now = True
                    sleep_remaining = self.sleep_duration
                    self.sleep_se.onset(weights[: self.st, self.st : self.ex])
                    self.sleep_ee.onset(weights[self.st : self.ex, self.st : self.ex])

            if t % num_steps == 0:
                if self.track_stats:
                    _track_stats = True
                if self.save_plots:
                    num, thread = self._maybe_plot(
                        t,
                        spikes,
                        spike_trace,
                        spike_labels,
                        weights,
                        x_tar_se,
                        x_tar_ee,
                        self.num_steps,
                        self.plot_iterations,
                    )
                    plot_threads.append(thread)
                    # update num
                    num += 1

            # Activate napping babyyy
            while sleep_remaining > 0:
                # remove inputs
                spikes_buf[:] = spikes_prev
                spikes_buf[: self.st] = 0

                # update membrane potential
                (
                    mp,
                    I_syn_exc,
                    I_syn_inh,
                    delta_mp_ex_,
                    delta_mp_ih_,
                    delta_I_syn_ex_,
                    delta_I_syn_ih_,
                ) = self.membrane.step(
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
                ) = self.neuron.step(
                    mp=mp,
                    a=a,
                    spikes=self.empty_spikes,
                    spike_trace=spike_trace,
                    spike_threshold=spike_threshold,
                )

                # synapse updates
                # clip weights
                weights = self.clipper.step(weights=weights)
                # perform learning
                weights, m_x_pre, m_first_term, m_delta_w = self.learner.step(
                    spike_trace=spike_trace,
                    weights=weights,
                    spikes=spikes_buf,
                    x_tar_se=x_tar_se,
                    x_tar_ee=x_tar_ee,
                )
                # regularize weights
                weights[: self.st, self.st : self.ex] = self.sleep_se.step(
                    weights[: self.st, self.st : self.ex]
                )
                weights[self.st : self.ex, self.st : self.ex] = self.sleep_ee.step(
                    weights[self.st : self.ex, self.st : self.ex]
                )

                np.copyto(weights_exc, weights[:, self.st : self.ex].T)
                np.copyto(weights_inh, weights[:, self.ex : self.ih].T)

                sleep_remaining -= 1

                if sleep_remaining <= 0:
                    noisy_potential_now = False
            (
                mp,
                I_syn_exc,
                I_syn_inh,
                delta_mp_ex_,
                delta_mp_ih_,
                delta_I_syn_ex_,
                delta_I_syn_ih_,
            ) = self.membrane.step(
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
                mp_ex += mp[: self.N_exc].mean()
                mp_ih += mp[self.N_exc :].mean()

            # update spikes array
            (
                mp,
                spikes[t],
                spike_threshold,
                a,
                spike_trace,
            ) = self.neuron.step(
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
                weights = self.clipper.step(weights=weights)
                # perform learning
                weights, m_x_pre, m_first_term, m_delta_w = self.learner.step(
                    spike_trace=spike_trace,
                    weights=weights,
                    spikes=spikes_prev,
                    x_tar_se=x_tar_se,
                    x_tar_ee=x_tar_ee,
                )
                # regularize weights
                if self.normalize_weights and normalize_now:
                    weights[: self.st, self.st : self.ex] = self.norm_se.step(
                        weights[: self.st, self.st : self.ex]
                    )
                    weights[self.st : self.ex, self.st : self.ex] = self.norm_ee.step(
                        weights[self.st : self.ex, self.st : self.ex]
                    )
                # update x-tar

                np.copyto(weights_exc, weights[:, self.st : self.ex].T)
                np.copyto(weights_inh, weights[:, self.ex : self.ih].T)

                update_weights_now = False
                normalize_now = False

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

        return (
            weights,
            spikes,
            mp,
            spike_threshold,
            spike_labels,
            I_syn_exc,
            I_syn_inh,
            a,
            spike_trace,
            x_tar_se,
            x_tar_ee,
        )
