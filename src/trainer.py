# import external packages
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np

# import internal packages
from regularization import Sleep, Normalizer
from neurons import NeuronState, MembranePotential, update_x_tar
from performance import report_RAM_usage, spawn_plot_thread
from synapses import Learner, Clipper
from trackers import TrainTracker


@dataclass
class Trainer:
    resting_potential: float
    membrane_resistance_exc: float | int
    membrane_resistance_inh: float | int
    min_weight_exc: float
    max_weight_exc: float
    min_weight_inh: float
    max_weight_inh: float
    N_inh: int
    N_exc: int
    learning_rate: float
    tau_LTP: int | float
    tau_LTD: int | float
    max_mp: float
    min_mp: float
    w_max: float
    w_target: float
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
    clip_weights: bool
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
    track_weights: bool
    track_stats: bool
    stat_tracking_frequency: int
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
            membrane_resistance_inh=self.membrane_resistance_inh,
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
            st=self.st,
            ex=self.ex,
            ih=self.ih,
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

    def step(
        self,
        weights,
        mp,
        spikes,
        spike_labels,
        spike_trace,
        training_mode,
        spike_threshold,
        I_syn_exc,
        I_syn_inh,
        a,
        x_tar_se,
        x_tar_ee,
        num=0,
        sleep_remaining=0,
        _track_stats=False,
        normalize_now=False,
        update_weights_now=False,
        noisy_potential_now=False,
    ):
        # define desc and stat-collection
        if training_mode == "train":
            desc = "Training network:"
            track_stats = self.track_stats
            track_weights = self.track_weights
            self.tracker.reset()
        elif training_mode == "val":
            desc = "Validating network"
            track_weights = False
            track_stats = False
        elif training_mode == "test":
            desc = "Testing network:"
            track_weights = False
            track_stats = False
        else:
            raise ValueError("training_mode must be 'train', 'val', or 'test'.")

        # set variables
        plot_threads = []
        mp_prev = mp.copy()
        spikes_prev = spikes[0].copy()
        weights_exc = np.ascontiguousarray(weights[:, self.st : self.ex].T)
        weights_inh = np.ascontiguousarray(weights[:, self.ex : self.ih].T)

        # prepare progress bar
        pbar = tqdm(range(1, spikes.shape[0]), desc=desc, leave=False, mininterval=1.0)

        # report ram before training
        report_RAM_usage()

        # precompute x_tar_se and x_tar_ee
        x_tar_se, x_tar_ee = update_x_tar(
            spike_trace=spike_trace,
            N_x=self.N_x,
        )

        # run training for T iterations
        for t in pbar:
            if training_mode == "train":
                if t % self.update_weights_freq == 0:
                    update_weights_now = True
                if t % self.reg_frequency == 0:
                    if self.normalize_weights:
                        normalize_now = True
                    elif self.sleep:
                        noisy_potential_now = True
                        sleep_remaining = self.sleep_duration
                        self.sleep_se.onset(weights[: self.st, self.st : self.ex])
                        self.sleep_ee.onset(
                            weights[self.st : self.ex, self.st : self.ex]
                        )

            if t % self.stat_tracking_frequency == 0:
                if track_stats:
                    _track_stats = True
                if self.save_plots:
                    num, thread = spawn_plot_thread(
                        t,
                        spikes,
                        spike_trace,
                        spike_labels,
                        weights,
                        x_tar_se,
                        x_tar_ee,
                        self.num,
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
                if self.clip_weights:
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
                # update x_tar
                x_tar_se, x_tar_ee = update_x_tar(
                    spike_trace=spike_trace,
                    N_x=self.N_x,
                )

                np.copyto(weights_exc, weights[:, self.st : self.ex].T)
                np.copyto(weights_inh, weights[:, self.ex : self.ih].T)

                sleep_remaining -= 1

                if sleep_remaining <= 0:
                    noisy_potential_now = False
            # WAKE UP!
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

            # synapse updates
            if update_weights_now:
                if self.clip_weights:
                    # clip weights
                    weights = self.clipper.step(weights=weights)
                # perform learning
                weights, m_x_pre, m_first_term, m_delta_w = self.learner.step(
                    spike_trace=spike_trace,
                    weights=weights,
                    spikes=spikes_prev,
                    x_tar_se=x_tar_se,
                    x_tar_ee=x_tar_ee,
                    track_weights=track_weights,
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
                x_tar_se, x_tar_ee = update_x_tar(
                    spike_trace=spike_trace,
                    N_x=self.N_x,
                )
                # run synapse tracker
                if track_weights:
                    self.tracker.track_synapse(
                        m_x_pre, m_first_term, m_delta_w, x_tar_se, x_tar_ee
                    )

                np.copyto(weights_exc, weights[:, self.st : self.ex].T)
                np.copyto(weights_inh, weights[:, self.ex : self.ih].T)

                update_weights_now = False
                normalize_now = False

            if _track_stats:
                self.tracker.track_neuron(
                    mp=mp,
                    delta_mp_ex=delta_mp_ex_,
                    delta_mp_ih=delta_mp_ih_,
                    I_syn_exc=I_syn_exc,
                    I_syn_inh=I_syn_inh,
                    delta_I_syn_ex_=delta_I_syn_ex_,
                    delta_I_syn_ih=delta_I_syn_ih_,
                    a=a,
                    spike_threshold=spike_threshold,
                    spike_trace=spike_trace,
                )

            # Update maintained previous-step state for next iteration
            mp_prev = mp
            spikes_prev = spikes[t]

        # After the loop, before return:
        for pt in plot_threads:
            pt.join(timeout=0)  # don't block, daemon threads finish on their own

        # print final stats - might add if-statement clause for this. Bit annoying for long runs.
        self.tracker.print(
            weights=weights,
            spikes=spikes,
            track_weights=track_weights,
            track_stats=track_stats,
        )

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
