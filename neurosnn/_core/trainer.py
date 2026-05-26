from dataclasses import dataclass
from tqdm import tqdm
import numpy as np

from neurosnn._core.regularization import Sleep, Normalizer
from neurosnn._core.neurons import NeuronState, MembranePotential, update_x_tar
from neurosnn._utils.performance import report_RAM_usage, spawn_plot_thread
from neurosnn._core.synapses import Learner, Clipper
from neurosnn._core.trackers import TrainTracker


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
    w_target_se: float
    w_target_ee: float
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
    plot_iterations: int
    st: int
    ex: int
    ih: int
    mp_new: np.ndarray
    time_per_item: int
    normalize_weights: bool
    nonzero_pre_idx: list
    track_stats: bool
    update_weights_freq: int
    reg_frequency: int
    sleep_duration: int
    track_weights: bool
    stat_tracking_frequency: int
    reg_mode: str
    train_weights: bool
    nz_cols_se: list
    nz_rows_se: list
    nz_cols_ee: list
    nz_rows_ee: list
    nz_cols_exc: list
    nz_rows_exc: list
    record_fn_se: "callable | None" = None
    record_fn_ee: "callable | None" = None
    record_fn_awake_se: "callable | None" = None
    record_fn_awake_ee: "callable | None" = None

    def __post_init__(self):
        self.track_stats = np.uint8(self.track_stats)
        self.track_weights = np.uint8(self.track_weights)
        self.spike_adaption = np.uint8(self.spike_adaption)
        self.clip_weights = np.uint8(self.clip_weights)
        self.sleep = np.uint8(self.sleep)
        self.train_weights = np.uint8(self.train_weights)
        self.normalize_weights = np.uint8(self.normalize_weights)

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
        self.learner = Learner(
            learning_rate=self.learning_rate,
            N_x=self.N_x,
            nonzero_pre_idx=self.nonzero_pre_idx,
            w_max=self.w_max,
            mu_weight=self.mu_weight,
        )
        self.clipper = Clipper(
            nz_cols=self.nz_cols_exc,
            nz_rows=self.nz_rows_exc,
            min_weight_exc=self.min_weight_exc,
            max_weight_exc=self.max_weight_exc,
        )
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

        if self.sleep:
            self.sleep_se = Sleep(
                mode=self.reg_mode,
                duration=self.sleep_duration,
                w_target=self.w_target_se,
                initial_sums=self.initial_sums_se,
                nz_rows=self.nz_rows_se,
                nz_cols=self.nz_cols_se,
                record_fn=self.record_fn_se,
            )
            self.sleep_ee = Sleep(
                mode=self.reg_mode,
                duration=self.sleep_duration,
                w_target=self.w_target_ee,
                initial_sums=self.initial_sums_ee,
                nz_rows=self.nz_rows_ee,
                nz_cols=self.nz_cols_ee,
                record_fn=self.record_fn_ee,
            )
        elif self.normalize_weights:
            self.norm_se = Normalizer(
                mode=self.reg_mode,
                initial_sum=self.initial_sums_se,
                target=self.w_target_se,
                nz_rows=self.nz_rows_se,
                nz_cols=self.nz_cols_se,
                weight_cols=self.N_exc,
                record_fn=self.record_fn_se,
            )
            self.norm_ee = Normalizer(
                mode=self.reg_mode,
                initial_sum=self.initial_sums_ee,
                target=self.w_target_ee,
                nz_rows=self.nz_rows_ee,
                nz_cols=self.nz_cols_ee,
                weight_cols=self.N_exc,
                record_fn=self.record_fn_ee,
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
        num=0,
        sleep_remaining=0,
        _track_stats=np.uint8(0),
        normalize_now=np.uint8(0),
        update_weights_now=np.uint8(0),
        noisy_potential_now=np.uint8(0),
    ):
        if training_mode == "train":
            desc = "Training network:"
            track_stats = self.track_stats
            track_weights = self.track_weights
            save_plots = self.save_plots
            train_weights = self.train_weights
            self.tracker.reset()
        elif training_mode == "val":
            desc = "Validating network"
            track_weights = np.uint8(0)
            track_stats = np.uint8(0)
            save_plots = False
            train_weights = np.uint8(0)
        elif training_mode == "test":
            desc = "Testing network:"
            track_weights = np.uint8(0)
            track_stats = np.uint8(0)
            train_weights = np.uint8(0)
            save_plots = False
        else:
            raise ValueError("training_mode must be 'train', 'val', or 'test'.")

        mp_prev = mp.copy()
        spikes_prev = spikes[0].copy()
        weights_exc = np.ascontiguousarray(weights[:, self.st : self.ex].T)
        weights_inh = np.ascontiguousarray(
            weights[self.st : self.ex, self.ex : self.ih].T
        )

        pbar = tqdm(range(1, spikes.shape[0]), desc=desc, leave=False, mininterval=1.0)

        report_RAM_usage()

        x_tar_se, x_tar_ee = update_x_tar(
            spike_trace=spike_trace,
            N_x=self.N_x,
        )
        for t in pbar:
            if training_mode == "train":
                if t % self.update_weights_freq == 0:
                    update_weights_now = np.uint8(1)
                if t % self.reg_frequency == 0:
                    if self.normalize_weights:
                        normalize_now = np.uint8(1)
                    elif self.sleep:
                        noisy_potential_now = np.uint8(1)
                        sleep_remaining = self.sleep_duration
                        self.sleep_se.onset(weights[: self.st, self.st : self.ex])
                        self.sleep_ee.onset(
                            weights[self.st : self.ex, self.st : self.ex]
                        )

            if t % self.stat_tracking_frequency == 0:
                if track_stats:
                    _track_stats = np.uint8(1)
                if save_plots:
                    num, _ = spawn_plot_thread(
                        t,
                        spikes,
                        spike_trace,
                        spike_labels,
                        weights,
                        x_tar_se,
                        x_tar_ee,
                        num,
                        self.plot_iterations,
                        self.st,
                        self.ex,
                        self.ih,
                        self.dataset,
                        self.run,
                        self.save_plots,
                    )

            if sleep_remaining > 0:
                spikes_buf = spikes_prev.copy()
                spikes_buf[: self.st] = 0
                current_spikes = np.zeros(self.ih, dtype=np.int8)

            while sleep_remaining > 0 and train_weights:
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

                current_spikes[:] = 0
                (
                    mp,
                    spikes_buf,
                    spike_threshold,
                    a,
                    spike_trace,
                ) = self.neuron.step(
                    mp=mp,
                    a=a,
                    spikes=current_spikes,
                    spike_trace=spike_trace,
                    spike_threshold=spike_threshold,
                )

                if sleep_remaining % 10 == 0:
                    if self.clip_weights:
                        weights = self.clipper.step(weights=weights)
                    weights, m_x_pre, m_first_term, m_delta_w = self.learner.step(
                        spike_trace=spike_trace,
                        weights=weights,
                        spikes=spikes_buf,
                        track_weights=np.uint8(0),
                        x_tar_se=x_tar_se,
                        x_tar_ee=x_tar_ee,
                    )
                    np.copyto(weights_exc, weights[:, self.st : self.ex].T)
                    np.copyto(
                        weights_inh, weights[self.st : self.ex, self.ex : self.ih].T
                    )

                weights[: self.st, self.st : self.ex] = self.sleep_se.step(
                    weights[: self.st, self.st : self.ex], t
                )
                weights[self.st : self.ex, self.st : self.ex] = self.sleep_ee.step(
                    weights[self.st : self.ex, self.st : self.ex], t
                )

                sleep_remaining -= 1
                mp_prev = mp

                if sleep_remaining <= 0:
                    noisy_potential_now = np.uint8(0)

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

            if update_weights_now and train_weights:
                if self.clip_weights:
                    weights = self.clipper.step(weights=weights)
                weights, m_x_pre, m_first_term, m_delta_w = self.learner.step(
                    spike_trace=spike_trace,
                    weights=weights,
                    spikes=spikes_prev,
                    x_tar_se=x_tar_se,
                    x_tar_ee=x_tar_ee,
                    track_weights=track_weights,
                )
                if self.normalize_weights and normalize_now:
                    weights[: self.st, self.st : self.ex] = self.norm_se.step(
                        weights[: self.st, self.st : self.ex], t
                    )
                    weights[self.st : self.ex, self.st : self.ex] = self.norm_ee.step(
                        weights[self.st : self.ex, self.st : self.ex], t
                    )
                x_tar_se, x_tar_ee = update_x_tar(
                    spike_trace=spike_trace,
                    N_x=self.N_x,
                )
                if track_weights:
                    self.tracker.track_synapse(
                        m_x_pre,
                        m_first_term,
                        m_delta_w,
                        x_tar_se,
                        x_tar_ee,
                    )

                np.copyto(weights_exc, weights[:, self.st : self.ex].T)
                np.copyto(weights_inh, weights[self.st : self.ex, self.ex : self.ih].T)

                if self.record_fn_awake_se is not None:
                    self.record_fn_awake_se(weights[: self.st, self.st : self.ex], t)
                if self.record_fn_awake_ee is not None:
                    self.record_fn_awake_ee(
                        weights[self.st : self.ex, self.st : self.ex], t
                    )

                update_weights_now = np.uint8(0)
                normalize_now = np.uint8(0)

            if _track_stats:
                self.tracker.track_neuron(
                    mp=mp,
                    delta_mp_ex_=delta_mp_ex_,
                    delta_mp_ih_=delta_mp_ih_,
                    I_syn_exc=I_syn_exc,
                    I_syn_inh=I_syn_inh,
                    delta_I_syn_ex_=delta_I_syn_ex_,
                    delta_I_syn_ih_=delta_I_syn_ih_,
                    a=a,
                    spike_threshold=spike_threshold,
                    spike_trace=spike_trace,
                )
                _track_stats = np.uint8(0)

            mp_prev = mp
            spikes_prev = spikes[t]

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
            self.tracker.to_dict(),
        )
