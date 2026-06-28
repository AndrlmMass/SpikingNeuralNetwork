from dataclasses import dataclass
from tqdm import tqdm
import numpy as np

from neurosnn._core.regularization import Sleep, Normalizer
from neurosnn._core.neurons import NeuronState, MembranePotential, update_x_tar, update_slow_traces
from neurosnn._utils.performance import report_RAM_usage, spawn_plot_thread
from neurosnn._core.synapses import Learner, TripletLearner, Clipper
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
    x_tar_mode: str
    x_tar_pct_se: float
    x_tar_pct_ee: float
    nz_cols_se: list
    nz_rows_se: list
    nz_cols_ee: list
    nz_rows_ee: list
    nz_cols_exc: list
    nz_rows_exc: list
    use_triplet: bool = False
    tau_x: float = 101.0
    tau_y: float = 125.0
    A2_plus: float = 5e-10
    A3_plus: float = 6.2e-3
    A2_minus: float = 7e-3
    A3_minus: float = 2.3e-4
    record_fn_se: "callable | None" = None
    record_fn_ee: "callable | None" = None
    record_fn_awake_se: "callable | None" = None
    record_fn_awake_ee: "callable | None" = None
    x_tar_static_se: float = 0.2
    x_tar_static_ee: float = 0.2

    '''
    Trainer object takes neuron dynamics arrays (spike trace, membrane potential,
    STDP, clipping, regularization and tracking) and updates iteratively
    throughout training. 
    '''

    def __post_init__(self):
        # convert to numba-consistent integer type
        self.track_stats = np.uint8(self.track_stats)
        self.track_weights = np.uint8(self.track_weights)
        self.spike_adaption = np.uint8(self.spike_adaption)
        self.clip_weights = np.uint8(self.clip_weights)
        self.sleep = np.uint8(self.sleep)
        self.train_weights = np.uint8(self.train_weights)
        self.normalize_weights = np.uint8(self.normalize_weights)

        # initiate neuron object
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

        # initiate membrane object
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

        # initiate learner (STDP-object)
        self.learner = Learner(
            learning_rate=self.learning_rate,
            N_x=self.N_x,
            nonzero_pre_idx=self.nonzero_pre_idx,
            w_max=self.w_max,
            mu_weight=self.mu_weight,
        )

        # triplet STDP — slow-trace decays and persistent state arrays
        if self.use_triplet:
            self._decay_r2 = np.exp(-self.dt / self.tau_x)
            self._decay_o2 = np.exp(-self.dt / self.tau_y)
            self._r2_train = np.zeros(self.N_x + self.N_exc)
            self._o2_train = np.zeros(self.N_exc)
            self.triplet_learner = TripletLearner(
                learning_rate=self.learning_rate,
                N_x=self.N_x,
                nonzero_pre_idx=list(self.nonzero_pre_idx),
                w_max=self.w_max,
                w_min=self.min_weight_exc,
                mu_weight=self.mu_weight,
                A2_plus=self.A2_plus,
                A3_plus=self.A3_plus,
                A2_minus=self.A2_minus,
                A3_minus=self.A3_minus,
            )

        # initiate clipper object
        self.clipper = Clipper(
            nz_cols=self.nz_cols_exc,
            nz_rows=self.nz_rows_exc,
            min_weight_exc=self.min_weight_exc,
            max_weight_exc=self.max_weight_exc,
        )

        # initiate tracker object
        self.tracker = TrainTracker(
            N_exc=self.N_exc,
            st=self.st,
            ex=self.ex,
            ih=self.ih,
            track_stats=self.track_stats,
            track_weights=self.track_weights,
        )
        # catch regularization inconcistencies
        if self.sleep and self.normalize_weights:
            raise ValueError("sleep and normalize_weights cannot both be true.")

        # initiate sleep regularization
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
        # OR initiate normalization
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
        '''
        Step-function runs a full training batch and returns dynamic arrays
        '''

        # prepare run configurations based on run-type (train, val or test)
        if training_mode == "train":
            desc = "Training network:"
            track_stats = self.track_stats
            track_weights = self.track_weights
            save_plots = self.save_plots
            train_weights = self.train_weights # we only update weights during train
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

        # prepare dynamic arrays derived from existing arrays
        mp_prev = mp.copy()
        spikes_prev = spikes[0].copy()
        weights_exc = np.ascontiguousarray(weights[:, self.st : self.ex].T)
        weights_inh = np.ascontiguousarray(
            weights[self.st : self.ex, self.ex : self.ih].T
        )

        # slow traces: persistent during training, fresh zeros for val/test
        if self.use_triplet:
            if training_mode == "train":
                r2 = self._r2_train
                o2 = self._o2_train
            else:
                r2 = np.zeros(self.N_x + self.N_exc)
                o2 = np.zeros(self.N_exc)

        # initiate tqdm object
        pbar = tqdm(range(1, spikes.shape[0]), desc=desc, leave=False, mininterval=1.0, colour="green", ascii=" <><><><>><<>")

        # compare RAM usage across train, val and test to ensure no runaway RAM consumption
        report_RAM_usage()

        # intiate x target values (set to mean of starting weights)
        x_tar_se, x_tar_ee = update_x_tar(
            spike_trace=spike_trace,
            N_x=self.N_x,
            mode=self.x_tar_mode,
            pct_se=self.x_tar_pct_se,
            pct_ee=self.x_tar_pct_ee,
            static_se=self.x_tar_static_se,
            static_ee=self.x_tar_static_ee,
        )
        # loop across time T 
        for t in pbar:
            # check if weights and regularization should be active if train mode
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
            # check if stat tracking or network plotting should occur for the current timestep
            if t % self.stat_tracking_frequency == 0:
                if track_stats:
                    _track_stats = np.uint8(1)
                # check if spawning multi-thread plotter for current run (ony during train)
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
            # update dynamic arrays if sleep is about to begin 
            if sleep_remaining > 0:
                # extract recent spikes
                spikes_buf = spikes_prev.copy()
                # remove input spikes (no data training during sleep)
                spikes_buf[: self.st] = 0
                # pre-allocate empty spikes array
                current_spikes = np.zeros(self.ih, dtype=np.int8)
            
            ########## SLEEP PHASE ########## 

            # run while loop during sleep without iterating timestep t
            while sleep_remaining > 0:
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
                    noisy_potential_now=noisy_potential_now, # true during sleep
                )
                # reset current spiking array (array only 1xN, no time dimension)
                current_spikes[:] = 0
                # update dynamic neuron arrays 
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
                # update slow traces during sleep (keeps r2/o2 consistent with fast trace)
                if self.use_triplet:
                    r2, o2 = update_slow_traces(
                        current_spikes, r2, o2, self._decay_r2, self._decay_o2,
                        self.N_x, self.N_exc,
                    )

                # update weights every 10% of sleep timesteps (around 1% during wake period)
                if sleep_remaining % 10 == 0:
                    # clip weights if object initialized
                    if self.clip_weights:
                        weights = self.clipper.step(weights=weights)
                    # apply STDP (trace or triplet)
                    if self.use_triplet:
                        weights = self.triplet_learner.step(
                            weights=weights,
                            spikes=spikes_buf,
                            spike_trace=spike_trace,
                            r2=r2,
                            o2=o2,
                        )
                        m_x_pre, m_first_term, m_delta_w = 0.0, 0.0, 0.0
                    else:
                        weights, m_x_pre, m_first_term, m_delta_w = self.learner.step(
                            spike_trace=spike_trace,
                            weights=weights,
                            spikes=spikes_buf,
                            track_weights=np.uint8(0),
                            x_tar_se=x_tar_se,
                            x_tar_ee=x_tar_ee,
                        )
                    # convert to pre-transposed arrays for efficient membrane potential computing
                    np.copyto(weights_exc, weights[:, self.st : self.ex].T)
                    np.copyto(
                        weights_inh, weights[self.st : self.ex, self.ex : self.ih].T
                    )
                # regularize weights with sleep for SE (input to excitatory)
                weights[: self.st, self.st : self.ex] = self.sleep_se.step(
                    weights[: self.st, self.st : self.ex], t
                )
                # regularize weights with sleep for EE (excitatory to excitatory)
                weights[self.st : self.ex, self.st : self.ex] = self.sleep_ee.step(
                    weights[self.st : self.ex, self.st : self.ex], t
                )
                # decrease remaining sleep duration
                sleep_remaining -= 1
                # update current membrane potential to newest
                mp_prev = mp
                # turn off noisy membrane potential if sleep is finished
                if sleep_remaining <= 0:
                    noisy_potential_now = np.uint8(0)

            ########## AWAKE PHASE ##########

            # update membrane potential based on spiking activity
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
            # update dynamic neuron arrays
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

            # update slow traces with current timestep's spikes (awake phase)
            if self.use_triplet:
                r2, o2 = update_slow_traces(
                    spikes[t], r2, o2, self._decay_r2, self._decay_o2,
                    self.N_x, self.N_exc,
                )

            # clip, update and regularize weights if train mode
            if update_weights_now:
                # apply clipping if object initiated
                if self.clip_weights:
                    weights = self.clipper.step(weights=weights)
                # apply STDP (trace or triplet)
                if self.use_triplet:
                    weights = self.triplet_learner.step(
                        weights=weights,
                        spikes=spikes_prev,
                        spike_trace=spike_trace,
                        r2=r2,
                        o2=o2,
                    )
                    m_x_pre, m_first_term, m_delta_w = 0.0, 0.0, 0.0
                else:
                    weights, m_x_pre, m_first_term, m_delta_w = self.learner.step(
                        spike_trace=spike_trace,
                        weights=weights,
                        spikes=spikes_prev,
                        x_tar_se=x_tar_se,
                        x_tar_ee=x_tar_ee,
                        track_weights=track_weights,
                    )
                # apply normalization
                if normalize_now:
                    weights[: self.st, self.st : self.ex] = self.norm_se.step(
                        weights[: self.st, self.st : self.ex], t
                    )
                    weights[self.st : self.ex, self.st : self.ex] = self.norm_ee.step(
                        weights[self.st : self.ex, self.st : self.ex], t
                    )
                # update target arrays
                x_tar_se, x_tar_ee = update_x_tar(
                    spike_trace=spike_trace,
                    N_x=self.N_x,
                    mode=self.x_tar_mode,
                    pct_se=self.x_tar_pct_se,
                    pct_ee=self.x_tar_pct_ee,
                    static_se=self.x_tar_static_se,
                    static_ee=self.x_tar_static_ee,
                )
                # update synapse tracking
                if track_weights:
                    self.tracker.track_synapse(
                        m_x_pre,
                        m_first_term,
                        m_delta_w,
                        m_ltp,
                        m_ltd,
                        x_tar_se,
                        x_tar_ee,
                    )
                # pre-transpose weight arrays for efficient handling
                np.copyto(weights_exc, weights[:, self.st : self.ex].T)
                np.copyto(weights_inh, weights[self.st : self.ex, self.ex : self.ih].T)
                # record weights if tracking object initiated
                if self.record_fn_awake_se is not None:
                    self.record_fn_awake_se(weights[: self.st, self.st : self.ex], t)
                if self.record_fn_awake_ee is not None:
                    self.record_fn_awake_ee(
                        weights[self.st : self.ex, self.st : self.ex], t
                    )
                # turn off weights updating and normalization (only performed periodically)
                update_weights_now = np.uint8(0)
                normalize_now = np.uint8(0)
            # track neuron stats 
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
            # map previous membrane potential and spikes to current
            mp_prev = mp
            spikes_prev = spikes[t]

        # print stats if object exists
        self.tracker.print(
            weights=weights,
            spikes=spikes,
            track_weights=track_weights,
            track_stats=track_stats,
            spike_trace=spike_trace,
            training_mode=training_mode,
            x_tar_se=x_tar_se,
            x_tar_ee=x_tar_ee,
            x_tar_mode=self.x_tar_mode,
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
