from dataclasses import dataclass


@dataclass
class TrainTracker:
    N_exc: int
    st: int
    ex: int
    ih: int
    track_stats: bool
    track_weights: bool

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.delta_mp_ex = self.delta_mp_ih = 0.0
        self.mp_ex = self.mp_ih = 0.0
        self.delta_I_syn_ex = self.delta_I_syn_ih = 0.0
        self.I_syn_ex = self.I_syn_ih = 0.0
        self.a_ex = self.spike_threshold_ex = self.spike_trace_ex = 0.0
        self.track_count = self.x_tar_count = 0
        self.x_tar_sum_se = self.x_tar_sum_ee = 0.0
        self.first_term_sum = self.delta_w_sum = self.x_pre_sum = 0.0
        self.second_term_sum = 0.0

    def track_neuron(
        self,
        mp,
        delta_mp_ex_,
        delta_mp_ih_,
        I_syn_exc,
        I_syn_inh,
        delta_I_syn_ex_,
        delta_I_syn_ih_,
        a,
        spike_threshold,
        spike_trace,
    ):
        self.delta_mp_ex += delta_mp_ex_.mean()
        self.delta_mp_ih += delta_mp_ih_.mean()
        self.mp_ex += mp[: self.N_exc].mean()
        self.mp_ih += mp[self.N_exc :].mean()
        self.delta_I_syn_ex += delta_I_syn_ex_.mean()
        self.delta_I_syn_ih += delta_I_syn_ih_.mean()
        self.I_syn_ex += I_syn_exc.mean()
        self.I_syn_ih += I_syn_inh.mean()
        self.a_ex += a.mean()
        self.spike_threshold_ex += spike_threshold.mean()
        self.spike_trace_ex += spike_trace.mean()
        self.track_count += 1

    def track_synapse(
        self,
        m_x_pre,
        m_first_term,
        m_second_term,
        m_delta_w,
        x_tar_se,
        x_tar_ee,
    ):
        self.x_pre_sum += m_x_pre
        self.first_term_sum += m_first_term
        self.second_term_sum += m_second_term
        self.delta_w_sum += m_delta_w
        self.x_tar_sum_se += x_tar_se.mean()
        self.x_tar_sum_ee += x_tar_ee.mean()
        self.x_tar_count += 1

    def print(self, weights, spikes, track_weights, track_stats):
        if not (self.track_stats or self.track_weights):
            return

        # compute the mean of the trackers
        mean_delta_mp_ex = self.delta_mp_ex / max(1, self.track_count)
        mean_delta_mp_ih = self.delta_mp_ih / max(1, self.track_count)
        mean_mp_ex = self.mp_ex / max(1, self.track_count)
        mean_mp_ih = self.mp_ih / max(1, self.track_count)
        mean_delta_I_syn_ex = self.delta_I_syn_ex / max(1, self.track_count)
        mean_delta_I_syn_ih = self.delta_I_syn_ih / max(1, self.track_count)
        mean_I_syn_ex = self.I_syn_ex / max(1, self.track_count)
        mean_I_syn_ih = self.I_syn_ih / max(1, self.track_count)
        mean_a_ex = self.a_ex / max(1, self.track_count)
        # mean_a_ih = a_ih.mean()
        mean_spike_threshold_ex = self.spike_threshold_ex / max(1, self.track_count)
        # mean_spike_threshold_ih = spike_threshold_ih.mean()
        mean_spike_trace_ex = self.spike_trace_ex / max(1, self.track_count)
        spikes_st = spikes[:, : self.st].mean(axis=0)
        spikes_ex = spikes[:, self.st : self.ex].mean(axis=0)
        spikes_ih = spikes[:, self.ex : self.ih].mean(axis=0)

        if track_stats:
            print(f"Mean delta mp ex: {mean_delta_mp_ex}")
            print(f"Mean delta mp ih: {mean_delta_mp_ih}")
            print(f"Mean delta I syn ex: {mean_delta_I_syn_ex}")
            print(f"Mean delta I syn ih: {mean_delta_I_syn_ih}")
            print(f"Mean membrane potential ex: {mean_mp_ex}")
            print(f"Mean membrane potential ih: {mean_mp_ih}")
            print(f"Mean I syn ex: {mean_I_syn_ex}")
            print(f"Mean I syn ih: {mean_I_syn_ih}")
            print(f"Mean a ex: {mean_a_ex}")
            # print(f"Mean a ih: {mean_a_ih}")
            print(f"Mean spike threshold ex: {mean_spike_threshold_ex}")
            # print(f"Mean spike threshold ih: {mean_spike_threshold_ih}")
            print(f"Mean spikes st: {spikes_st.mean()}")
            print(f"Mean spikes ih: {spikes_ih.mean()}")
            print(f"Mean spikes ex: {spikes_ex.mean()}")
            print(f"Mean spike trace ex: {mean_spike_trace_ex}")
            print(
                f"weights st->ex: ",
                weights[: self.st, self.st : self.ex][
                    weights[: self.st, self.st : self.ex] != 0
                ].mean(),
            )
            print(
                f"weights ex->ex: ",
                weights[self.st : self.ex, self.st : self.ex][
                    weights[self.st : self.ex, self.st : self.ex] != 0
                ].mean(),
            )
            print(
                f"weights ex->ih: ",
                weights[self.st : self.ex, self.ex : self.ih][
                    weights[self.st : self.ex, self.ex : self.ih] != 0
                ].mean(),
            )
            print(
                f"weights ih->ex: ",
                weights[self.ex : self.ih, self.st : self.ex][
                    weights[self.ex : self.ih, self.st : self.ex] != 0
                ].mean(),
            )
            # After each batch, print these:
            print(
                "std ex->ih:",
                weights[self.st : self.ex, self.ex : self.ih][
                    weights[self.st : self.ex, self.ex : self.ih] != 0
                ].std(),
            )
            print(
                "std ex->ex:",
                weights[self.st : self.ex, self.st : self.ex][
                    weights[self.st : self.ex, self.st : self.ex] != 0
                ].std(),
            )
            print(
                "std st->ex:",
                weights[: self.st, self.st : self.ex][
                    weights[: self.st, self.st : self.ex] != 0
                ].std(),
            )
        if track_weights:
            # calculate mean
            x_pre_mean = self.x_pre_mean / self.x_tar_count
            first_term_sum = self.m_first_term / self.x_tar_count
            second_term_mean = self.m_second_term / self.x_tar_count
            delta_w_sum = self.m_delta_w / self.x_tar_count
            x_tar_mean_se = self.x_tar_se / self.x_tar_count
            x_tar_mean_ee = self.x_tar_ee / self.x_tar_count
            print(f"Mean x_pre: {x_pre_mean}")
            print(f"Mean first_term: {first_term_sum}")
            print(f"Mean second_term: {second_term_mean}")
            print(f"Mean delta_w: {delta_w_sum}")
            print(f"Mean x_tar se: {x_tar_mean_se}")
            print(f"Mean x_tar ee: {x_tar_mean_ee}")
