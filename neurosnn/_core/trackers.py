from dataclasses import dataclass
from typing import Optional


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

    def track_synapse(self, m_x_pre, m_first_term, m_delta_w, x_tar_se, x_tar_ee):
        self.x_pre_sum += m_x_pre
        self.first_term_sum += m_first_term
        self.delta_w_sum += m_delta_w
        self.x_tar_sum_se += x_tar_se.mean()
        self.x_tar_sum_ee += x_tar_ee.mean()
        self.x_tar_count += 1

    def to_dict(self) -> Optional[dict]:
        if self.track_count == 0 and self.x_tar_count == 0:
            return None
        n = max(1, self.track_count)
        nx = max(1, self.x_tar_count)
        return {
            "mean_mp_exc": self.mp_ex / n,
            "mean_mp_inh": self.mp_ih / n,
            "mean_delta_mp_exc": self.delta_mp_ex / n,
            "mean_delta_mp_inh": self.delta_mp_ih / n,
            "mean_I_syn_exc": self.I_syn_ex / n,
            "mean_I_syn_inh": self.I_syn_ih / n,
            "mean_delta_I_syn_exc": self.delta_I_syn_ex / n,
            "mean_delta_I_syn_inh": self.delta_I_syn_ih / n,
            "mean_adaptation": self.a_ex / n,
            "mean_spike_threshold": self.spike_threshold_ex / n,
            "mean_spike_trace": self.spike_trace_ex / n,
            "mean_x_pre": self.x_pre_sum / nx,
            "mean_first_term": self.first_term_sum / nx,
            "mean_delta_w": self.delta_w_sum / nx,
            "mean_x_tar_se": self.x_tar_sum_se / nx,
            "mean_x_tar_ee": self.x_tar_sum_ee / nx,
        }

    def print(self, weights, spikes, track_weights, track_stats):
        if not (track_stats or track_weights):
            return

        d = self.to_dict()
        if d is None:
            return

        spikes_st = spikes[:, : self.st].mean(axis=0)
        spikes_ex = spikes[:, self.st : self.ex].mean(axis=0)
        spikes_ih = spikes[:, self.ex : self.ih].mean(axis=0)

        if track_stats:
            print(f"Mean delta mp exc:        {d['mean_delta_mp_exc']:.5f}")
            print(f"Mean delta mp inh:        {d['mean_delta_mp_inh']:.5f}")
            print(f"Mean delta I syn exc:     {d['mean_delta_I_syn_exc']:.5f}")
            print(f"Mean delta I syn inh:     {d['mean_delta_I_syn_inh']:.5f}")
            print(f"Mean membrane pot exc:    {d['mean_mp_exc']:.5f}")
            print(f"Mean membrane pot inh:    {d['mean_mp_inh']:.5f}")
            print(f"Mean I syn exc:           {d['mean_I_syn_exc']:.5f}")
            print(f"Mean I syn inh:           {d['mean_I_syn_inh']:.5f}")
            print(f"Mean adaptation:          {d['mean_adaptation']:.5f}")
            print(f"Mean spike threshold:     {d['mean_spike_threshold']:.5f}")
            print(f"Mean spike trace:         {d['mean_spike_trace']:.5f}")
            print(f"Mean spikes input:        {spikes_st.mean():.5f}")
            print(f"Mean spikes exc:          {spikes_ex.mean():.5f}")
            print(f"Mean spikes inh:          {spikes_ih.mean():.5f}")

        if track_weights:
            nz = lambda block: block[block != 0]
            print(f"Mean x_pre:               {d['mean_x_pre']:.5f}")
            print(f"Mean first term:          {d['mean_first_term']:.5f}")
            print(f"Mean delta_w:             {d['mean_delta_w']:.5f}")
            print(f"Mean x_tar SE:            {d['mean_x_tar_se']:.5f}")
            print(f"Mean x_tar EE:            {d['mean_x_tar_ee']:.5f}")
            w = weights
            print(f"Weights ST->EX mean:      {nz(w[:self.st, self.st:self.ex]).mean():.5f}")
            print(f"Weights EX->EX mean:      {nz(w[self.st:self.ex, self.st:self.ex]).mean():.5f}")
            print(f"Weights EX->IH mean:      {nz(w[self.st:self.ex, self.ex:self.ih]).mean():.5f}")
            print(f"Weights IH->EX mean:      {nz(w[self.ex:self.ih, self.st:self.ex]).mean():.5f}")
            print(f"Weights ST->EX std:       {nz(w[:self.st, self.st:self.ex]).std():.5f}")
            print(f"Weights EX->EX std:       {nz(w[self.st:self.ex, self.st:self.ex]).std():.5f}")
            print(f"Weights EX->IH std:       {nz(w[self.st:self.ex, self.ex:self.ih]).std():.5f}")
