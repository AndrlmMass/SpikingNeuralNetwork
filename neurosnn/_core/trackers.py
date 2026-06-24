from dataclasses import dataclass
from typing import Optional

from neurosnn._utils.logger import (
    rf_within_concentration,
    rf_diversity,
    ei_balance,
    population_activity,
    trace_spread,
)


@dataclass
class TrainTracker:
    N_exc: int
    st: int
    ex: int
    ih: int
    track_stats: bool
    track_weights: bool

    '''
    Tracking object for neuron dynamics arrays. Initiates trackers, then periodically computes mean
    scores per tracking dimension when called. Optional print function. 
    '''

    def __post_init__(self):
        # resets object for every new training run
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
        # snapshot diagnostics (RF-collapse harness); computed once in print()
        self._diag = {}

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
        '''
        Computes mean scores for tracking periodically when called. 
        '''
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
        '''
        Computes mean scores for tracking periodically when called. 
        '''
        self.x_pre_sum += m_x_pre
        self.first_term_sum += m_first_term
        self.delta_w_sum += m_delta_w
        self.x_tar_sum_se += x_tar_se.mean()
        self.x_tar_sum_ee += x_tar_ee.mean()
        self.x_tar_count += 1

    def to_dict(self) -> Optional[dict]:
        '''
        Store variables in dict. 
        '''
        if self.track_count == 0 and self.x_tar_count == 0:
            return None
        n = max(1, self.track_count)
        nx = max(1, self.x_tar_count)
        d = {
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
        # propagate snapshot diagnostics so they flow to TrainResult.stats
        d.update(self._diag)
        return d

    def print(self, weights, spikes, track_weights, track_stats, spike_trace=None):
        '''
        Print function for runtime tracking
        '''
        if not (track_stats or track_weights):
            return
        # extract dict
        d = self.to_dict()
        if d is None:
            return
        # compute spike means
        spikes_st = spikes[:, : self.st].mean(axis=0)
        spikes_ex = spikes[:, self.st : self.ex].mean(axis=0)
        spikes_ih = spikes[:, self.ex : self.ih].mean(axis=0)

        # print stats
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
            self._print_diagnostics(weights, spikes, spike_trace)

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

    def _print_diagnostics(self, weights, spikes, spike_trace):
        '''
        RF-collapse harness: compute the five snapshot diagnostics, store the
        scalars on self._diag (so to_dict propagates them to TrainResult.stats),
        and print them grouped by synapse- vs neuron-side.
        '''
        W_se = weights[: self.st, self.st : self.ex]
        W_ee = weights[self.st : self.ex, self.st : self.ex]
        # synapse-side: SE (feed-forward RFs) and EE (recurrent) tracked separately
        # so the causal test can see whether the recurrent matrix itself collapses
        # (rows becoming correlated) vs just transmitting SE collapse.
        rf_entropy, rf_gini = rf_within_concentration(W_se)
        mean_cos, pr, pr_norm = rf_diversity(W_se)
        ee_entropy, ee_gini = rf_within_concentration(W_ee)
        ee_cos, ee_pr, ee_pr_norm = rf_diversity(W_ee)
        ei_med, ei_p90, _ = ei_balance(weights, spikes, self.st, self.ex, self.ih)
        trace_p50, trace_p90 = trace_spread(spike_trace, self.st, self.ex)
        # neuron-side
        active_frac, sparseness, _ = population_activity(spikes, self.st, self.ex)

        self._diag = {
            "rf_entropy": rf_entropy,
            "rf_gini": rf_gini,
            "rf_mean_cosine": mean_cos,
            "rf_participation_ratio": pr,
            "rf_pr_norm": pr_norm,
            "ee_entropy": ee_entropy,
            "ee_gini": ee_gini,
            "ee_mean_cosine": ee_cos,
            "ee_participation_ratio": ee_pr,
            "ee_pr_norm": ee_pr_norm,
            "ei_ratio_median": ei_med,
            "ei_ratio_p90": ei_p90,
            "trace_p50": trace_p50,
            "trace_p90": trace_p90,
            "active_frac_exc": active_frac,
            "pop_sparseness": sparseness,
        }

        print("--- Diagnostics (synapse: SE feed-forward) ---")
        print(f"RF entropy (nats):        {rf_entropy:.5f}")
        print(f"RF Gini:                  {rf_gini:.5f}")
        print(f"RF mean cosine:           {mean_cos:.5f}")
        print(f"RF participation ratio:   {pr:.3f} (norm {pr_norm:.5f})")
        print("--- Diagnostics (synapse: EE recurrent) ---")
        print(f"EE entropy (nats):        {ee_entropy:.5f}")
        print(f"EE Gini:                  {ee_gini:.5f}")
        print(f"EE mean cosine:           {ee_cos:.5f}")
        print(f"EE participation ratio:   {ee_pr:.3f} (norm {ee_pr_norm:.5f})")
        print("--- Diagnostics (balance/trace) ---")
        print(f"E/I ratio median:         {ei_med:.5f}")
        print(f"E/I ratio p90:            {ei_p90:.5f}")
        print(f"Trace p50/p90:            {trace_p50:.5f} / {trace_p90:.5f}")
        print("--- Diagnostics (neuron) ---")
        print(f"Active-E fraction:        {active_frac:.5f}")
        print(f"Population sparseness:    {sparseness:.5f}")
