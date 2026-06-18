# Background & citation

## The model

`neurosnn` implements a biologically plausible spiking network in the tradition of
unsupervised, STDP-trained SNNs for visual classification. Its architecture — an
excitatory/inhibitory population with feedforward stimulus drive, recurrent excitation, and
feedback inhibition, trained by spike-trace STDP with homeostatic weight regulation — follows
and extends the model of [Zenke et al. (2015)](https://www.nature.com/articles/ncomms7922),
with a learning rule in the family described by
[Diehl & Cook (2015)](https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full).

The library is the software basis for two research articles:

1. **Sleep-Based Homeostatic Regularization for Stabilizing Spike-Timing-Dependent Plasticity in Recurrent Spiking Neural Networks** — Reproducing core functionality from Zenke with custom learning rule and static sleep regularization method employed [Massey et al. (2026)](https://arxiv.org/abs/2601.08447).
2. **Exploring napping paradigm for Recurrent Spiking Neural Networks** — studying how sleep-like synaptic downscaling
   (the [`Sleep`](../guides/sleep-homeostasis.md) regularizer) interacts with STDP and compare against normalization (article in IWAI conference proceedings).

## What it extends

Two aspects go beyond the original formulations and are worth flagging for readers comparing
against the source papers:

- **Dynamic `x_tar`.** The STDP target trace is computed as the *running population-mean
  trace* (per pathway) rather than held as a fixed constant, making it a moving homeostatic
  balance point. See [Learning (Trace STDP)](../guides/learning-stdp.md).
- **Sleep as an explicit process.** Beyond deterministic weight normalization, `neurosnn`
  models sleep episodes with silenced input, spontaneous noise-driven activity, and gradual
  multiplicative downscaling. See [Sleep & homeostasis](../guides/sleep-homeostasis.md).

## Citation

!!! note "Citation"
    Citation details for the two articles will be added on publication. For now, please cite
    the repository and the foundational works:

  @article{massey2026sleep,
      title={Sleep-Based Homeostatic Regularization for Stabilizing Spike-Timing-Dependent Plasticity in Recurrent Spiking Neural Networks},
      author={Massey, Andreas and Hubin, Aliaksandr and Nichele, Stefano and S{\ae}b{\o}, Solve},
      year={2026},
      eprint={2601.08447},
      archivePrefix={arXiv},
      primaryClass={cs.NE},
      url={https://arxiv.org/abs/2601.08447},
      journal={arXiv preprint arXiv:2601.08447}
  }

## See also

- [Reproducing experiments](experiments.md) — the experiment scripts behind the articles.
