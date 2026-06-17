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

1. **A Zenke et al. replication study** — reproducing the core dynamics and learning behaviour.
2. **A sleep-protocol homeostasis article** — studying how sleep-like synaptic downscaling
   (the [`Sleep`](../guides/sleep-homeostasis.md) regularizer) interacts with STDP.

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

    - Zenke, F., Agnes, E. J., & Gerstner, W. (2015). *Diverse synaptic plasticity mechanisms
      orchestrated to form and retrieve memories in spiking neural networks.* Nature
      Communications, 6, 6922.
    - Diehl, P. U., & Cook, M. (2015). *Unsupervised learning of digit recognition using
      spike-timing-dependent plasticity.* Frontiers in Computational Neuroscience, 9, 99.

## See also

- [Reproducing experiments](experiments.md) — the experiment scripts behind the articles.
