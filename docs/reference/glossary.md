# Glossary

Short definitions of the biological and machine-learning terms used throughout these docs.

**Adaptation (spike-frequency adaptation)**
: A mechanism that raises a neuron's effective threshold after each spike and decays back
  over time, briefly making recently-active neurons harder to re-excite. See
  [Neuron dynamics](../guides/neuron-dynamics.md).

**BCM rule**
: A theory of synaptic plasticity (Bienenstock–Cooper–Munro) in which the direction of
  weight change depends on activity relative to a sliding threshold. `neurosnn`'s soft
  weight bound is described as "BCM-style."

**E / I (excitatory / inhibitory)**
: The two neuron populations in a `Layer`. Excitatory neurons drive their targets up;
  inhibitory neurons drive them down. See [Connectivity](../guides/connectivity.md).

**Homeostasis**
: Regulatory processes that keep network activity (here, total synaptic weight) within a
  stable range. Implemented by the [Sleep / Normalize](../guides/sleep-homeostasis.md)
  regularizers.

**LIF (Leaky Integrate-and-Fire)**
: The neuron model: membrane potential integrates input current, leaks toward rest, and
  fires a spike when it crosses threshold. See [Neuron dynamics](../guides/neuron-dynamics.md).

**Pathway (SE / EE / EI / IE)**
: A directed class of connections between populations — Stimulus→Excitatory,
  Excitatory→Excitatory, Excitatory→Inhibitory, Inhibitory→Excitatory.

**Phi (φ, η²)**
: A class-separation metric: the fraction of representational variance explained by class
  labels. See [Diagnostics & metrics](../guides/diagnostics.md).

**Poisson rate coding**
: Encoding a real-valued input as a spike train whose rate is proportional to the value, with
  spikes drawn stochastically. See [Data & input encoding](../guides/data-encoding.md).

**Receptive field (RF)**
: The spatial pattern of inputs a neuron is connected to, often a Gaussian or oriented blob.
  See [Connectivity](../guides/connectivity.md).

**Soft weight bound**
: A multiplicative `(w_max − w)` term that slows weight growth as it approaches `w_max`,
  causing graceful saturation instead of unbounded growth. See
  [Learning](../guides/learning-stdp.md).

**Sleep episode**
: A period during which input is silenced, membrane noise drives spontaneous activity, and
  weights are gradually downscaled. See [Sleep & homeostasis](../guides/sleep-homeostasis.md).

**Spike trace**
: A low-pass-filtered record of a neuron's recent spiking: it jumps on each spike and decays
  exponentially. Drives the STDP rule. See [Learning](../guides/learning-stdp.md).

**STDP (Spike-Timing-Dependent Plasticity)**
: A learning rule where synaptic change depends on the relative timing of pre- and
  postsynaptic activity. `neurosnn` uses a trace-based formulation.

**x_tar**
: The population-mean spike trace used as the homeostatic balance point in the STDP rule,
  computed separately for the SE and EE pathways.
