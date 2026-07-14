# Quickstart

This page walks through a complete training run end to end. Every step is annotated; the
[Core concepts](core-concepts.md) page explains *why* the API is shaped this way.

## A full training script

```python
import neurosnn as snn

# 1. Define connectivity.
#    Factory functions return a WeightsSpec describing densities and peak amplitudes
#    for the four pathways (S→E, E→E, E→I, I→E).
weights = snn.weights.random(
    density_se=0.05, density_ee=0.05, density_ei=0.05, density_ie=0.05,
    peak_se=1.0,     peak_ee=0.5,     peak_ei=1.0,     peak_ie=-0.7,
)

# 2. Build a layer — one excitatory/inhibitory population pair plus its membrane dynamics.
layer = snn.Layer(
    N_exc=1024,
    N_inh=225,
    membrane=snn.membrane.LIF(
        tau_m_exc=20.0,   tau_m_inh=15.0,
        tau_syn_exc=10.0, tau_syn_inh=9.0,
        membrane_resistance_exc=15.0,
        membrane_resistance_inh=15.0,
        resting_potential=-70.0,
        reset_potential=-80.0,
        spike_threshold=-55.0,
    ),
    weights=weights,
)

# 3. Configure the learning rule.
learner = snn.learner.TraceSTDP(
    learning_rate=0.0004,
    tau_trace=20,
    w_max=10.0,
    mu_weight=0.6,
)

# 4. Configure the model — data source and training bookkeeping.
model = snn.Model(
    input_size=784,                 # 28×28 MNIST, flattened
    classes=list(range(10)),
    num_steps=350,                  # timesteps each image is presented for
    image_dataset="mnist",
    all_images_train=5000, batch_image_train=1000,
    all_images_val=1000,   batch_image_val=1000,
    all_images_test=1000,  batch_image_test=1000,
)

# 5. Train. model.train() is a generator: one TrainResult per batch.
for result in model.train(layers=[layer], learner=learner, epochs=3):
    if result.batch % 5 == 0 and result.accuracy is not None:
        val = model.validate()
        print(
            f"epoch {result.epoch + 1}  batch {result.batch}"
            f"  train {result.accuracy:.3f}  val {val.accuracy:.3f}"
        )

print(f"Test accuracy: {model.test().accuracy:.3f}")
```

## Adding sleep regularisation

Sleep-like synaptic homeostasis is opt-in through a single extra argument. Everything else
stays the same:

```python
regularizer = snn.regularizer.Sleep(duration=200, frequency=1050, mode="neuron")

for result in model.train(layers=[layer], learner=learner,
                          regularizer=regularizer, epochs=3):
    ...
```

See [Sleep & homeostasis](../guides/sleep-homeostasis.md) for what the `mode` values mean and
when to prefer `Sleep` over `Normalize`.

!!! note "Inspecting the network before training"
    Weights are built from the `Layer` spec on the **first** generator iteration. To inspect
    or plot the initial weight matrix before any learning happens, advance the generator once
    with `next(...)` and read `result.weights`.

## Next steps

- [Core concepts](core-concepts.md) — the mental model behind `Model`, `Layer`, `learner`, and `regularizer`.
- [The training loop](../guides/training-loop.md) — `TrainResult`, `validate()`, `test()`, and the generator pattern in depth.
