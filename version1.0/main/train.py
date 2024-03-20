# Create script to train network
import numpy as np
from tqdm import tqdm


def train_data(
    R: float | int,
    A: float | int,
    B: float | int,
    beta: float | int,
    delta: float | int,
    ideal_w: float | int | np.ndarray,
    time: int,
    V_th: int,
    V_rest: int,
    V_reset: int,
    dt: float | int,
    tau_m: float | int,
    training_data: np.ndarray,
    N_excit_neurons: int,
    N_inhib_neurons: int,
    N_input_neurons: int,
    MemPot: np.ndarray,
    W_se: np.ndarray,
    W_ee: np.ndarray,
    W_ei: np.ndarray,
    W_ie: np.ndarray,
):
    num_neurons = N_excit_neurons + N_inhib_neurons + N_input_neurons
    spikes = np.zeros((time, num_neurons))
    pre_synaptic_trace = np.zeros((time, num_neurons - N_input_neurons))
    post_synaptic_trace = np.zeros((time, num_neurons - N_input_neurons))

    # Add input data before training for input neurons
    spikes[:, :N_input_neurons] = training_data

    # Loop through time and update membrane potential, spikes and weights
    for t in tqdm(range(1, time), desc="Training network"):

        # Decay traces
        pre_synaptic_trace *= np.exp(-dt / tau_m)
        post_synaptic_trace *= np.exp(-dt / tau_m)

        # Update stimulation-excitation, excitation-excitation and inhibitory-excitatory synapses
        for n in range(N_excit_neurons):

            # Update incoming spikes as I_in
            I_in = (
                np.dot(
                    W_se[t, :, n],
                    spikes[t, :N_input_neurons],
                )
                + np.dot(
                    W_ee[t, :, n],
                    spikes[
                        t,
                        N_input_neurons + 1 : N_input_neurons + N_excit_neurons,
                    ],
                )
                + np.dot(
                    W_ie[t, :, n],
                    spikes[t, n + N_input_neurons + N_excit_neurons + 1 :],
                )
            )

            # Update membrane potential based on I_in
            MemPot[t, n] = (
                MemPot[t - 1, n] + (-MemPot[t - 1, n] + V_rest + R * I_in) / tau_m
            )

            # Update spikes
            if MemPot[t, n] > V_th:
                spikes[t, n + N_input_neurons + 1] = 1
                pre_synaptic_trace[t, n + N_input_neurons + 1] += 1
                post_synaptic_trace[t, n + N_input_neurons + 1] += 1
                MemPot[t, n] = V_reset
            else:
                spikes[t, n + N_input_neurons + 1] = 0

            # Get all pre-synaptic indices
            pre_syn_indices = np.nonzero(W_se[t, :, n])

            # Loop through each synapse to update strength
            for s in range(pre_syn_indices):

                # Use the current trace values for STDP calculation
                pre_trace = pre_synaptic_trace[t, s]
                post_trace = post_synaptic_trace[t, n + N_input_neurons + 1]

                # Get learning components
                hebb = A * pre_trace * post_trace**2 - B * pre_trace * post_trace
                hetero_syn = -beta * (W_se[t, s, n] - ideal_w) * post_trace**4
                dopamine_reg = delta * pre_trace

                # Assemble components to update weight
                W_se[t, s, n] = hebb + hetero_syn + dopamine_reg

            # Get all pre-synaptic indices
            pre_syn_indices = np.nonzero(W_ee[t, :, n])

            # Loop through each synapse to update strength
            for s in range(pre_syn_indices):
                if s == n:
                    raise UserWarning(
                        "There are self-connections within the W_ee array"
                    )

                # Use the current trace values for STDP calculation
                pre_trace = pre_synaptic_trace[t, s + N_excit_neurons]
                post_trace = post_synaptic_trace[t, n + N_excit_neurons]

                # Get learning components
                hebb = A * pre_trace * post_trace**2 - B * pre_trace * post_trace
                hetero_syn = -beta * (W_ee[t, s, n] - ideal_w) * post_trace**4
                dopamine_reg = delta * pre_trace

                # Assemble components to update weight
                W_ee[t, s, n] = hebb + hetero_syn + dopamine_reg

        # Update excitatory-inhibitory weights
        for n in range(N_excit_neurons):

            # Update incoming spikes as I_in
            I_in = np.dot(
                W_ie[t, :, n],
                spikes[
                    t,
                    N_input_neurons + 1 : N_input_neurons + N_excit_neurons,
                ],
            )
            # Update membrane potential based on I_in
            MemPot[t, n + N_excit_neurons + 1] = (
                MemPot[t - 1, n + N_excit_neurons + 1]
                + (-MemPot[t - 1, n + N_excit_neurons + 1] + V_rest + R * I_in) / tau_m
            )

            # Update spikes
            if MemPot[t, n + N_excit_neurons + 1] > V_th:
                spikes[t, n + N_input_neurons + N_excit_neurons + 1] = 1
                pre_synaptic_trace[t, n + N_input_neurons + N_excit_neurons + 1] += 1
                post_synaptic_trace[t, n + N_input_neurons + N_excit_neurons + 1] += 1
                MemPot[t, n + N_excit_neurons + 1] = V_reset
            else:
                spikes[t, n + N_input_neurons + N_excit_neurons + 1] = 0

        # Update excitatory-inhibitory weights
        for n in range(N_inhib_neurons):

            # Update incoming spikes as I_in
            I_in = np.dot(
                W_ei[t, :, n],
                spikes[t, N_input_neurons + N_excit_neurons + 1 :],
            )
            # Update membrane potential based on I_in
            MemPot[t, n + N_excit_neurons] = (
                MemPot[t - 1, n + N_excit_neurons]
                + (-MemPot[t - 1, n + N_excit_neurons] + V_rest + R * I_in) / tau_m
            )

            # Update spikes
            if MemPot[t, n] > V_th:
                spikes[t, n + N_input_neurons + N_excit_neurons] = 1
                pre_synaptic_trace[t, n + N_input_neurons + N_excit_neurons] += 1
                post_synaptic_trace[t, n + N_input_neurons + N_excit_neurons] += 1
                MemPot[t, n + N_excit_neurons] = V_reset
            else:
                spikes[t, n + N_input_neurons + N_excit_neurons] = 0

    return (
        spikes,
        MemPot,
        W_se,
        W_ee,
        W_ei,
        W_ie,
        pre_synaptic_trace,
        post_synaptic_trace,
        num_neurons,
    )
