# Train network script
import numpy as np
from tqdm import tqdm
import pdb
import os
import sys

# Set current working directories and add relevant directories to path
if os.path.exists(
    "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.0"
):
    os.chdir(
        "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.0"
    )
    sys.path.append(
        "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.0\\plot"
    )
    sys.path.append(
        "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.0\\tool"
    )
else:
    os.chdir(
        "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\version1.0"
    )
    sys.path.append(
        "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\version1.0\\plot"
    )
    sys.path.append(
        "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\version1.0\\tool"
    )

from plot_training import *


def train_data(
    R: float | int,
    A: float | int,
    B: float | int,
    P: float | int,
    w_p: float | int,
    beta: float | int,
    delta: float | int,
    time: int,
    V_th: int,
    V_rest: int,
    V_reset: int,
    dt: float | int,
    tau_m: float | int,
    tau_const: float | int,
    training_data: np.ndarray,
    N_excit_neurons: int,
    N_inhib_neurons: int,
    N_input_neurons: int,
    MemPot: np.ndarray,
    max_weight: float | int,
    min_weight: float | int,
    W_se: np.ndarray,
    W_se_ideal: np.ndarray,
    W_ee: np.ndarray,
    W_ee_ideal: np.ndarray,
    W_ei: np.ndarray,
    W_ei_ideal: np.ndarray,
    W_ie: np.ndarray,
    W_ie_ideal: np.ndarray,
    update_frequency: int,
    callback: None,
):
    num_neurons = N_excit_neurons + N_inhib_neurons + N_input_neurons
    spikes = np.zeros((time, num_neurons))
    pre_synaptic_trace = np.zeros((num_neurons))
    post_synaptic_trace = np.zeros((num_neurons - N_input_neurons))
    # Add input data before training for input neurons
    spikes[:, :N_input_neurons] = training_data

    I_in_sum = []

    # Loop through time and update membrane potential, spikes and weights
    for t in tqdm(range(1, time - 1), desc="Training network"):

        # Update decay traces
        pre_synaptic_trace *= np.exp(-dt / tau_m)
        post_synaptic_trace *= np.exp(-dt / tau_m)

        # Update stimulation-excitation, excitation-excitation and inhibitory-excitatory synapsees
        for n in range(0, N_excit_neurons - 1):

            # Update incoming spikes as I_in
            I_in = (
                np.dot(
                    W_se[t - 1, :, n],
                    spikes[t - 1, :N_input_neurons],  # This might be wrong
                )
                + np.dot(
                    W_ee[t - 1, :, n],
                    spikes[
                        t - 1,
                        N_input_neurons : N_input_neurons + N_excit_neurons,
                    ],
                )
                + np.dot(
                    W_ie[t - 1, :, n],
                    spikes[t - 1, N_input_neurons + N_excit_neurons :],
                )
            )
            I_in_sum.append(I_in)

            # Update membrane potential based on I_in
            MemPot[t, n] = (
                MemPot[t - 1, n] + (-(MemPot[t - 1, n] - V_rest) + R * I_in) / tau_m
            )

            # Get nonzero weights from input neurons to current excitatory neuron
            nonzero_se_ws = np.nonzero(W_se[t - 1, :, n])[0]

            # Update pre_synaptic_trace for W_ee
            for w in range(len(nonzero_se_ws)):
                if spikes[t - 1, nonzero_se_ws[w]] == 1:
                    pre_synaptic_trace[nonzero_se_ws[w]] += 1

            # Get nonzero weights from excitatory neurons to current excitatory neuron
            nonzero_ee_ws = np.nonzero(W_ee[t - 1, :, n])[0]

            # Update pre_synaptic_trace for W_ee
            for w in range(len(nonzero_ee_ws)):
                if spikes[t - 1, N_input_neurons + nonzero_ee_ws[w]] == 1:
                    pre_synaptic_trace[N_input_neurons + nonzero_ee_ws[w]] += 1

            # Get non-zero weights from inhibitory neurons to current excitatory neuron
            nonzero_ie_ws = np.nonzero(W_ie[t - 1, :, n])[0]

            # Update pre_synaptic_trace for W_ie
            for w in range(len(nonzero_ie_ws)):
                if (
                    spikes[t - 1, N_input_neurons + N_excit_neurons + nonzero_ie_ws[w]]
                    == 1
                ):
                    pre_synaptic_trace[N_excit_neurons + nonzero_ie_ws[w]] += 1

            # Update spikes
            if MemPot[t, n] > V_th:
                spikes[t, n + N_input_neurons] = 1
                post_synaptic_trace[n] += 1
                MemPot[t, n] = V_reset
            else:
                spikes[t, n + N_input_neurons] = 0

            # Get all pre-synaptic indices
            pre_syn_indices = nonzero_se_ws

            # Check if pre_syn_indices is an empty list
            if pre_syn_indices.size != 0:

                # Loop through each synapse to update strength
                for s in range(len(pre_syn_indices)):

                    # Update ideal weight
                    W_se_ideal[pre_syn_indices[s], n] = tau_const * (
                        W_se[t - 1, pre_syn_indices[s], n]
                        - W_se_ideal[pre_syn_indices[s], n]
                        - P
                        * W_se_ideal[pre_syn_indices[s], n]
                        * ((w_p / 2) - W_se_ideal[pre_syn_indices[s], n])
                        * (w_p - W_se_ideal[pre_syn_indices[s], n])
                    )

                    # Use the current trace values for STDP calculation
                    pre_trace = pre_synaptic_trace[pre_syn_indices[s]]
                    post_trace = post_synaptic_trace[n]

                    # Get learning components
                    hebb = A * pre_trace * post_trace**2 - B * pre_trace * post_trace
                    hetero_syn = (
                        -beta
                        * (
                            W_se[t, pre_syn_indices[s], n]
                            - W_se_ideal[pre_syn_indices[s], n]
                        )
                        * post_trace**4
                    )
                    dopamine_reg = delta * pre_trace

                    # Assemble components to update weight
                    W_se[t, pre_syn_indices[s], n] = round(hebb + dopamine_reg)

            # Get all pre-synaptic indices
            pre_syn_indices = nonzero_ee_ws

            # Check if W_ee have any nonzero weights
            if pre_syn_indices.size != 0:
                # Loop through each synapse to update strength
                for s in range(len(pre_syn_indices)):
                    if pre_syn_indices[s] == n:
                        raise UserWarning(
                            "There are self-connections within the W_ee array"
                        )

                    W_ee_ideal[pre_syn_indices[s], n] = tau_const * (
                        W_ee[t - 1, pre_syn_indices[s], n]
                        - W_ee_ideal[pre_syn_indices[s], n]
                        - P
                        * W_ee_ideal[pre_syn_indices[s], n]
                        * ((w_p / 2) - W_ee_ideal[pre_syn_indices[s], n])
                        * (w_p - W_ee_ideal[pre_syn_indices[s], n])
                    )

                    # Use the current trace values for STDP calculation
                    pre_trace = pre_synaptic_trace[N_excit_neurons + pre_syn_indices[s]]
                    post_trace = post_synaptic_trace[n]

                    # Get learning components
                    hebb = A * pre_trace * post_trace**2 - B * pre_trace * post_trace
                    hetero_syn = (
                        -beta
                        * (
                            W_ee[t, pre_syn_indices[s], n]
                            - W_ee_ideal[pre_syn_indices[s], n]
                        )
                        * post_trace**4
                    )
                    dopamine_reg = delta * pre_trace
                    if n == 0:
                        print(
                            "hebb: ",
                            hebb,
                            "hetero_syn: ",
                            hetero_syn,
                            "dopamine_reg: ",
                            dopamine_reg,
                            "W_ee: ",
                            W_ee[t, pre_syn_indices[s], n],
                        )
                        print(hebb + hetero_syn + dopamine_reg)

                    # Assemble components to update weight
                    W_ee[t, pre_syn_indices[s], n] = round(hebb + dopamine_reg, 4)

        # Update inhibitory-exitatory weights
        for n in range(0, N_excit_neurons - 1):

            # Update incoming spikes as I_in
            I_in = np.dot(
                W_ie[t, :, n],
                spikes[t, N_input_neurons + N_excit_neurons :],
            )
            # Update membrane potential based on I_in
            MemPot[t, n] = (
                MemPot[t - 1, n] + (-MemPot[t - 1, n] + V_rest + R * I_in) / tau_m
            )

            # Update spikes
            if MemPot[t, n] > V_th:
                spikes[t, n + N_input_neurons] = 1
                post_synaptic_trace[n] += 1
                MemPot[t, n] = V_reset
            else:
                spikes[t, n + N_input_neurons] = 0

            # Update pre_synaptic_trace for W_ee
            for w in range(len(nonzero_ie_ws)):
                if (
                    spikes[t - 1, N_input_neurons + N_excit_neurons + nonzero_ie_ws[w]]
                    == 1
                ):
                    pre_synaptic_trace[
                        N_input_neurons + N_excit_neurons + nonzero_ie_ws[w]
                    ] += 1

        # Update excitatory-inhibitory weights
        for n in range(0, N_inhib_neurons - 1):

            # Update incoming spikes as I_in
            I_in = np.dot(
                W_ei[t, :, n],
                spikes[t, N_input_neurons : N_input_neurons + N_excit_neurons],
            )
            # Update membrane potential based on I_in
            MemPot[t, n + N_excit_neurons] = (
                MemPot[t - 1, n + N_excit_neurons]
                + (-MemPot[t - 1, n + N_excit_neurons] + V_rest + R * I_in) / tau_m
            )

            # Update spikes
            if MemPot[t, n] > V_th:
                spikes[t, n + N_input_neurons + N_excit_neurons] = 1
                post_synaptic_trace[n + N_excit_neurons + N_inhib_neurons] += 1
                MemPot[t, n + N_excit_neurons] = V_reset
            else:
                spikes[t, n + N_input_neurons + N_excit_neurons] = 0

            # Get nonzero weights from excitatory neurons to current excitatory neuron
            nonzero_ei_ws = np.nonzero(W_ei[t - 1, :, n])[0]

            # Update pre_synaptic_trace for W_ee
            for w in range(len(nonzero_ei_ws)):
                if spikes[t - 1, N_input_neurons + nonzero_ei_ws[w]] == 1:
                    pre_synaptic_trace[N_input_neurons + nonzero_ee_ws[w]] += 1

        if t % update_frequency == 0 and callback is not None:
            callback(spikes, W_se, W_ee, t)

        # Clip weights to avoid runaway effects
        if t % 10 == 0:
            W_se = np.clip(W_se, min_weight, max_weight)
            W_ee = np.clip(W_ee, min_weight, max_weight)
            W_ei = np.clip(W_ei, min_weight, max_weight)
            W_ie = np.clip(W_ie, min_weight, max_weight)
            W_se_ideal = np.clip(W_se_ideal, min_weight, max_weight)
            W_ee_ideal = np.clip(W_ee_ideal, min_weight, max_weight)
            W_ei_ideal = np.clip(W_ei_ideal, min_weight, max_weight)
            W_ie_ideal = np.clip(W_ie_ideal, min_weight, max_weight)

    return (
        spikes,
        MemPot,
        W_se,
        W_se_ideal,
        W_ee,
        W_ee_ideal,
        W_ei,
        W_ei_ideal,
        W_ie,
        W_ie_ideal,
        pre_synaptic_trace,
        post_synaptic_trace,
        I_in_sum,
    )
