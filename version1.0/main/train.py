# Train network script
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
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
from plot_network import *


def train_data(
    R: float | int,
    A: float | int,
    P: float | int,
    w_p: float | int,
    beta: float | int,
    delta: float | int,
    time: int,
    V_th: int,
    V_rest: int,
    V_reset: int,
    dt: float | int,
    tau_plus: float | int,
    tau_minus: float | int,
    tau_slow: float | int,
    tau_m: float | int,
    tau_ht: float | int,
    tau_hom: float | int,
    tau_const: float | int,
    tau_H: float | int,
    tau_stdp: float | int,
    learning_rate: float | int,
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
    gamma: float | int,
    save_model: bool,
):
    num_neurons = N_excit_neurons + N_inhib_neurons + N_input_neurons
    spikes = np.zeros((time, num_neurons))
    pre_synaptic_trace = np.zeros((time, num_neurons))
    post_synaptic_trace = np.zeros((time, num_neurons - N_input_neurons))
    slow_pre_synaptic_trace = np.zeros((time, num_neurons))
    C = np.full(num_neurons, A)
    z_ht = np.ones((num_neurons))
    z_istdp = np.zeros((N_excit_neurons))
    H = 0
    B = np.full(num_neurons - N_inhib_neurons, A)

    # Add input data before training for input neurons
    spikes[:, :N_input_neurons] = training_data

    # Loop through time and update membrane potential, spikes and weights
    for t in tqdm(range(1, time), desc="Training network"):
        I_in_sum = []

        # Update decay traces
        pre_synaptic_trace[t] *= 1 - (dt / tau_plus)
        post_synaptic_trace[t] *= 1 - (dt / tau_minus)
        slow_pre_synaptic_trace[t] *= 1 - (dt / tau_slow)
        z_ht *= 1 - (dt / tau_ht)
        C *= 1 - (dt / tau_hom)

        # Update stimulation-excitation, excitation-excitation and inhibitory-excitatory synapsees
        for n in range(0, N_excit_neurons):

            # Update spikes x weights and sum in I_in
            I_in = round(
                np.dot(
                    W_se[t - 1, :, n],
                    spikes[t - 1, :N_input_neurons],
                )
                + np.dot(
                    W_ee[t - 1, :, n],
                    spikes[
                        t - 1,
                        N_input_neurons : N_input_neurons + N_excit_neurons,
                    ],
                )
                - np.dot(
                    W_ie[t - 1, :, n],
                    spikes[t - 1, N_input_neurons + N_excit_neurons :],
                ),
                5,
            )

            I_in_sum.append(I_in)

            # Update membrane potential based on I_in
            delta_MemPot = round(
                (-((MemPot[t - 1, n] - V_rest) + R * I_in) / tau_m) * dt, 4
            )
            nu_mem = MemPot[t - 1, n] - delta_MemPot
            MemPot[t, n] = nu_mem

            # Get nonzero weights from input neurons to current excitatory neuron
            nonzero_se_ws = np.nonzero(W_se[t - 1, :, n])[0]

            # Update pre_synaptic_trace for W_se
            for w in range(len(nonzero_se_ws)):
                if spikes[t - 1, nonzero_se_ws[w]] == 1:
                    pre_synaptic_trace[t, nonzero_se_ws[w]] += dt
                    slow_pre_synaptic_trace[t, nonzero_se_ws[w]] += dt

            # Get nonzero weights from excitatory neurons to current excitatory neuron
            nonzero_ee_ws = np.nonzero(W_ee[t - 1, :, n])[0]

            # Update pre_synaptic_trace for W_ee
            for w in range(len(nonzero_ee_ws)):
                if spikes[t - 1, N_input_neurons + nonzero_ee_ws[w]] == 1:
                    pre_synaptic_trace[t, N_input_neurons + nonzero_ee_ws[w]] += dt
                    slow_pre_synaptic_trace[t, N_input_neurons + nonzero_ee_ws[w]] += dt

            # Get non-zero weights from inhibitory neurons to current excitatory neuron
            nonzero_ie_ws = np.nonzero(W_ie[t - 1, :, n])[0]

            # Update pre_synaptic_trace for W_ie
            for w in range(len(nonzero_ie_ws)):
                if (
                    spikes[t - 1, N_input_neurons + N_excit_neurons + nonzero_ie_ws[w]]
                    == 1
                ):
                    pre_synaptic_trace[
                        t, N_input_neurons + N_excit_neurons + nonzero_ie_ws[w]
                    ] += dt
                    slow_pre_synaptic_trace[
                        t, N_input_neurons + N_excit_neurons + nonzero_ie_ws[w]
                    ] += dt

            # Update spikes
            if MemPot[t, n] > V_th:
                spikes[t, n + N_input_neurons] = 1
                post_synaptic_trace[t, n] += dt
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
                    W_se_ideal[pre_syn_indices[s], n] += (
                        dt
                        * tau_const
                        * (
                            W_se[t - 1, pre_syn_indices[s], n]
                            - W_se_ideal[pre_syn_indices[s], n]
                            - P
                            * W_se_ideal[pre_syn_indices[s], n]
                            * ((w_p / 2) - W_se_ideal[pre_syn_indices[s], n])
                            * (w_p - W_se_ideal[pre_syn_indices[s], n])
                        )
                    )

                    # Use the current trace values for STDP calculation
                    pre_trace = round(pre_synaptic_trace[t, pre_syn_indices[s]], 4)
                    post_trace = round(post_synaptic_trace[t, n], 4)
                    slow_trace = round(
                        slow_pre_synaptic_trace[t, pre_syn_indices[s]], 4
                    )

                    # Update z_ht, C and B
                    if spikes[t, pre_syn_indices[s]] == 1:
                        z_ht[pre_syn_indices[s]] += dt
                        C[pre_syn_indices[s]] += (z_ht[pre_syn_indices[s]]) ** 2

                    if A * C[pre_syn_indices[s]] <= 1:
                        B[pre_syn_indices[s]] = C[pre_syn_indices[s]]
                    else:
                        B[pre_syn_indices[s]] = A

                    # Get learning components
                    triplet_LTP = (
                        A * pre_trace * slow_trace * spikes[t, pre_syn_indices[s]]
                    )
                    doublet_LTD = (
                        B[pre_syn_indices[s]]
                        * post_trace
                        * slow_trace
                        * spikes[t, N_input_neurons + n]
                    )

                    Hebb = triplet_LTP - doublet_LTD
                    Hetero = (
                        -beta
                        * (
                            W_se[t - 1, pre_syn_indices[s], n]
                            - W_se_ideal[pre_syn_indices[s], n]
                        )
                        * (post_trace) ** 3
                        * spikes[t, pre_syn_indices[s]]
                    )
                    transmitter = delta * spikes[t, N_input_neurons + n]
                    # Assemble components to update weight
                    delta_w = (Hebb + Hetero + transmitter) * dt
                    if delta_w > 0.0001:
                        print(
                            "delta_w",
                            delta_w,
                            "Hebb",
                            Hebb,
                            "Hetero",
                            Hetero,
                            "transmitter",
                            transmitter,
                            "A",
                            A,
                            "B",
                            B[n],
                            "C",
                            C[pre_syn_indices[s]],
                            "z_ht",
                            z_ht[pre_syn_indices[s]],
                        )

                    W_se[t, pre_syn_indices[s], n] = (
                        W_se[t - 1, pre_syn_indices[s], n] + delta_w
                    )

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

                    W_ee_ideal[pre_syn_indices[s], n] = (
                        dt
                        * tau_const
                        * (
                            W_ee[t - 1, pre_syn_indices[s], n]
                            - W_ee_ideal[pre_syn_indices[s], n]
                            - P
                            * W_ee_ideal[pre_syn_indices[s], n]
                            * ((w_p / 2) - W_ee_ideal[pre_syn_indices[s], n])
                            * (w_p - W_ee_ideal[pre_syn_indices[s], n])
                        )
                    )

                    # Use the current trace values for STDP calculation
                    pre_trace = pre_synaptic_trace[
                        t, N_input_neurons + pre_syn_indices[s]
                    ]
                    post_trace = post_synaptic_trace[t, n]
                    slow_trace = slow_pre_synaptic_trace[
                        t, N_input_neurons + pre_syn_indices[s]
                    ]

                    # Update z_ht, C and B
                    if spikes[t, N_input_neurons + pre_syn_indices[s]] == 1:
                        z_ht[N_input_neurons + pre_syn_indices[s]] += dt

                        C[N_input_neurons + pre_syn_indices[s]] += (
                            z_ht[N_input_neurons + pre_syn_indices[s]] ** 2
                        )

                    if A * C[N_input_neurons + pre_syn_indices[s]] <= 1:
                        B[N_input_neurons + pre_syn_indices[s]] = C[
                            N_input_neurons + pre_syn_indices[s]
                        ]
                    else:
                        B[N_input_neurons + pre_syn_indices[s]] = A

                    # Get learning components
                    triplet_LTP = (
                        A
                        * post_trace
                        * slow_trace  # presynaptic slow trace
                        * spikes[t, N_input_neurons + pre_syn_indices[s]]
                    )
                    doublet_LTD = (
                        B[N_input_neurons + pre_syn_indices[s]]
                        * post_trace
                        * slow_trace
                        * spikes[t, N_input_neurons + n]
                    )

                    Hebb = triplet_LTP - doublet_LTD
                    Hetero = (
                        -beta
                        * (
                            W_ee[t - 1, pre_syn_indices[s], n]
                            - W_ee_ideal[pre_syn_indices[s], n]
                        )
                        * (post_trace) ** 3
                        * spikes[t, N_input_neurons + pre_syn_indices[s]]
                    )
                    transmitter = delta * spikes[t, N_input_neurons + n]
                    delta_w = (Hebb + Hetero + transmitter) * dt

                    # Assemble components to update weight
                    W_ee[t, pre_syn_indices[s], n] = (
                        W_ee[t - 1, pre_syn_indices[s], n] + delta_w
                    )

            # Update weights based on ISP-function from Zenke between inhibitory and excitatory neurons
            if nonzero_ie_ws.size != 0:
                # Update H
                H += (
                    -(H / tau_H) + sum(spikes[t, N_input_neurons + N_excit_neurons :])
                ) * dt
                G = H - gamma
                z_istdp[n] += (
                    -z_istdp[n] / tau_stdp
                    + sum(spikes[t, N_input_neurons + N_excit_neurons :])
                ) * dt

                for id, s in enumerate(nonzero_ie_ws):

                    # Get traces
                    pre_trace = pre_synaptic_trace[
                        t, N_input_neurons + N_excit_neurons + s
                    ]
                    post_trace = post_synaptic_trace[t, n]

                    # Calculate delta weights
                    delta_w = (
                        learning_rate
                        * G
                        * (z_istdp[n] + 1)
                        * spikes[t, N_input_neurons + N_excit_neurons + s]
                        + post_trace * spikes[t, n]
                    )

                    # Update weights
                    W_ie[t, s, n] = W_ie[t - 1, s, n] + delta_w

        # Update excitatory-inhibitory mempot and spikes
        for n in range(0, N_inhib_neurons):

            # Update incoming spikes as I_in
            I_in = np.dot(
                W_ei[t - 1, :, n],
                spikes[t - 1, N_input_neurons : N_input_neurons + N_excit_neurons],
            )

            I_in_sum.append(I_in)
            # Update membrane potential based on I_in
            delta_MemPot = (
                -((MemPot[t - 1, n + N_excit_neurons] - V_rest) + R * I_in) / tau_m
            ) * dt
            MemPot[t, n + N_excit_neurons] = MemPot[t - 1, n + N_excit_neurons] - round(
                delta_MemPot, 4
            )

            # Update spikes
            if MemPot[t, n + N_excit_neurons] > V_th:
                spikes[t, n + N_input_neurons + N_excit_neurons] = 1
                post_synaptic_trace[t, n + N_excit_neurons] += dt
                slow_pre_synaptic_trace[t, n + N_excit_neurons] += dt
                MemPot[t, n + N_excit_neurons] = V_reset
            else:
                spikes[t, n + N_input_neurons + N_excit_neurons] = 0

        # Ensure weights continue their value to the next time step
        W_ei[t] = W_ei[t - 1]
        W_ie[t] = W_ie[t - 1]

        # Ensure spike traces continue their value to the next time step
        if t < time - 1:
            pre_synaptic_trace[t + 1] = pre_synaptic_trace[t]
            post_synaptic_trace[t + 1] = post_synaptic_trace[t]
            slow_pre_synaptic_trace[t + 1] = slow_pre_synaptic_trace[t]

    # print weights heatmapped
    draw_weights_layer(W_ei[-1], "W_ei", "Inhibitory Neurons", "Excitatory Neurons")
    draw_weights_layer(W_ie[-1], "W_ie", "Excitatory Neurons", "Inhibitory Neurons")
    draw_weights_layer(W_se[-1], "W_se", "Input Neurons", "Excitatory Neurons")
    draw_weights_layer(W_ee[-1], "W_ee", "Excitatory Neurons", "Excitatory Neurons")

    if save_model:
        # Create folder to save model
        if not os.path.exists("model"):
            os.makedirs("model")

        # Save model weights, spikes and MemPot
        np.save("model/W_se.npy", W_se)
        np.save("model/W_ee.npy", W_ee)
        np.save("model/W_ei.npy", W_ei)
        np.save("model/W_ie.npy", W_ie)
        np.save("model/spikes.npy", spikes)
        np.save("model/MemPot.npy", MemPot)
        np.save("model/pre_synaptic_trace.npy", pre_synaptic_trace)
        np.save("model/post_synaptic_trace.npy", post_synaptic_trace)
        np.save("model/slow_pre_synaptic_trace.npy", slow_pre_synaptic_trace)

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
        slow_pre_synaptic_trace,
        I_in_sum,
    )
