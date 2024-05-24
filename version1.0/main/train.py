# Train network script
import numpy as np
from tqdm import tqdm
from numba import njit
import os
import sys

# Set the current directory based on the existence of a specific path
if os.path.exists(
    "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.0"
):
    base_path = "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.0"
else:
    base_path = "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\version1.0"

os.chdir(base_path)
sys.path.append(os.path.join(base_path, "plot"))
sys.path.append(os.path.join(base_path, "tool"))
sys.path.append(os.path.join(base_path, "train_packages"))

from plot_training import *
from plot_network import *
from membrane_potential import (
    adjust_membrane_threshold,
    update_membrane_potential,
)
from weight_updating import exc_weight_update, inh_weight_update


def train_data(
    R: float | int,
    A: float | int,
    P: float | int,
    w_p: float | int,
    beta: float | int,
    delta: float | int,
    time: int,
    V_th_: int,
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
    item_lim: int,
    items: int,
):
    # Initiate relevant arrays and variables
    num_neurons = N_excit_neurons + N_inhib_neurons + N_input_neurons
    spikes = np.zeros((time, num_neurons))
    pre_synaptic_trace = np.zeros((time, num_neurons))
    post_synaptic_trace = np.zeros((time, num_neurons - N_input_neurons))
    slow_pre_synaptic_trace = np.zeros((time, num_neurons))
    C = np.full(num_neurons, A)
    z_ht = np.ones((num_neurons))
    z_istdp = np.zeros((N_inhib_neurons))
    H = 0
    V_th_ = float(V_th_)
    V_th = np.full(num_neurons - N_input_neurons, V_th_)

    # Add input data before training for input neurons
    spikes[:, :N_input_neurons] = training_data

    # Define update frequency for adaptive threshold
    update_freq = time // 100

    # Convert functions if njit is true
    njit_ = int(items > item_lim)
    if njit_:
        adjust_membrane_threshold_func = njit(adjust_membrane_threshold)
        update_membrane_potential_func = njit(update_membrane_potential)
        exc_weight_update_func = njit(exc_weight_update)
        inh_weight_update_func = njit(inh_weight_update)
        print("running njit")
    else:
        adjust_membrane_threshold_func = adjust_membrane_threshold
        update_membrane_potential_func = update_membrane_potential
        exc_weight_update_func = exc_weight_update
        inh_weight_update_func = inh_weight_update
        print("running without njit")

    # Loop through time and update membrane potential, spikes and weights => infinite knowledge
    for t in tqdm(range(1, time), desc="Training network"):

        # Calculate euler time unit
        euler_unit = int(t - update_freq > 0) * (t - update_freq)

        # Update adaptive membrane potential threshold
        if t % update_freq == 0:
            V_th = adjust_membrane_threshold_func(
                spikes[t - 1],
                V_th,
                V_reset,
                N_input_neurons,
                N_excit_neurons,
                N_inhib_neurons,
            )

        # Update membrane potential
        MemPot[t] = update_membrane_potential_func(
            MemPot[t - 1],
            W_se[t - 1],
            W_ee[t - 1],
            W_ie[t - 1],
            W_ei[t - 1],
            spikes[t - 1],
            dt,
            N_excit_neurons,
            N_input_neurons,
            N_inhib_neurons,
            V_rest,
            R,
            tau_m,
        )

        # Update spikes based on membrane potential
        spike_mask = MemPot[t] > V_th
        spikes[t, N_input_neurons:] = spike_mask.astype(int)
        MemPot[t][spike_mask] = V_reset
        V_th[spike_mask] = V_th_

        # Update excitatory weights
        (
            W_se[t],
            W_ee[t],
            W_se_ideal,
            W_ee_ideal,
            pre_synaptic_trace[t],
            post_synaptic_trace[t],
            slow_pre_synaptic_trace[t],
            z_ht,
            C,
        ) = exc_weight_update_func(
            dt,
            tau_const,
            W_se[t - 1],
            W_ee[t - 1],
            W_se_ideal,
            W_ee_ideal,
            P,
            t,
            w_p,
            spikes[t - 1],
            N_input_neurons,
            N_excit_neurons,
            N_inhib_neurons,
            pre_synaptic_trace[t - 1],
            post_synaptic_trace[t - 1],
            slow_pre_synaptic_trace[euler_unit],
            tau_plus,
            tau_minus,
            tau_slow,
            tau_ht,
            tau_hom,
            A,
            beta,
            delta,
            update_freq,
            z_ht,
            C,
        )

        # Update inhibitory weights
        W_ie[t], z_istdp, H, post_synaptic_trace[t, :-N_inhib_neurons] = (
            inh_weight_update_func(
                H,
                dt,
                W_ie[t - 1],
                z_istdp,
                tau_H,
                gamma,
                tau_stdp,
                learning_rate,
                spikes[t - 1],
                N_input_neurons,
                N_inhib_neurons,
                post_synaptic_trace[t - 1],
            )
        )

        # Ensure weights continue their value to the next time step
        W_ei[t] = W_ei[t - 1]

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
        np.save("model/z_istdp_trace.npy", z_istdp)

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
        z_istdp,
    )
