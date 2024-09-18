# Train network script
import numpy as np
from tqdm import tqdm
from numba import njit
import os
import sys

# Set the current directory based on the existence of a specific path
if os.path.exists(
    "C:\\Users\\Bruker\\OneDrive\\Documents\\NMBU_\\BONSAI\\SNN\\SpikingNeuralNetwork\\src"
):
    base_path = "C:\\Users\\Bruker\\OneDrive\\Documents\\NMBU_\\BONSAI\\SNN\\SpikingNeuralNetwork\\src"
else:
    base_path = "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\src"

os.chdir(base_path)
sys.path.append(os.path.join(base_path, "main"))
sys.path.append(os.path.join(base_path, "plot"))
sys.path.append(os.path.join(base_path, "tool"))
sys.path.append(os.path.join(base_path, "train_packages"))

from plot_training import *
from plot_network import *
from membrane_potential import update_membrane_potential_conduct
from weight_updating import exc_weight_update, inh_weight_update


def train_model(
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
    tau_cons: float | int,
    tau_H: float | int,
    tau_istdp: float | int,
    tau_ampa: float | int,
    tau_nmda: float | int,
    tau_gaba: float | int,
    tau_thr: float | int,
    tau_d: float | int,
    tau_f: float | int,
    tau_a: float | int,
    tau_b: float | int,
    delta_a: float | int,
    delta_b: float | int,
    U_exc: float | int,
    U_inh: float | int,
    learning_rate: float | int,
    training_data: np.ndarray,
    N_excit_neurons: int,
    N_inhib_neurons: int,
    N_input_neurons: int,
    W_static: np.ndarray,
    W_plastic: np.ndarray,
    W_plastic_ideal: np.ndarray,
    W_plastic_2d: np.ndarray,
    W_plastic_plt_idx: np.ndarray,
    gamma: float | int,  # Where is gamma used?
    alpha_exc: float | int,
    alpha_inh: float | int,
    U_cons: float | int,
    run_njit: bool,
):
    # Initiate relevant traces and variables
    num_neurons = N_excit_neurons + N_inhib_neurons + N_input_neurons
    spikes = np.zeros((time, num_neurons))
    pre_synaptic_trace = np.zeros((time, num_neurons))
    post_synaptic_trace = np.zeros((time, N_excit_neurons))
    slow_pre_synaptic_trace = np.zeros((time, num_neurons))
    C = np.full(num_neurons, A)
    z_i = np.zeros(N_excit_neurons)
    z_j = np.zeros(N_inhib_neurons)
    z_ht = np.zeros(num_neurons)
    x = np.zeros((N_input_neurons + N_excit_neurons, 1))
    u = np.zeros((N_input_neurons + N_excit_neurons, 1))
    H = 0
    V_th_ = float(V_th_)
    V_th = np.full(num_neurons - N_input_neurons, V_th_)
    V_th_array = np.zeros((100))  # for plotting
    V_th_array[0] = V_th_
    g_nmda = np.zeros((N_excit_neurons + N_inhib_neurons, 1))
    g_ampa = np.zeros((N_excit_neurons + N_inhib_neurons, 1))
    g_gaba = np.zeros((N_excit_neurons + N_inhib_neurons, 1))
    g_a = np.zeros((N_excit_neurons, 1))
    g_b = np.zeros((N_excit_neurons, 1))
    # Generate membrane potential and spikes array
    MemPot = np.zeros((time, (N_excit_neurons + N_inhib_neurons)))
    MemPot[0, :] = V_rest
    spikes = np.zeros((time, num_neurons))

    # Add input data before training for input neurons
    spikes[:, :N_input_neurons] = training_data

    # Define update frequency for adaptive threshold
    update_freq = time // 100

    # Convert functions if njit is true
    njit_ = run_njit

    if njit_:
        update_membrane_potential_conduct_func = njit(update_membrane_potential_conduct)
        exc_weight_update_func = njit(exc_weight_update)
        inh_weight_update_func = njit(inh_weight_update)
        print("Running njit")
    else:
        update_membrane_potential_conduct_func = update_membrane_potential_conduct
        exc_weight_update_func = exc_weight_update
        inh_weight_update_func = inh_weight_update
        print("Running without njit")

    # Loop through time and update membrane potential, spikes, and weights
    for t in tqdm(range(1, time), desc="Training network"):

        # Calculate Euler time unit
        euler_unit = int(t - update_freq > 0) * (t - update_freq)

        # Update membrane potential
        MemPot[t], V_th, g_ampa, g_nmda, g_gaba, x, u, g_a, g_b = (
            update_membrane_potential_conduct_func(
                MemPot[t - 1],
                U_inh,
                U_exc,
                V_th,
                V_th_,
                W_plastic,
                W_static,
                spikes[t - 1],
                u,
                x,
                dt,
                N_input_neurons,
                N_inhib_neurons,
                V_rest,
                tau_m,
                alpha_exc,
                alpha_inh,
                tau_ampa,
                tau_nmda,
                tau_gaba,
                tau_thr,
                tau_d,
                tau_f,
                tau_a,
                tau_b,
                delta_a,
                delta_b,
                g_ampa,
                g_nmda,
                g_gaba,
                g_a,
                g_b,
                U_cons,
            )
        )

        # Update spikes based on membrane potential
        spike_mask = MemPot[t] > V_th
        spikes[t, N_input_neurons:] = spike_mask.astype(int)
        MemPot[t][spike_mask] = V_reset

        # Update excitatory weights
        (
            W_plastic[:N_input_neurons],
            W_plastic[N_input_neurons:-N_inhib_neurons],
            W_plastic_ideal[:N_input_neurons],
            W_plastic_ideal[N_input_neurons:],
            pre_synaptic_trace[t, :N_input_neurons],
            post_synaptic_trace[t],
            slow_pre_synaptic_trace[t, :N_input_neurons],
            z_ht[:N_input_neurons],
            C[:N_input_neurons],
            pre_synaptic_trace[t, N_input_neurons:-N_inhib_neurons],
            slow_pre_synaptic_trace[t, N_input_neurons:-N_inhib_neurons],
            z_ht[N_input_neurons:-N_inhib_neurons],
            C[N_input_neurons:-N_inhib_neurons],
        ) = exc_weight_update_func(
            dt,
            tau_cons,
            W_plastic[:N_input_neurons],
            W_plastic[N_input_neurons:-N_inhib_neurons],
            W_plastic_ideal[:N_input_neurons],
            W_plastic_ideal[N_input_neurons:],
            P,
            w_p,
            spikes[t - 1],
            N_input_neurons,
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
            z_ht,
            C,
        )

        # Update inhibitory weights
        (
            W_plastic[-N_inhib_neurons:],
            z_i,
            z_j,
            H,
        ) = inh_weight_update_func(
            H,
            dt,
            W_plastic[-N_inhib_neurons:],
            z_i,
            z_j,
            tau_H,
            gamma,
            tau_istdp,
            learning_rate,
            spikes[t - 1, -N_inhib_neurons:],
            spikes[t - 1, N_input_neurons:-N_inhib_neurons],
        )

        # Assign the selected indices to the first ro
        if t % update_freq == 0:
            W_plastic_2d[t] = W_plastic[
                W_plastic_plt_idx[:, 0], W_plastic_plt_idx[:, 1]
            ]

    return (
        W_plastic_2d,
        spikes,
        MemPot,
        pre_synaptic_trace,
        post_synaptic_trace,
        slow_pre_synaptic_trace,
        C,
        z_ht,
        x,
        u,
        H,
        z_i,
        z_j,
        V_th_array,
        W_plastic,
        W_static,
        V_th_,
        g_nmda,
        g_ampa,
        g_gaba,
        g_a,
        g_b,
    )
