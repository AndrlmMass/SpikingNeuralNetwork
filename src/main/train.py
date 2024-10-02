# Import external libraries
from numba import njit
import numpy as np
from tqdm import tqdm

# Import internal libraries
from plot.plot_training import *
from plot.plot_network import *
from train_packages.membrane_potential import update_membrane_potential_conduct
from train_packages.weight_updating import exc_weight_update, inh_weight_update


def train_model(
    A: int | float,  # static
    P: int | float,  # static
    w_p: int | float,  # static
    beta: int | float,  # static
    delta: int | float,  # static
    euler: int,  # static
    time: int,  # static
    V_th_: int,  # static
    V_rest: int,  # static
    dt: int | float,  # static
    tau_plus: int | float,  # static
    tau_minus: int | float,  # static
    tau_slow: int | float,  # static
    tau_m: int | float,  # static
    tau_ht: int | float,  # static
    tau_hom: int | float,  # static
    tau_cons: int | float,  # static
    tau_H: int | float,  # static
    tau_istdp: int | float,  # static
    tau_ampa: int | float,  # static
    tau_nmda: int | float,  # static
    tau_gaba: int | float,  # static
    tau_thr: int | float,  # static
    tau_d: int | float,  # static
    tau_f: int | float,  # static
    tau_a: int | float,  # static
    tau_b: int | float,  # static
    delta_a: int | float,  # static
    delta_b: int | float,  # static
    U_exc: int,  # static
    U_inh: int,  # static
    learning_rate: int | float,  # static
    training_data: np.ndarray,  # static
    N_excit_neurons: int,  # static
    N_inhib_neurons: int,  # static
    N_input_neurons: int,  # static
    W_static: np.ndarray,  # static
    W_plastic: np.ndarray,  # plastic
    W_plastic_ideal: np.ndarray,  # plastic
    W_plastic_plt: np.ndarray,  # plastic
    gamma: int | float,  # static
    alpha_exc: int | float,  # static
    alpha_inh: int | float,  # static
    U_cons: int | float,  # static
    th_rest: int,  # static
    th_refact: int,  # static
    run_njit: bool,  # static
):
    # Initiate relevant traces and variables
    num_neurons = N_excit_neurons + N_inhib_neurons + N_input_neurons
    spikes = np.zeros((time, num_neurons))
    pre_trace = np.zeros((time, N_input_neurons + N_excit_neurons))
    post_trace = np.zeros((time, N_excit_neurons))
    slow_trace = np.zeros((time, N_excit_neurons))
    C = np.full(N_excit_neurons, A)
    z_i = np.zeros(N_excit_neurons)
    z_j = np.zeros(N_inhib_neurons)
    z_ht = np.zeros(N_excit_neurons)
    x = np.zeros((N_input_neurons + N_excit_neurons, 1))
    u = np.zeros((N_input_neurons + N_excit_neurons, 1))
    H = 0.0
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
    MemPot[0] = V_rest
    spikes[:, :N_input_neurons] = training_data

    # Define update frequency for adaptive threshold for every percent in update
    update_freq = time // 100

    # update functions based on njit arg
    if run_njit:
        update_membrane_potential_conduct_ = njit(update_membrane_potential_conduct)
        exc_weight_update_ = njit(exc_weight_update)
        inh_weight_update_ = njit(inh_weight_update)
    else:
        update_membrane_potential_conduct_ = update_membrane_potential_conduct
        exc_weight_update_ = exc_weight_update
        inh_weight_update_ = inh_weight_update

    # Loop through time and update membrane potential, spikes, and weights
    for t in tqdm(range(1, time), desc="Training network"):
        # Update membrane potential
        MemPot[t], g_ampa, g_nmda, g_gaba, x, u, g_a, g_b = (
            update_membrane_potential_conduct_(
                MemPot[t - 1],
                U_inh,
                U_exc,
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
        MemPot[t][spike_mask] = V_rest

        # Update spiking threshold based on who has spiked
        V_th = V_th + dt / tau_thr * (th_rest - V_th)
        V_th[spike_mask] = th_refact

        # Update excitatory weights
        (
            W_plastic[:N_input_neurons],
            W_plastic[N_input_neurons:-N_inhib_neurons],
            W_plastic_ideal[:N_input_neurons],
            W_plastic_ideal[N_input_neurons:],
            pre_trace[t, :N_input_neurons],
            post_trace[t],
            slow_trace[t],
            z_ht,
            C,
            pre_trace[t, N_input_neurons:],
        ) = exc_weight_update_(
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
            pre_trace[t - 1],
            post_trace[t - 1],
            post_trace[max(t - euler, 0)],
            slow_trace[max(t - euler, 0)],
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
        W_plastic[-N_inhib_neurons:], z_i, z_j, H = inh_weight_update_(
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

        ## Calculate the mean, high, and low weights for each plastic group ##
        if t % update_freq == 0:
            W_plastic_plt[t, 0] = np.round(np.mean(W_plastic[:N_input_neurons]), 5)
            W_plastic_plt[t, 1] = np.round(np.amax(W_plastic[:N_input_neurons]), 5)
            W_plastic_plt[t, 2] = np.round(np.amin(W_plastic[:N_input_neurons]), 5)

            W_plastic_plt[t, 3] = np.round(
                np.mean(W_plastic[N_input_neurons:-N_inhib_neurons]), 5
            )
            W_plastic_plt[t, 4] = np.round(
                np.amax(W_plastic[N_input_neurons:-N_inhib_neurons]), 5
            )
            W_plastic_plt[t, 5] = np.round(
                np.amin(W_plastic[N_input_neurons:-N_inhib_neurons]), 5
            )

            W_plastic_plt[t, 6] = np.round(np.mean(W_plastic[-N_inhib_neurons:]), 5)
            W_plastic_plt[t, 7] = np.round(np.amax(W_plastic[-N_inhib_neurons:]), 5)
            W_plastic_plt[t, 8] = np.round(np.amin(W_plastic[-N_inhib_neurons:]), 5)

    return (
        W_plastic_plt,
        spikes,
        MemPot,
        pre_trace,
        post_trace,
        slow_trace,
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
