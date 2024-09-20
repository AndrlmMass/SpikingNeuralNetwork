import jax.numpy as jnp
from tqdm import tqdm
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
    A: float,
    P: float,
    w_p: float,
    beta: float,
    delta: float,
    time: int,
    V_th_: float,
    V_rest: float,
    V_reset: float,
    dt: float,
    tau_plus: float,
    tau_minus: float,
    tau_slow: float,
    tau_m: float,
    tau_ht: float,
    tau_hom: float,
    tau_cons: float,
    tau_H: float,
    tau_istdp: float,
    tau_ampa: float,
    tau_nmda: float,
    tau_gaba: float,
    tau_thr: float,
    tau_d: float,
    tau_f: float,
    tau_a: float,
    tau_b: float,
    delta_a: float,
    delta_b: float,
    U_exc: float,
    U_inh: float,
    learning_rate: float,
    training_data: jnp.ndarray,
    N_excit_neurons: int,
    N_inhib_neurons: int,
    N_input_neurons: int,
    W_static: jnp.ndarray,
    W_plastic: jnp.ndarray,
    W_plastic_ideal: jnp.ndarray,
    W_plastic_plt: jnp.ndarray,
    gamma: float,
    alpha_exc: float,
    alpha_inh: float,
    U_cons: float,
    th_rest: float | int,
    th_refact: float | int,
):
    # Initiate relevant traces and variables
    num_neurons = N_excit_neurons + N_inhib_neurons + N_input_neurons
    spikes = jnp.zeros((time, num_neurons))
    pre_synaptic_trace = jnp.zeros((time, num_neurons))
    post_synaptic_trace = jnp.zeros((time, N_excit_neurons))
    slow_pre_synaptic_trace = jnp.zeros((time, num_neurons))
    C = jnp.full(num_neurons, A)
    z_i = jnp.zeros(N_excit_neurons)
    z_j = jnp.zeros(N_inhib_neurons)
    z_ht = jnp.zeros(num_neurons)
    x = jnp.zeros((N_input_neurons + N_excit_neurons, 1))
    u = jnp.zeros((N_input_neurons + N_excit_neurons, 1))
    H = 0.0
    V_th = jnp.full(num_neurons - N_input_neurons, V_th_)
    V_th_array = jnp.zeros((100))  # for plotting
    V_th_array = V_th_array.at[0].set(V_th_)
    g_nmda = jnp.zeros((N_excit_neurons + N_inhib_neurons, 1))
    g_ampa = jnp.zeros((N_excit_neurons + N_inhib_neurons, 1))
    g_gaba = jnp.zeros((N_excit_neurons + N_inhib_neurons, 1))
    g_a = jnp.zeros((N_excit_neurons, 1))
    g_b = jnp.zeros((N_excit_neurons, 1))
    # Generate membrane potential and spikes array
    MemPot = jnp.zeros((time, (N_excit_neurons + N_inhib_neurons)))
    MemPot = MemPot.at[0, :].set(V_rest)
    spikes = spikes.at[:, :N_input_neurons].set(training_data)

    # Define update frequency for adaptive threshold
    update_freq = time // 100

    # Loop through time and update membrane potential, spikes, and weights
    for t in tqdm(range(1, time), desc="Training network"):

        # Calculate Euler time unit
        euler_unit = int(t - update_freq > 0) * (t - update_freq)

        # Update membrane threshold
        V_th = V_th + dt / tau_thr * (V_th_ - V_th)

        # Update membrane potential
        MemPot_t, V_th_nu, g_ampa, g_nmda, g_gaba, x, u, g_a, g_b = (
            update_membrane_potential_conduct(
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
        # Add new membrane potential to array
        MemPot = MemPot.at[t].set(MemPot_t)
        V_th = V_th.at[t].set(V_th_nu)

        # Update spikes based on membrane potential
        spike_mask = MemPot_t > V_th
        spikes = spikes.at[t, N_input_neurons:].set(spike_mask.astype(int))
        MemPot_t = jnp.where(spike_mask, V_reset, MemPot_t)
        MemPot = MemPot.at[t].set(MemPot_t)

        # Update spiking threshold based on who has spiked
        V_th_upd = jnp.where(spike_mask, th_refact, V_th_nu)
        V_th = V_th.at[t].set(V_th_upd)

        # Update excitatory weights
        (
            W_se,
            W_ee,
            W_se_ideal,
            W_ee_ideal,
            pre_trace_se,
            pre_trace_ee,
            post_trace_se,
            post_trace_ee,
            slow_trace_se,
            slow_trace_se,
            z_ht_se,
            z_ht_ee,
            C_se,
            C_ee,
        ) = exc_weight_update(
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
        W_ie, z_i, z_j, H = inh_weight_update(
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

        # Assign pre-existing weight arrays and misc to new values

        # Assign the selected indices to the first row
        if t % update_freq == 0:
            # Calculate the mean, high, and low weights for each plastic group
            W_se_mean = jnp.round(jnp.mean(W_plastic[:N_input_neurons]), 5)
            W_se_high = jnp.round(jnp.amax(W_plastic[:N_input_neurons]), 5)
            W_se_low = jnp.round(jnp.amin(W_plastic[:N_input_neurons]), 5)

            W_plastic_plt = W_plastic_plt.at[t, 0].set(W_se_mean)
            W_plastic_plt = W_plastic_plt.at[t, 1].set(W_se_high)
            W_plastic_plt = W_plastic_plt.at[t, 2].set(W_se_low)

            W_ee_mean = jnp.round(
                jnp.mean(W_plastic[N_input_neurons:-N_inhib_neurons]), 5
            )
            W_ee_high = jnp.round(
                jnp.amax(W_plastic[N_input_neurons:-N_inhib_neurons]), 5
            )
            W_ee_low = jnp.round(
                jnp.amin(W_plastic[N_input_neurons:-N_inhib_neurons]), 5
            )

            W_plastic_plt = W_plastic_plt.at[t, 3].set(W_ee_mean)
            W_plastic_plt = W_plastic_plt.at[t, 4].set(W_ee_high)
            W_plastic_plt = W_plastic_plt.at[t, 5].set(W_ee_low)

            W_ie_mean = jnp.round(jnp.mean(W_plastic[-N_inhib_neurons:]), 5)
            W_ie_high = jnp.round(jnp.amax(W_plastic[-N_inhib_neurons:]), 5)
            W_ie_low = jnp.round(jnp.amin(W_plastic[-N_inhib_neurons:]), 5)

            W_plastic_plt = W_plastic_plt.at[t, 6].set(W_ie_mean)
            W_plastic_plt = W_plastic_plt.at[t, 7].set(W_ie_high)
            W_plastic_plt = W_plastic_plt.at[t, 8].set(W_ie_low)

            print(f"W_ee_mean: {W_ee_mean}\r")

    return (
        W_plastic_plt,
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
