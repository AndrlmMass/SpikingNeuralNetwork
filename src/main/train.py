# Import external libraries
import jax.numpy as jnp
from tqdm import tqdm

# Import internal libraries
from plot.plot_training import *
from plot.plot_network import *
from train_packages.membrane_potential import update_membrane_potential_conduct
from train_packages.weight_updating import exc_weight_update, inh_weight_update


def train_model(
    A: int | float,
    P: int | float,
    w_p: int | float,
    beta: int | float,
    delta: int | float,
    time: int,
    V_th_: int,
    V_rest: int,
    dt: int | float,
    tau_plus: int | float,
    tau_minus: int | float,
    tau_slow: int | float,
    tau_m: int | float,
    tau_ht: int | float,
    tau_hom: int | float,
    tau_cons: int | float,
    tau_H: int | float,
    tau_istdp: int | float,
    tau_ampa: int | float,
    tau_nmda: int | float,
    tau_gaba: int | float,
    tau_thr: int | float,
    tau_d: int | float,
    tau_f: int | float,
    tau_a: int | float,
    tau_b: int | float,
    delta_a: int | float,
    delta_b: int | float,
    U_exc: int,
    U_inh: int,
    learning_rate: int | float,
    training_data: np.ndarray,
    N_excit_neurons: int,
    N_inhib_neurons: int,
    N_input_neurons: int,
    W_static: np.ndarray,
    W_plastic: np.ndarray,
    W_plastic_ideal: np.ndarray,
    W_plastic_plt: np.ndarray,
    gamma: int | float,
    alpha_exc: int | float,
    alpha_inh: int | float,
    U_cons: int | float,
    th_rest: float,
    th_refact: float,
):
    # Define a helper function for logging
    def convert_and_log(var_name, var_value):
        try:
            converted = jnp.asarray(var_value, dtype=jnp.float16)
            print(f"Successfully converted {var_name}")
            return converted
        except Exception as e:
            print(f"Error converting {var_name}: {e}")
            raise

    # Convert scalar float parameters
    A = convert_and_log("A", A)
    P = convert_and_log("P", P)
    w_p = convert_and_log("w_p", w_p)
    beta = convert_and_log("beta", beta)
    delta = convert_and_log("delta", delta)
    V_th_ = convert_and_log("V_th_", V_th_)
    V_rest = convert_and_log("V_rest", V_rest)
    dt = convert_and_log("dt", dt)
    tau_plus = convert_and_log("tau_plus", tau_plus)
    tau_minus = convert_and_log("tau_minus", tau_minus)
    tau_slow = convert_and_log("tau_slow", tau_slow)
    tau_m = convert_and_log("tau_m", tau_m)
    tau_ht = convert_and_log("tau_ht", tau_ht)
    tau_hom = convert_and_log("tau_hom", tau_hom)
    tau_cons = convert_and_log("tau_cons", tau_cons)
    tau_H = convert_and_log("tau_H", tau_H)
    tau_istdp = convert_and_log("tau_istdp", tau_istdp)
    tau_ampa = convert_and_log("tau_ampa", tau_ampa)
    tau_nmda = convert_and_log("tau_nmda", tau_nmda)
    tau_gaba = convert_and_log("tau_gaba", tau_gaba)
    tau_thr = convert_and_log("tau_thr", tau_thr)
    tau_d = convert_and_log("tau_d", tau_d)
    tau_f = convert_and_log("tau_f", tau_f)
    tau_a = convert_and_log("tau_a", tau_a)
    tau_b = convert_and_log("tau_b", tau_b)
    delta_a = convert_and_log("delta_a", delta_a)
    delta_b = convert_and_log("delta_b", delta_b)
    U_exc = convert_and_log("U_exc", U_exc)
    U_inh = convert_and_log("U_inh", U_inh)
    learning_rate = convert_and_log("learning_rate", learning_rate)

    # Convert NumPy arrays
    training_data = convert_and_log("training_data", training_data)
    W_static = convert_and_log("W_static", W_static)
    W_plastic = convert_and_log("W_plastic", W_plastic)
    W_plastic_ideal = convert_and_log("W_plastic_ideal", W_plastic_ideal)
    W_plastic_plt = convert_and_log("W_plastic_plt", W_plastic_plt)

    # Convert additional float parameters
    gamma = convert_and_log("gamma", gamma)
    alpha_exc = convert_and_log("alpha_exc", alpha_exc)
    alpha_inh = convert_and_log("alpha_inh", alpha_inh)
    U_cons = convert_and_log("U_cons", U_cons)

    # Convert threshold parameters (float or int)
    th_rest = convert_and_log("th_rest", th_rest)
    th_refact = convert_and_log("th_refact", th_refact)

    # Initiate relevant traces and variables
    num_neurons = jnp.int16(N_excit_neurons + N_inhib_neurons + N_input_neurons)
    spikes = jnp.zeros((time, num_neurons), dtype=jnp.float16)
    pre_synaptic_trace = jnp.zeros((time, num_neurons), dtype=jnp.float16)
    post_synaptic_trace = jnp.zeros((time, N_excit_neurons), dtype=jnp.float16)
    slow_post_synaptic_trace = jnp.zeros((time, N_excit_neurons), dtype=jnp.float16)
    C = jnp.full(N_excit_neurons, A, dtype=jnp.float16)
    z_i = jnp.zeros(N_excit_neurons, dtype=jnp.float16)
    z_j = jnp.zeros(N_inhib_neurons, dtype=jnp.float16)
    z_ht = jnp.zeros(N_excit_neurons, dtype=jnp.float16)
    x = jnp.zeros((N_input_neurons + N_excit_neurons, 1), dtype=jnp.float16)
    u = jnp.zeros((N_input_neurons + N_excit_neurons, 1), dtype=jnp.float16)
    H = jnp.float16(0.0)
    V_th = jnp.full(num_neurons - N_input_neurons, V_th_, dtype=jnp.float16)
    V_th_array = jnp.zeros((100), dtype=jnp.float16)  # for plotting
    V_th_array = V_th_array.at[0].set(jnp.float16(V_th_))
    g_nmda = jnp.zeros((N_excit_neurons + N_inhib_neurons, 1), dtype=jnp.float16)
    g_ampa = jnp.zeros((N_excit_neurons + N_inhib_neurons, 1), dtype=jnp.float16)
    g_gaba = jnp.zeros((N_excit_neurons + N_inhib_neurons, 1), dtype=jnp.float16)
    g_a = jnp.zeros((N_excit_neurons, 1), dtype=jnp.float16)
    g_b = jnp.zeros((N_excit_neurons, 1), dtype=jnp.float16)
    # Generate membrane potential and spikes array
    MemPot = jnp.zeros((time, (N_excit_neurons + N_inhib_neurons)), dtype=jnp.float16)
    MemPot = MemPot.at[0, :].set(jnp.float16(V_rest))
    spikes = spikes.at[:, :N_input_neurons].set(training_data.astype(jnp.float16))

    # Define update frequency for adaptive threshold for every percent in update
    update_freq = time // 100

    # Loop through time and update membrane potential, spikes, and weights
    for t in tqdm(range(1, time), desc="Training network"):

        # Calculate Euler time unit
        euler_unit = int(t - update_freq > 0) * (t - update_freq)

        # Update membrane potential
        MemPot_t, g_ampa, g_nmda, g_gaba, x, u, g_a, g_b = (
            update_membrane_potential_conduct(
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
        spike_mask = MemPot_t > V_th
        spikes = spikes.at[t, N_input_neurons:].set(spike_mask.astype(int))
        MemPot_t = jnp.where(spike_mask, V_rest, MemPot_t)
        MemPot = MemPot.at[t].set(MemPot_t)

        # Update spiking threshold based on who has spiked
        V_th = V_th + dt / tau_thr * (th_rest - V_th)
        V_th = jnp.where(spike_mask, th_refact, V_th)

        # Update excitatory weights
        (
            W_se,
            W_ee,
            W_se_ideal,
            W_ee_ideal,
            pre_trace_se,
            post_trace,
            slow_trace,
            z_ht,
            C,
            pre_trace_ee,
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
            slow_post_synaptic_trace[euler_unit],
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

        # Update the original arrays for traces
        pre_synaptic_trace = pre_synaptic_trace.at[t, :N_input_neurons].set(
            pre_trace_se
        )
        pre_synaptic_trace = pre_synaptic_trace.at[
            t, N_input_neurons:-N_inhib_neurons
        ].set(pre_trace_ee)
        post_synaptic_trace = post_synaptic_trace.at[t].set(post_trace)
        slow_post_synaptic_trace = slow_post_synaptic_trace.at[t].set(slow_trace)

        # Update original weights and ideal weights
        W_plastic = W_plastic.at[:N_input_neurons].set(W_se)
        W_plastic = W_plastic.at[N_input_neurons:-N_inhib_neurons].set(W_ee)
        W_plastic = W_plastic.at[-N_inhib_neurons:].set(W_ie)

        W_plastic_ideal = W_plastic_ideal.at[:N_input_neurons].set(W_se_ideal)
        W_plastic_ideal = W_plastic_ideal.at[N_input_neurons:].set(W_ee_ideal)

        ## Calculate the mean, high, and low weights for each plastic group ##
        W_se_mean = jnp.round(jnp.mean(W_plastic[:N_input_neurons]), 5)
        W_se_high = jnp.round(jnp.amax(W_plastic[:N_input_neurons]), 5)
        W_se_low = jnp.round(jnp.amin(W_plastic[:N_input_neurons]), 5)

        W_plastic_plt = W_plastic_plt.at[t, 0].set(W_se_mean)
        W_plastic_plt = W_plastic_plt.at[t, 1].set(W_se_high)
        W_plastic_plt = W_plastic_plt.at[t, 2].set(W_se_low)

        W_ee_mean = jnp.round(jnp.mean(W_plastic[N_input_neurons:-N_inhib_neurons]), 5)
        W_ee_high = jnp.round(jnp.amax(W_plastic[N_input_neurons:-N_inhib_neurons]), 5)
        W_ee_low = jnp.round(jnp.amin(W_plastic[N_input_neurons:-N_inhib_neurons]), 5)

        W_plastic_plt = W_plastic_plt.at[t, 3].set(W_ee_mean)
        W_plastic_plt = W_plastic_plt.at[t, 4].set(W_ee_high)
        W_plastic_plt = W_plastic_plt.at[t, 5].set(W_ee_low)

        W_ie_mean = jnp.round(jnp.mean(W_plastic[-N_inhib_neurons:]), 5)
        W_ie_high = jnp.round(jnp.amax(W_plastic[-N_inhib_neurons:]), 5)
        W_ie_low = jnp.round(jnp.amin(W_plastic[-N_inhib_neurons:]), 5)

        W_plastic_plt = W_plastic_plt.at[t, 6].set(W_ie_mean)
        W_plastic_plt = W_plastic_plt.at[t, 7].set(W_ie_high)
        W_plastic_plt = W_plastic_plt.at[t, 8].set(W_ie_low)

    return (
        W_plastic_plt,
        spikes,
        MemPot,
        pre_synaptic_trace,
        post_synaptic_trace,
        slow_post_synaptic_trace,
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
