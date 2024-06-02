# Update weights of the network

# Import libraries
import numpy as np


def exc_weight_update(
    dt,
    tau_const,
    W_se,
    W_ee,
    W_se_ideal,
    W_ee_ideal,
    P,
    t,
    w_p,
    spikes,
    N_input_neurons,
    N_inhib_neurons,
    pre_synaptic_trace,
    post_synaptic_trace,
    slow_pre_synaptic_trace,
    tau_plus,
    tau_minus,
    tau_slow,
    tau_ht,
    tau_hom,
    A,
    beta,
    delta,
    euler,
    z_ht,
    C,
):
    ## W_se weights ##

    # Check if ideal weight needs to be updated
    update_bin = int(t % euler == 0)

    # Update ideal weights if t is divisible by euler
    W_se_ideal += (
        (dt / tau_const)
        * (
            W_se
            - W_se_ideal
            - P * W_se_ideal * ((w_p / 2) - W_se_ideal) * (w_p - W_se_ideal)
        )
        * update_bin
    )
    # Update spike variables
    post_spikes_se = spikes[N_input_neurons:-N_inhib_neurons]
    pre_spikes_se = spikes[:N_input_neurons]

    # Initiate presynaptic and postsynaptic traces
    pre_trace_se = pre_synaptic_trace[:N_input_neurons]
    post_trace_se = post_synaptic_trace[N_input_neurons:-N_inhib_neurons]
    slow_trace_se = slow_pre_synaptic_trace[:N_input_neurons]
    z_ht_se = z_ht[:N_input_neurons]
    C_se = C[:N_input_neurons]

    # Update synaptic traces
    pre_trace_se += dt * (-pre_trace_se / tau_plus + pre_spikes_se)
    post_spikes_se += dt * (-post_trace_se / tau_minus + post_spikes_se)
    slow_trace_se += dt * (-slow_trace_se / tau_slow + pre_spikes_se)

    # Update z_th, C and B variables
    z_ht_se += dt * (-z_ht_se / tau_ht + pre_spikes_se)
    C_se += dt * (-C_se / tau_hom + z_ht_se**2)
    B_se = np.where(C <= 1 / A, A * C, A)

    # Get learning components
    A_z = A * pre_trace_se * slow_trace_se
    Beta_z = beta * (post_trace_se**3) * (W_se - W_se_ideal)
    B_z = B_se * post_trace_se - delta

    # Compute the differential update for weights using Euler's method
    delta_w_se = dt * (post_spikes_se * (A_z - Beta_z) - pre_spikes_se * B_z)

    # Update the weights
    W_se += delta_w_se

    ## W_ee weights ##

    # Update ideal weights if t is divisble by euler
    W_ee_ideal += (
        (dt / tau_const)
        * (
            W_ee
            - W_ee_ideal
            - P * W_ee_ideal * ((w_p / 2) - W_ee_ideal) * (w_p - W_ee_ideal)
        )
        * update_bin
    )

    # Update spike variables
    post_spikes_ee = spikes[N_input_neurons:-N_inhib_neurons]
    pre_spikes_ee = spikes[N_input_neurons:-N_inhib_neurons]

    # Initiate presynaptic and postsynaptic traces
    pre_trace_ee = pre_synaptic_trace[N_input_neurons:-N_inhib_neurons]
    slow_trace_ee = slow_pre_synaptic_trace[N_input_neurons:-N_inhib_neurons]
    z_ht_ee = z_ht[N_input_neurons:-N_inhib_neurons]
    C_ee = C[N_input_neurons:-N_inhib_neurons]

    # Update synaptic traces
    pre_trace_ee += dt * (-pre_trace_ee / tau_plus + pre_spikes_ee)
    post_trace_ee = post_trace_se
    slow_trace_ee += dt * (-slow_trace_ee / tau_slow + pre_spikes_ee)

    # Update z_th, C and B variables
    z_ht_ee += dt * (-z_ht_ee / tau_ht + pre_spikes_se)
    C_ee += dt * (-C_ee / tau_hom + z_ht_ee**2)
    B_ee = np.where(C <= 1 / A, A * C, A)

    # Get learning components
    A_z = A * pre_trace_ee * slow_trace_ee
    Beta_z = beta * (post_trace_ee**3) * (W_se - W_se_ideal)
    B_z = B_ee * post_spikes_ee - delta

    # Compute the differential update for weights using Euler's method
    delta_w_ee = dt * (post_spikes_ee * (A_z - Beta_z) - pre_spikes_ee * B_z)

    # Update the weights
    W_ee += delta_w_ee

    return (
        W_se,
        W_ee,
        W_se_ideal,
        W_ee_ideal,
        pre_trace_se,
        post_trace_se,
        slow_trace_se,
        z_ht_se,
        C_se,
        pre_trace_ee,
        post_trace_ee,
        slow_trace_ee,
        z_ht_ee,
        C_ee,
    )


def inh_weight_update(
    H,
    dt,
    W_ie,
    z_istdp,
    tau_H,
    gamma,
    tau_stdp,
    learning_rate,
    spikes,
    N_input_neurons,
    N_inhib_neurons,
    post_synaptic_trace,
):

    # Define post and pre spikes
    post_spikes = spikes[N_input_neurons:-N_inhib_neurons]
    pre_spikes = spikes[-N_inhib_neurons:]

    # Update synaptic traces using Euler's method
    z_istdp += dt * (-z_istdp / tau_stdp + pre_spikes)
    post_trace = post_synaptic_trace[N_input_neurons:-N_inhib_neurons]

    # Update H using Euler's method
    H += dt * (-H / tau_H + np.sum(spikes[N_input_neurons:-N_inhib_neurons]))
    G = H - gamma

    # Reshape arrays for matrix operations
    z_istdp_reshaped = z_istdp.reshape(-1, 1)
    post_spikes_reshaped = post_spikes.reshape(1, -1)
    pre_spikes_reshaped = pre_spikes.reshape(-1, 1)

    # Calculate delta weights
    delta_w = (
        dt
        * learning_rate
        * G
        * (
            (z_istdp_reshaped + 1) @ post_spikes_reshaped
            + z_istdp_reshaped @ pre_spikes_reshaped.T
        )
    )

    # Update weights with constraint
    W_ie += delta_w

    # Clip weights to be between 0 and 5
    W_ie = np.clip(W_ie, 0, 5)

    return W_ie, z_istdp, H
