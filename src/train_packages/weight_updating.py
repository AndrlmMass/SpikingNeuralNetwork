# Update weights of the network

# Import libraries
import numpy as np


def exc_weight_update(
    dt,
    tau_cons,
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
    # Update ideal weights if t is divisible by euler
    W_se_ideal += (dt / tau_cons) * (
        W_se
        - W_se_ideal
        - P * W_se_ideal * ((w_p / 2) - W_se_ideal) * (w_p - W_se_ideal)
    )

    # Update spike variables
    post_spikes_se = spikes[N_input_neurons:-N_inhib_neurons]
    pre_spikes_se = spikes[:N_input_neurons]

    # Initiate presynaptic and postsynaptic traces
    pre_trace_se = pre_synaptic_trace[:N_input_neurons]
    post_trace_se = post_synaptic_trace
    slow_trace_se = slow_pre_synaptic_trace[:N_input_neurons]
    z_ht_se = z_ht[:N_input_neurons]
    C_se = C[:N_input_neurons]

    # Update synaptic traces
    pre_trace_se += dt * ((-pre_trace_se / tau_plus) + pre_spikes_se)
    post_trace_se += dt * ((-post_trace_se / tau_minus) + post_spikes_se)
    slow_trace_se += dt * ((-slow_trace_se / tau_slow) + post_spikes_se)

    # Update z_th, C and B variables
    z_ht_se += dt * (-z_ht_se / tau_ht + post_spikes_se)
    C_se += dt * (-C_se / tau_hom + z_ht_se**2)
    B_se = np.where(C_se <= 1 / A, A * C_se, A)

    # Get learning components
    triplet_LTP = A * pre_trace_se * slow_trace_se
    heterosynaptic = beta * post_trace_se**3 * (W_se - W_se_ideal)
    transmitter = pre_spikes_se * (B_se * post_trace_se - delta)

    # Compute the differential update for weights using Euler's method
    delta_w_se = dt * (post_spikes_se * (triplet_LTP - heterosynaptic) - transmitter)

    # Update the weights
    W_se += delta_w_se

    W_se = np.clip(W_se, 0.1, 5)

    ## W_ee weights ##

    # Update ideal weights if t is divisble by euler
    W_ee_ideal += (dt / tau_cons) * (
        W_ee
        - W_ee_ideal
        - P * W_ee_ideal * ((w_p / 2) - W_ee_ideal) * (w_p - W_ee_ideal)
    )

    # Update spike variables
    post_spikes_ee = spikes[N_input_neurons:-N_inhib_neurons]
    pre_spikes_ee = spikes[N_input_neurons:-N_inhib_neurons]

    # Initiate presynaptic and postsynaptic traces
    pre_trace_ee = pre_synaptic_trace[N_input_neurons:-N_inhib_neurons]
    post_trace_ee = post_synaptic_trace
    slow_trace_ee = slow_pre_synaptic_trace[N_input_neurons:-N_inhib_neurons]
    z_ht_ee = z_ht[N_input_neurons:-N_inhib_neurons]
    C_ee = C[N_input_neurons:-N_inhib_neurons]

    # Update synaptic traces
    pre_trace_ee += dt * (-pre_trace_ee / tau_plus + pre_spikes_ee)
    post_trace_ee += dt * (-post_trace_ee / tau_minus + post_spikes_ee)
    slow_trace_ee += dt * (-slow_trace_ee / tau_slow + post_spikes_ee)

    # Update z_th, C and B variables
    z_ht_ee += dt * (-z_ht_ee / tau_ht + post_spikes_ee)
    C_ee += dt * (-C_ee / tau_hom + z_ht_ee**2)
    B_ee = np.where(C_ee <= 1 / A, A * C_ee, A)

    # Get learning components
    triplet_LTP = A * pre_trace_ee * slow_trace_ee
    heterosynaptic = beta * post_trace_ee**3 * (W_ee - W_ee_ideal)
    transmitter = pre_spikes_ee * (B_ee * post_trace_ee - delta)

    # Compute the differential update for weights using Euler's method
    delta_w_ee = dt * (post_spikes_ee * (triplet_LTP - heterosynaptic) - transmitter)

    # Update the weights
    W_ee += delta_w_ee

    W_ee = np.clip(W_ee, 0.1, 5.0)

    ## W_se weights ##
    # mean_spikes = np.mean(spikes)
    # z_ht_mean = np.mean(z_ht)
    # C_mean = np.mean(C)
    # mean_w_se = np.mean(W_se)
    # mean_w_ee = np.mean(W_ee)
    # mean_w_se_ideal = np.mean(W_se_ideal)
    # mean_w_ee_ideal = np.mean(W_ee_ideal)
    # mean_post_trace_se = np.mean(post_trace_se)
    # mean_post_trace_ee = np.mean(post_trace_ee)
    # mean_pre_trace_se = np.mean(pre_trace_se)
    # mean_pre_trace_ee = np.mean(pre_trace_ee)

    # print(
    #     "W_ee_ideal",
    #     np.mean(W_ee_ideal),
    #     "W_se_ideal",
    #     np.mean(W_se_ideal),
    #     "W_ee",
    #     np.mean(W_ee),
    #     "W_se",
    #     np.mean(W_se),
    # )

    # print(sum_post_trace_ee, sum_post_trace_se, sum_pre_trace_ee, sum_pre_trace_se)

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
        slow_trace_ee,
        z_ht_ee,
        C_ee,
    )


def inh_weight_update(
    H,
    dt,
    W_inh,
    z_i,
    z_j,
    tau_H,
    gamma,
    tau_stdp,
    learning_rate,
    pre_spikes,
    post_spikes,
):
    # Update synaptic traces using Euler's method
    z_i += dt * (-z_i / tau_stdp + post_spikes)
    z_j += dt * (-z_j / tau_stdp + pre_spikes)

    # Update H using Euler's method
    H += dt * (-H / tau_H + np.sum(post_spikes))
    G = H - gamma

    # Reshape arrays for matrix operations
    z_i_reshaped = np.expand_dims(z_i, axis=1)
    z_j_reshaped = np.expand_dims(z_j, axis=1)
    post_spikes_reshaped = np.expand_dims(post_spikes, axis=1)
    pre_spikes_reshaped = np.expand_dims(pre_spikes, axis=1)

    # Calculate delta weights
    delta_w = (
        dt
        * learning_rate
        * G
        * (
            np.dot(z_i_reshaped + 1, pre_spikes_reshaped.T)
            + np.dot(post_spikes_reshaped, z_j_reshaped.T)
        )
    )
    # Update weights with constraint
    W_inh += delta_w

    W_inh = np.clip(W_inh, 0.1, 5.0)

    return W_inh, z_i, z_j, H
