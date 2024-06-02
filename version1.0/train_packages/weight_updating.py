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
    N_excit_neurons,
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
    pre_trace_se = pre_synaptic_trace[:-N_inhib_neurons]
    post_trace_se = post_synaptic_trace[N_input_neurons:-N_inhib_neurons]
    slow_trace_se = slow_pre_synaptic_trace[:-N_inhib_neurons]

    # Update synaptic traces
    pre_trace_se += dt * (-pre_trace_se / tau_plus + pre_spikes_se)
    post_spikes_ee += dt * (-post_trace_se / tau_minus + post_spikes_se)
    slow_trace_se += dt * (-slow_trace_se / tau_slow + pre_spikes_se)

    # Update z_th, C and B -> this is only done once for W_se and W_ee
    z_ht += dt * (-z_ht / tau_ht + current_spikes)

    # Update C trace
    C = C + dt * (-C / tau_hom + z_ht**2)

    # Update B value based on C
    B = np.where(C <= 1 / A, A * C, A)

    # Get learning components
    triplet_LTP = A * post_trace_se * slow_trace_se * pre_spikes_se

    doublet_LTD = B[:N_input_neurons] * pre_trace_se * post_spikes_se

    Hebb = np.round(triplet_LTP - doublet_LTD, 6)

    Hetero = np.round(
        -beta * (W_se - W_se_ideal) * post_trace_se**3 * pre_spikes_se,
        6,
    )
    transmitter = np.round(delta * post_spikes_se, 6)

    # Assemble components to update weight
    delta_w = Hebb + Hetero + transmitter

    W_se += delta_w

    ## W_ee weights ##

    # Update ideal weights if t is divisble by euler
    W_ee_ideal += (
        dt
        * tau_const
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
    post_trace_ee = post_synaptic_trace[N_input_neurons:-N_inhib_neurons]
    slow_trace_ee = slow_pre_synaptic_trace[N_input_neurons:-N_inhib_neurons]

    # Update synaptic traces
    pre_trace_ee += dt * (-pre_trace_ee / tau_plus + pre_spikes_ee)
    post_trace_ee += dt * (-post_trace_ee / tau_minus + post_spikes_ee)
    slow_trace_ee += dt * (-slow_trace_ee / tau_slow + pre_spikes_ee)

    # Get learning components
    triplet_LTP = A * post_trace_ee * slow_trace_ee * pre_spikes_ee

    doublet_LTD = B[N_input_neurons:-N_inhib_neurons] * pre_trace_ee * post_spikes_ee

    Hebb = np.round(triplet_LTP - doublet_LTD, 6)

    Hetero = np.round(
        -beta * (W_ee - W_ee_ideal) * (post_trace_ee) ** 3 * pre_spikes_ee,
        6,
    )

    transmitter = delta * post_spikes_ee

    # Assemble components to update weight
    delta_w = Hebb + Hetero + transmitter

    W_ee += delta_w

    return (
        W_se,
        W_ee,
        W_se_ideal,
        W_ee_ideal,
        pre_synaptic_trace,
        post_synaptic_trace,
        slow_pre_synaptic_trace,
        z_ht,
        C,
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

    # Update synaptic traces
    z_istdp = z_istdp * np.exp(-dt / tau_stdp) + pre_spikes * dt
    post_trace = post_synaptic_trace[:-N_inhib_neurons] + post_spikes * dt

    # Update H
    H += (-(H / tau_H) + sum(spikes[N_input_neurons:-N_input_neurons])) * dt
    G = H - gamma

    # Reshape arrays
    z_istdp_reshaped = z_istdp.reshape(-1, 1)
    post_spikes_reshaped = post_spikes.reshape(1, -1)
    post_trace_reshaped = post_trace.reshape(1, -1)
    pre_spikes_reshaped = pre_spikes.reshape(-1, 1)

    # Calculate delta weights
    delta_w = learning_rate * G * np.dot(
        (z_istdp_reshaped + 1), post_spikes_reshaped
    ) + np.dot(pre_spikes_reshaped, post_trace_reshaped)

    # Updat weights
    W_ie += delta_w

    return W_ie, z_istdp, H, post_trace
