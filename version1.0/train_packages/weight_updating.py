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
    z_ht,
    C,
):
    ## W_se weights ##

    # Update ideal weights
    print(W_se_ideal.shape)
    W_se_ideal += (
        dt
        * tau_const
        * (
            W_se
            - W_se_ideal
            - P * W_se_ideal * ((w_p / 2) - W_se_ideal) * (w_p - W_se_ideal)
        )
    )

    # Update spike variables
    post_spikes_se = spikes[N_input_neurons:-N_inhib_neurons]
    pre_spikes_se = spikes[:N_input_neurons]
    pre_synaptic_trace *= np.exp(-dt / tau_plus)
    post_synaptic_trace *= np.exp(-dt / tau_minus)
    slow_pre_synaptic_trace *= np.exp(-dt / tau_slow)
    print(pre_synaptic_trace.shape, pre_spikes_se.shape)

    # Update synaptic traces
    pre_trace_se = pre_synaptic_trace[:N_input_neurons] + pre_spikes_se * dt
    post_trace_se = post_synaptic_trace[:N_excit_neurons] + post_spikes_se * dt
    slow_trace_se = slow_pre_synaptic_trace[:N_input_neurons] + pre_spikes_se * dt

    # Update z_th, C and B -> this is only done once for W_se and W_ee
    z_ht = z_ht * np.exp(-dt / tau_ht) + spikes * dt
    C = C * np.exp(-dt / tau_hom) + z_ht**2
    B = np.where(A * C <= 1, C, A)

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

    # Update ideal weights
    W_ee_ideal += (
        dt
        * tau_const
        * (
            W_ee
            - W_ee_ideal
            - P * W_ee_ideal * ((w_p / 2) - W_ee_ideal) * (w_p - W_ee_ideal)
        )
    )

    # Update spike variables
    post_spikes_ee = spikes[N_input_neurons:-N_inhib_neurons]
    pre_spikes_ee = spikes[N_input_neurons:-N_inhib_neurons]

    # Update synaptic traces
    pre_trace_ee = (
        pre_synaptic_trace[N_input_neurons:-N_inhib_neurons] + pre_spikes_ee * dt
    )
    post_trace_ee = post_synaptic_trace[:-N_inhib_neurons] + post_spikes_ee * dt
    slow_trace_ee = (
        slow_pre_synaptic_trace[N_input_neurons:-N_inhib_neurons] + pre_spikes_se * dt
    )

    # Get learning components
    triplet_LTP = A * post_trace_ee * slow_trace_ee * pre_spikes_ee

    doublet_LTD = B[N_input_neurons:-N_inhib_neurons] * pre_trace_ee * post_spikes_ee

    Hebb = np.round(triplet_LTP - doublet_LTD, 6)

    Hetero = np.round(
        -beta * (W_se - W_se_ideal) * (post_trace_ee) ** 3 * pre_spikes_ee,
        6,
    )

    transmitter = delta * post_spikes_ee

    # Assemble components to update weight
    delta_w = Hebb + Hetero + transmitter

    W_se += delta_w

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
    print(z_istdp.shape, post_spikes.shape, post_trace.shape, pre_spikes.shape)

    # Calculate delta weights
    delta_w = learning_rate * G * (z_istdp + 1) * post_spikes + post_trace * pre_spikes

    # Update weights
    W_ie += delta_w

    return W_ie, z_istdp, H, post_trace
