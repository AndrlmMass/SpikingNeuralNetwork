import numpy as np


# Create function that takes in spikes and indices and outputs an adjusted membrane potential
def adjust_membrane_threshold(
    spikes, t, update_freq, V_th, V_reset, N_input_neurons, N_excit_neurons
):
    ## Excitatory neurons ##

    # Calculate total spikes in the previous time interval
    tot_spik_e = np.sum(spikes[N_input_neurons : N_input_neurons + N_excit_neurons,])

    # Calculate total spikes per neuron (axis=0) in the previous time interval
    per_spik_e = np.sum(
        spikes[N_input_neurons : N_input_neurons + N_excit_neurons,],
        axis=0,
    )

    # Calculate the ratio between local and global spikes
    ratio_e = np.where(tot_spik_e > 0, per_spik_e / tot_spik_e, 0.01)

    # Update the membrane potential threshold according to the radio_e
    V_th[:N_excit_neurons] = np.maximum(
        V_reset + (V_th[:N_excit_neurons] - V_reset) * np.exp(-ratio_e), V_reset
    )

    ## Inhibitory neurons ##

    # Calculate total spikes in the previous time interval
    tot_spik_i = np.sum(
        spikes[t - update_freq : t - 1, N_input_neurons + N_excit_neurons :]
    )

    # Calculate total spikes per neuron (axis=0) in the previous time interval
    per_spik_i = np.sum(
        spikes[t - update_freq : t - 1, N_input_neurons + N_excit_neurons :],
        axis=0,
    )

    # Calculate ratio between local and global spikes
    ratio_i = np.where(tot_spik_i > 0, per_spik_i / tot_spik_i, 0.01)

    # Update membrane potential threshold according to the ratio_i
    V_th[N_excit_neurons:] = np.maximum(
        V_reset + (V_th[N_excit_neurons:] - V_reset) * np.exp(-ratio_i), V_reset
    )

    return V_th


def update_membrane_potential(
    MemPot,
    W_se,
    W_ee,
    W_ie,
    W_ei,
    spikes,
    t,
    dt,
    N_excit_neurons,
    N_input_neurons,
    N_inhib_neurons,
    V_rest,
    R,
    tau_mm,
):
    # Update I_in
    I_in_e = (
        np.dot(W_se, spikes[:-1, :N_input_neurons])
        + np.dot(
            W_ee,
            spikes[:-1, N_input_neurons:-N_inhib_neurons],
        )
        + np.dot(
            W_ie.T,
            spikes[:-1, N_input_neurons + N_excit_neurons :],
        )
    )

    I_in_i = np.dot(
        W_ei,
        spikes[N_input_neurons:-N_inhib_neurons],
    )

    # Update membrane potential based on I_in
    delta_MemPot_e = (-((MemPot[:N_excit_neurons] - V_rest) + R * I_in_e) / tau_mm) * dt
    MemPot[:N_excit_neurons] = MemPot[:N_excit_neurons] - round(delta_MemPot_e, 4)

    delta_MemPot_i = (-((MemPot[N_excit_neurons:] - V_rest) + R * I_in_i) / tau_mm) * dt
    MemPot[N_excit_neurons:] = MemPot[N_excit_neurons:] - round(delta_MemPot_i, 4)

    return MemPot
