import numpy as np


# Create function that takes in spikes and indices and outputs an adjusted membrane potential
def adjust_membrane_threshold(
    spikes,
    V_th,
    N_input_neurons,
    N_excit_neurons,
    N_inhib_neurons,
    dt,
    tau_thr,
    v_rest,
):
    ## Excitatory neurons ##

    # Update the membrane potential threshold according to the radio_e
    # V_th[:N_excit_neurons] += (
    #     (dt / tau_thr)
    #     * (v_rest - V_th[:N_excit_neurons])
    #     * spikes[N_input_neurons:-N_inhib_neurons]
    # )

    ## Inhibitory neurons ##

    # Update membrane potential threshold according to the ratio_i
    # V_th[N_excit_neurons:] += (
    #     V_th[N_excit_neurons:]
    #     + (dt / tau_thr) * (v_rest - V_th[N_excit_neurons:]) * spikes[-N_inhib_neurons:]
    # )

    return V_th


def update_membrane_potential(
    MemPot,
    W_se,
    W_ee,
    W_ie,
    W_ei,
    spikes,
    dt,
    N_excit_neurons,
    N_input_neurons,
    N_inhib_neurons,
    V_rest,
    R,
    tau_m,
):
    # Update I_in
    I_in_e = (
        np.dot(W_se, spikes[:N_input_neurons])
        + np.dot(
            W_ee,
            spikes[N_input_neurons:-N_inhib_neurons],
        )
        - np.dot(
            W_ie.T,
            spikes[N_input_neurons + N_excit_neurons :],
        )
    )

    I_in_i = np.dot(
        W_ei.T,
        spikes[N_input_neurons:-N_inhib_neurons],
    )

    # Update membrane potential based on I_in
    delta_MemPot_e = (-((MemPot[:N_excit_neurons] - V_rest) + R * I_in_e) / tau_m) * dt
    MemPot[:N_excit_neurons] = MemPot[:N_excit_neurons] - np.round(delta_MemPot_e, 4)

    delta_MemPot_i = (-((MemPot[N_excit_neurons:] - V_rest) + R * I_in_i) / tau_m) * dt
    MemPot[N_excit_neurons:] = MemPot[N_excit_neurons:] - np.round(delta_MemPot_i, 4)

    I_in = sum(I_in_e) + sum(I_in_i)

    return MemPot, I_in
