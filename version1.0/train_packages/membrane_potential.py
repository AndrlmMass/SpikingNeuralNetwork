import numpy as np


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
            spikes[-N_inhib_neurons:],
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

    return MemPot, I_in_i, I_in_e


def update_membrane_potential_conduct(
    U,
    U_inh,
    U_exc,
    V_th,
    V_th_rest,
    W_exc,
    W_inh,
    W_exc_ideal,
    W_inh_ideal,
    S,
    u,
    x,
    dt,
    N_input_neurons,
    N_inhib_neurons,
    V_rest,
    tau_m,
    alpha,
    tau_ampa,
    tau_nmda,
    tau_gaba,
    tau_th,
    tau_d,
    tau_f,
    tau_a,
    tau_b,
    delta_a,
    delta_b,
):
    ### Update membrane potential for excitatory neurons ###

    # Update synaptic variables
    S_j = S.reshape((-1, 1))  # pre synaptic exc spikes
    x_j = x.reshape((-1, 1))  # pre synaptic thingy
    u_j = u.reshape((-1, 1))  # pre synaptic thingy
    S_i = S[N_input_neurons:-N_inhib_neurons].reshape((1, -1))  # post synaptic spikes
    x_j += dt * (((1 - x_j) / tau_d) - u_j * x_j * S_j)
    u_j += dt * (((U - u_j) / tau_f) + U * (1 - u_j) * S_j) - 3
    w_ij = W_exc  # weights

    g_ampa += dt * ((-g_ampa / tau_ampa) + (np.sum((w_ij * x_j) * (u_j * S_j.T))))
    g_nmda += tau_nmda * dt * (-g_nmda + g_ampa)
    g_exc += dt * (alpha * g_ampa + (1 - alpha) * g_nmda)
    g_gaba += dt * (-(g_gaba / tau_gaba) + np.sum(w_ij * S_j))
    g_a += dt * (-(g_a / tau_a) + delta_a * S_i)
    g_b += dt * (-(g_b / tau_b) + (delta_b * S_i))

    # Update membrane potential
    U += (
        tau_m * dt * ((V_rest - U) + g_exc * (U_exc - U) + (g_gaba + g_a) * (U_inh - U))
    )

    # Update spiking threshold
    V_th += tau_th * dt * (V_th_rest - V_th)

    return U, V_th, g_ampa, g_nmda, g_exc, g_gaba, g_a
