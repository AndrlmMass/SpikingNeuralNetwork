import numpy as np


def update_membrane_potential(
    MemPot,
    W_exc,
    W_inh,
    S,
    N_input_neurons,
    N_inhib_neurons,
    N_excit_neurons,
    tau_m,
    R,
    dt,
    V_rest,
):
    # Update I_in
    exc_e = np.dot(W_exc[:-N_inhib_neurons].T, S[:-N_inhib_neurons])
    exc_i = np.dot(W_exc[-N_inhib_neurons:].T, S[-N_inhib_neurons:])
    I_in_e = exc_e - exc_i

    I_in_i = np.dot(
        W_inh.T,
        S[N_input_neurons:-N_inhib_neurons],
    )

    # Update membrane potential based on I_in
    delta_MemPot_e = (-((MemPot[:N_excit_neurons] - V_rest) + R * I_in_e) / tau_m) * dt
    MemPot[:N_excit_neurons] = MemPot[:N_excit_neurons] - np.round(delta_MemPot_e, 4)

    delta_MemPot_i = (-((MemPot[N_excit_neurons:] - V_rest) + R * I_in_i) / tau_m) * dt
    MemPot[N_excit_neurons:] = MemPot[N_excit_neurons:] - np.round(delta_MemPot_i, 4)

    return MemPot


def update_membrane_potential_conduct(
    U,
    U_inh,
    U_exc,
    V_th,
    V_th_,
    W_exc,
    W_inh,
    S,
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
):
    """
    Overview over variables:

    -> Dynamic (i.e., changing for each step)
        U, V_th, g_ampa, g_nmda, g_gaba, x, u, g_a, g_b, W_exc, W_inh, S

    -> Static (i.e., not changing)
        U_inh, U_exc, V_th_, tau_m, dt, N_input_neurons, N_inhib_neurons,
        V_rest, alpha_exc, alpha_inh, tau_ampa, tau_nmda, tau_gaba, tau_thr,
        tau_d, tau_f, tau_a, tau_b, delta_a, delta_b, U_cons

    """

    ### Update excitatory membrane potential ###

    # Update spike indices
    S_j_exc = np.expand_dims(S[:-N_inhib_neurons], axis=1)  # presynaptic exc spikes
    S_j_inh = np.expand_dims(S[-N_inhib_neurons:], axis=1)  # presynaptic inh spikes
    S_i = np.expand_dims(
        S[N_input_neurons:-N_inhib_neurons], axis=1
    )  # postsynaptic spikes

    # Update weight indices
    w_ij_exc = W_exc[:-N_inhib_neurons]  # excitatory weights
    w_ij_inh = W_inh  # inhibitory weights

    # Update membrane potential indices
    U_e = np.expand_dims(U[:-N_inhib_neurons], axis=1)

    # Update traces indices
    g_ampa_exc = g_ampa[:-N_inhib_neurons]  # shape: (484, 1)
    g_nmda_exc = g_nmda[:-N_inhib_neurons]  # shape: (484, 1)
    g_gaba_exc = g_gaba  # shape: (121, 1)

    # Update traces
    g_a += dt * (-(g_a / tau_a) + delta_a * S_i)  # shape:(484, 1)
    g_b += dt * (-(g_b / tau_b) + delta_b * S_i)  # shape:(484, 1)
    u += dt * (((U_cons - u) / tau_f) + (U_cons * (1 - u) * S_j_exc))  # shape:(968, 1)
    x += dt * (((1 - x) / tau_d) - u * x * S_j_exc)  # shape:(968, 1)

    # Update transmitter channels
    g_ampa_exc += dt * (
        -(g_ampa_exc / tau_ampa) + (np.dot(w_ij_exc.T, (u * x * S_j_exc)))
    )
    g_nmda_exc += dt / tau_nmda * (g_ampa_exc - g_nmda_exc)  # DONE
    g_exc = alpha_exc * g_ampa_exc + (1 - alpha_exc) * g_nmda_exc  # DONE
    g_gaba_exc += dt * (-(g_gaba_exc / tau_gaba) + np.dot(w_ij_inh, S_j_inh))  # DONE

    # Update membrane potential
    delta_U_ex = (
        dt
        / tau_m
        * (
            (V_rest - U_e)
            + (g_exc * (U_exc - U_e))
            + (g_gaba_exc + g_a + g_b) * (U_inh - U_e)
        )
    )
    print(
        "Mempot exc",
        np.round(np.mean(U[:-N_inhib_neurons]), 4),
        "g_nmda",
        np.round(np.mean(g_nmda_exc), 4),
        "g_ampa",
        np.round(np.mean(g_ampa_exc), 4),
        "g_gaba",
        np.mean(g_gaba_exc),
    )
    U[:-N_inhib_neurons] = (U_e + delta_U_ex).reshape(-1)

    ### Update inhibitory membrane potential ###

    # Update spike indices
    S_j_inh = np.expand_dims(S[N_input_neurons:-N_inhib_neurons], axis=1)
    S_i = np.expand_dims(S[-N_inhib_neurons:], axis=1)  # postsynaptic spikes

    # Update weight indices
    w_ij_inh = W_exc[-N_inhib_neurons:]  # excitatory weights

    # Update membrane potential indices
    U_i = np.expand_dims(U[-N_inhib_neurons:], axis=1)

    # Update traces indices
    g_ampa_inh = g_ampa[-N_inhib_neurons:]
    g_nmda_inh = g_nmda[-N_inhib_neurons:]

    # Update transmitter channels
    g_ampa_inh += dt * (-(g_ampa_inh / tau_ampa) + (np.dot(w_ij_inh, S_j_inh)))
    g_nmda_inh += dt / tau_nmda * (g_ampa_inh - g_nmda_inh)  # DONE
    g_inh = alpha_inh * g_ampa_inh + (1 - alpha_inh) * g_nmda_inh  # DONE

    # Update membrane potential
    delta_U_in = dt / tau_m * ((V_rest - U_i) + (g_inh * (U_exc - U_i)) + (U_inh - U_i))
    U[-N_inhib_neurons:] = (U_i + delta_U_in).reshape(-1)

    # Update spiking threshold decay for excitatory neurons
    V_th += dt / tau_thr * (V_th_ - V_th)

    # Update transmitter levels
    g_ampa = np.concatenate((g_ampa_exc, g_ampa_inh), axis=0)
    g_nmda = np.concatenate((g_nmda_exc, g_nmda_inh), axis=0)
    g_gaba = g_gaba_exc

    return U, V_th, g_ampa, g_nmda, g_gaba, x, u, g_a, g_b
