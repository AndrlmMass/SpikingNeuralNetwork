import jax.numpy as jnp


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
    exc_e = jnp.dot(W_exc[:-N_inhib_neurons].T, S[:-N_inhib_neurons])
    exc_i = jnp.dot(W_exc[-N_inhib_neurons:].T, S[-N_inhib_neurons:])
    I_in_e = exc_e - exc_i

    I_in_i = jnp.dot(
        W_inh.T,
        S[N_input_neurons:-N_inhib_neurons],
    )

    # Update membrane potential based on I_in
    delta_MemPot_e = (-((MemPot[:N_excit_neurons] - V_rest) + R * I_in_e) / tau_m) * dt
    MemPot = MemPot.at[:N_excit_neurons].set(
        MemPot[:N_excit_neurons] - jnp.round(delta_MemPot_e, 4)
    )

    delta_MemPot_i = (-((MemPot[N_excit_neurons:] - V_rest) + R * I_in_i) / tau_m) * dt
    MemPot = MemPot.at[N_excit_neurons:].set(
        MemPot[N_excit_neurons:] - jnp.round(delta_MemPot_i, 4)
    )

    return MemPot


def update_membrane_potential_conduct(
    U,
    U_inh,
    U_exc,
    V_th,
    V_th_,
    W_plastic,
    W_static,
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
    ### Update excitatory membrane potential ###

    # Update spike indices
    S_j_exc = jnp.expand_dims(S[:-N_inhib_neurons], axis=1)  # presynaptic exc spikes
    S_j_inh = jnp.expand_dims(S[-N_inhib_neurons:], axis=1)  # presynaptic inh spikes
    S_i = jnp.expand_dims(
        S[N_input_neurons:-N_inhib_neurons], axis=1
    )  # postsynaptic spikes

    # Update weight indices
    w_ij_exc = W_plastic[:-N_inhib_neurons]  # plastic excitatory weights
    w_ij_inh = W_plastic[-N_inhib_neurons:]  # plastic inhibitory weights

    # Update membrane potential indices
    U_e = jnp.expand_dims(U[:-N_inhib_neurons], axis=1)

    # Update traces indices
    g_ampa_e = g_ampa[:-N_inhib_neurons]  # shape: (484, 1)
    g_nmda_e = g_nmda[:-N_inhib_neurons]  # shape: (484, 1)
    g_gaba_e = g_gaba[:-N_inhib_neurons]  # shape: (121, 1)

    # Update traces
    g_a = g_a + dt * (-(g_a / tau_a) + delta_a * S_i)  # shape:(484, 1)
    g_b = g_b + dt * (-(g_b / tau_b) + delta_b * S_i)  # shape:(484, 1)
    u = u + dt * (
        ((U_cons - u) / tau_f) + (U_cons * (1 - u) * S_j_exc)
    )  # shape:(968, 1)
    x = x + dt * (((1 - x) / tau_d) - u * x * S_j_exc)  # shape:(968, 1)

    # Update transmitter channels
    g_ampa_e = g_ampa_e + dt * (
        -(g_ampa_e / tau_ampa) + (jnp.dot(w_ij_exc.T, (u * x * S_j_exc)))
    )
    g_nmda_e = g_nmda_e + dt / tau_nmda * (g_ampa_e - g_nmda_e)
    g_e = alpha_exc * g_ampa_e + (1 - alpha_exc) * g_nmda_e
    g_gaba_e = g_gaba_e + dt * (-(g_gaba_e / tau_gaba) + jnp.dot(w_ij_inh.T, S_j_inh))

    # Update membrane potential
    delta_U_ex = (
        dt
        / tau_m
        * (
            (V_rest - U_e)
            + (g_e * (U_exc - U_e))
            + (g_gaba_e + g_a + g_b) * (U_inh - U_e)
        )
    )

    U = U.at[:-N_inhib_neurons].set((U_e + delta_U_ex).reshape(-1) + 0.01)

    ### Update inhibitory membrane potential ###

    # Update spike indices
    S_j_exc = jnp.expand_dims(S[N_input_neurons:-N_inhib_neurons], axis=1)
    S_j_inh = jnp.expand_dims(S[-N_inhib_neurons:], axis=1)
    S_i = jnp.expand_dims(S[-N_inhib_neurons:], axis=1)  # postsynaptic spikes

    # Update weight indices
    w_ij_exc = W_static[:-N_inhib_neurons]  # excitatory weights (4, 1)
    w_ij_inh = W_static[-N_inhib_neurons:]  # inhibitory weights (1, 1)

    # Update membrane potential indices
    U_i = jnp.expand_dims(U[-N_inhib_neurons:], axis=1)

    # Update traces indices
    g_ampa_i = g_ampa[-N_inhib_neurons:]
    g_nmda_i = g_nmda[-N_inhib_neurons:]
    g_gaba_i = g_gaba[-N_inhib_neurons:]

    # Update transmitter channels
    g_ampa_i = g_ampa_i + dt * (
        -(g_ampa_i / tau_ampa) + (jnp.dot(w_ij_exc.T, (S_j_exc)))
    )
    g_nmda_i = g_nmda_i + dt / tau_nmda * (g_ampa_i - g_nmda_i)
    g_i = alpha_inh * g_ampa_i + (1 - alpha_inh) * g_nmda_i
    g_gaba_i = g_gaba_i + dt * (-(g_gaba_i / tau_gaba) + jnp.dot(w_ij_inh.T, S_j_inh))

    # Update membrane potential
    delta_U_in = (
        dt
        / tau_m
        * ((V_rest - U_i) + (g_i * (U_exc - U_i)) + (g_gaba_i) * (U_inh - U_i))
    )

    U = U.at[-N_inhib_neurons:].set((U_i + delta_U_in).reshape(-1))

    # Update spiking threshold decay for excitatory neurons
    V_th = V_th + dt / tau_thr * (V_th_ - V_th)

    # Update transmitter levels
    g_ampa = jnp.concatenate((g_ampa_e, g_ampa_i), axis=0)
    g_nmda = jnp.concatenate((g_nmda_e, g_nmda_i), axis=0)
    g_gaba = jnp.concatenate((g_gaba_e, g_gaba_i), axis=0)

    return U, V_th, g_ampa, g_nmda, g_gaba, x, u, g_a, g_b
