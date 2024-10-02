from numba import prange


def update_membrane_potential_conduct(
    U,
    U_inh,
    U_exc,
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
    Total_N = len(S)
    N_presyn_exc = Total_N - N_inhib_neurons  # Total presynaptic excitatory neurons
    N_post_exc = (
        Total_N - N_input_neurons - N_inhib_neurons
    )  # Postsynaptic excitatory neurons
    N_presyn_inh = N_inhib_neurons  # Presynaptic inhibitory neurons
    idx_exc_start = N_input_neurons
    idx_exc_end = Total_N - N_inhib_neurons
    idx_inh_start = Total_N - N_inhib_neurons

    ### Update u and x for presynaptic excitatory neurons ###
    for j in prange(N_presyn_exc):
        S_j = S[j]
        u_j = u[j]
        x_j = x[j]
        u[j] = u_j + dt * ((U_cons - u_j) / tau_f + U_cons * (1 - u_j) * S_j)
        x[j] = x_j + dt * ((1 - x_j) / tau_d - u_j * x_j * S_j)

    ### Update postsynaptic excitatory neurons ###
    for i in prange(N_post_exc):
        neuron_idx = idx_exc_start + i
        S_i = S[neuron_idx]

        # Update g_a and g_b
        g_a[i] += dt * (-g_a[i] / tau_a + delta_a * S_i)
        g_b[i] += dt * (-g_b[i] / tau_b + delta_b * S_i)

        # Initialize conductance updates
        delta_g_ampa_e = -g_ampa[neuron_idx] / tau_ampa * dt
        delta_g_nmda_e = (g_ampa[neuron_idx] - g_nmda[neuron_idx]) / tau_nmda * dt

        # Sum over presynaptic excitatory neurons
        sum_exc = 0.0
        for j in range(N_presyn_exc):
            w_ij = W_plastic[neuron_idx, j]
            sum_exc += w_ij * u[j] * x[j] * S[j]
        delta_g_ampa_e += dt * sum_exc

        g_ampa[neuron_idx] += delta_g_ampa_e
        g_nmda[neuron_idx] += delta_g_nmda_e

        g_e = alpha_exc * g_ampa[neuron_idx] + (1 - alpha_exc) * g_nmda[neuron_idx]

        # Sum over presynaptic inhibitory neurons
        sum_inh = 0.0
        for j_inh in range(N_presyn_inh):
            presyn_inh_idx = idx_inh_start + j_inh
            w_ij_inh = W_plastic[neuron_idx, presyn_inh_idx]
            sum_inh += w_ij_inh * S[presyn_inh_idx]
        delta_g_gaba_e = dt * (-g_gaba[neuron_idx] / tau_gaba + sum_inh)
        g_gaba[neuron_idx] += delta_g_gaba_e

        # Update membrane potential
        delta_U_ex = (
            dt
            / tau_m
            * (
                V_rest
                - U[neuron_idx]
                + g_e * (U_exc - U[neuron_idx])
                + (g_gaba[neuron_idx] + g_a[i] + g_b[i]) * (U_inh - U[neuron_idx])
            )
        )
        U[neuron_idx] += delta_U_ex

    ### Update postsynaptic inhibitory neurons ###
    for i in prange(N_inhib_neurons):
        neuron_idx = idx_inh_start + i
        S_i = S[neuron_idx]

        # Initialize conductance updates
        delta_g_ampa_i = -g_ampa[neuron_idx] / tau_ampa * dt
        delta_g_nmda_i = (g_ampa[neuron_idx] - g_nmda[neuron_idx]) / tau_nmda * dt

        # Sum over presynaptic excitatory neurons
        sum_exc = 0.0
        for j in range(idx_exc_start, idx_exc_end):
            w_ij_exc = W_static[neuron_idx, j]
            sum_exc += w_ij_exc * S[j]
        delta_g_ampa_i += dt * sum_exc

        g_ampa[neuron_idx] += delta_g_ampa_i
        g_nmda[neuron_idx] += delta_g_nmda_i

        g_i = alpha_inh * g_ampa[neuron_idx] + (1 - alpha_inh) * g_nmda[neuron_idx]

        # Sum over presynaptic inhibitory neurons
        sum_inh = 0.0
        for j_inh in range(N_inhib_neurons):
            presyn_inh_idx = idx_inh_start + j_inh
            w_ij_inh = W_static[neuron_idx, presyn_inh_idx]
            sum_inh += w_ij_inh * S[presyn_inh_idx]
        delta_g_gaba_i = dt * (-g_gaba[neuron_idx] / tau_gaba + sum_inh)
        g_gaba[neuron_idx] += delta_g_gaba_i

        # Update membrane potential
        delta_U_in = (
            dt
            / tau_m
            * (
                V_rest
                - U[neuron_idx]
                + g_i * (U_exc - U[neuron_idx])
                + g_gaba[neuron_idx] * (U_inh - U[neuron_idx])
            )
        )
        U[neuron_idx] += delta_U_in

    return U, g_ampa, g_nmda, g_gaba, x, u, g_a, g_b
