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
                    W_se,
                    W_ee,
                    W_ie,
                    W_ei,
                    S,
                    dt,
                    N_excit_neurons,
                    N_input_neurons,
                    N_inhib_neurons,
                    V_rest,
                    tau_m,
                    alpha,
                    tau_ampa,
                    tau_nmda,
                    tau_gaba,
                    tau_d,
                    tau_f,
                    tau_a,
                    delta_a,
                    V_th,
                    ):
    
    # Update components for membrane change
    S_j = S[:] # Add post_synaptic neuron spikes here -> sure which are post in this scenario since we are going through all the neurons?
    
    x = dt * (((1-x)/tau_d) - u*x*S_j)
    u = dt * (((U-u)/tau_f)+U*(1-u)*S_j)
    
    g_ampa = dt * ((-g_ampa/tau_ampa) + (np.sum(x*u*w*S))) # It seems the weights are individualized for this updating, maybe this does not work?
    g_nmda = tau_nmda * dt * (-g_nmda + g_ampa)
    g_exc = dt * (alpha*g_ampa + (1-alpha)*g_nmda)
    g_gaba = dt * (-(g_gaba/tau_gaba)+np.sum(w*S_j))
    g_a = dt * (-(g_a/tau_a)+delta_a*S_j)


    # Update membrane potential
    U = tau_m * dt * ((V_rest - U) + g_exc*(U_exc-U)+(g_gaba + g_a)*(U_inh - U))

    # Update spiking threshold
    V_th
    
