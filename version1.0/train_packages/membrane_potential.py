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
    alpha,
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
    alpha_e,
    alpha_i,
    ):
    ### Update excitatory membrane potential ###

    # Update post and pre spikes
    S_j_exc = S[:-N_inhib_neurons].reshape(-1,1) # presynaptic exc spikes
    S_j_inh = S[-N_inhib_neurons].reshape(-1,1) # presynaptic inh spikes
    S_i = S[N_input_neurons:-N_inhib_neurons].reshape(-1,1) # postsynaptic spikes
    
    # Update weight indices
    w_ij_exc = W_exc[:-N_inhib_neurons] # exc weights
    w_ij_inh = W_exc[-N_inhib_neurons] # inh weights
    
    # Update traces indices
    g_a = g_a[:-N_inhib_neurons].reshape(-1,1)
    g_b = g_b[:-N_inhib_neurons].reshape(-1,1)
    x_exc = x[:-N_inhib_neurons].reshape(-1,1) # Need to be initiated with extra dim, not added for every time step
    u_exc = u[:-N_inhib_neurons].reshape(-1,1)
    
    # Update traces
    g_a += dt * (-(g_a/tau_a)+delta_a*S_i) # shape:(605, 1)
    g_b += dt * (-(g_b/tau_b)+delta_b*S_i) # shape:(605, 1)
    u_exc += dt * (((U-u_exc)/tau_f)+U*(1-u_exc)*S_j_exc) # shape:(605, 1)
    x_exc += dt * (((1-x_exc)/tau_d) - u_exc*x_exc*S_j_exc) # shape:(605, 1) 
    
    # Update transmitter channels
    g_ampa += dt * (-(g_ampa/tau_ampa)+np.sum(np.dot(w_ij_exc.T,u_exc)*np.dot(x_exc, S_j_exc))) 
    g_nmda += tau_nmda * dt * (g_ampa-g_nmda)
    g_exc += alpha_e*
    g_gaba
    


    # Update membrane potential
    U = tau_m * dt * ((V_rest - U) + g_exc*(U_exc-U)+(g_gaba + g_a + g_b)*(U_inh - U))

    # Update spiking threshold
    V_th
    
