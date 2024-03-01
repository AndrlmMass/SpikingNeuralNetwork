# Loop over each neuron and update membrane potential, GABA, NMDA & AMPA, and weights


def training_network(
    time,
    dt,
    tau_m,
    Vm,
    StimE_Ws,
    EE_EI_Ws,
    II_IE_Ws,
    neuronal_layers,
    g_ampa,
    g_nmda,
    g_gaba,
    tau_nmda,
    tau_ampa,
    tau_gaba,
):
    """
    Variables explanation:
    'items' are each datapoint with its respective dimensions (variables)
    'timesteps' are the number of time units it takes to encode an item
    'dt' is the timeunit that dictates the pace of time or change
    'Vm' is the membrane potential of a neuron
    'StimeE_Ws' are the weights between the input and excitatory neurons
    'EE_EI_Ws' are the weights between excitatory to excitatory and between excitatory to inhibitory neurons
    'II_IE_Ws' are the weights between inhibitory to inhibitory and between inhibitory to excitatory neurons

    """

    # Loop through

    for j in range(1, time.shape):
        # EXCITATORY NEURONS

        # Update conductance
        g_ampa[j] = g_ampa[j - 1] * dt / tau_ampa
        g_nmda[j] = g_nmda[j - 1] * dt / tau_nmda
        g_gaba[j] = g_gaba[j - 1] * dt / tau_gaba

        # INHIBITORY NEURONS

        for layer in range(neuronal_layers):

            # Loop over StimE weights
            for t in range(StimE_Ws):

                # Update spike-trace and spikes
                d = 2
                # Update weights using hetero -and homostasis based learning rules

            # Loop over EE & EI weights
            for l in range(EE_EI_Ws):
                # Calculate membrane potential
                d = 1
                # Update spike-trace and spikes

                # Update weights using hetero -and homostasis based learning rules

            # Loop over II & IE weights
            for p in range(II_IE_Ws):
                # Calculate membrane potential

                # Update spike-trace and spikes
                r = 2
                # Update weights using hetero -and homostasis based learning rules

    # Implement sleep or some other homeostatic normalization method
