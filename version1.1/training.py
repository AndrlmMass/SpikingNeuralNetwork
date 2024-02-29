# Loop over each neuron and update membrane potential, GABA, NMDA & AMPA, and weights

def training_network(items, timesteps, dt, Vm, StimE_Ws, EE_EI_Ws, II_IE_Ws):
    '''
    Variables explanation:
    'items' are each datapoint with its respective dimensions (variables)
    'timesteps' are the number of time units it takes to encode an item
    'dt' is the timeunit that dictates the pace of time or change
    'Vm' is the membrane potential of a neuron
    'StimeE_Ws' are the weights between the input and excitatory neurons
    'EE_EI_Ws' are the weights between excitatory to excitatory and between excitatory to inhibitory neurons
    'II_IE_Ws' are the weights between inhibitory to inhibitory and between inhibitory to excitatory neurons

    '''

    # Define length of looping

    for j in range(items.shape):
        for u in range(timesteps):
            # Loop over StimE weights
            for t in range(StimE_Ws):
                # Calculate membrane potential

                # Update spike-trace and spikes 

                # Update weights using hetero -and homostasis based learning rules

            # Loop over EE & EI weights 
            for l in range(EE_EI_Ws):
                # Calculate membrane potential

                # Update spike-trace and spikes 
                
                # Update weights using hetero -and homostasis based learning rules

            # Loop over II & IE weights 
            for p in range(II_IE_Ws):
                # Calculate membrane potential

                # Update spike-trace and spikes 
                
                # Update weights using hetero -and homostasis based learning rules
        
        # Implement sleep or some other homeostatic normalization method

        