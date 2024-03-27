    def train_data(self):
        self.spikes = np.zeros((self.time, self.num_neurons))
        
        # Add input data before training for input neurons
        self.spikes[:,:self.N_input_neurons] = self.training_data

        count = 0
        avg_IN = []
        for l in tqdm(range(self.time), desc="Training network"):
                # Decay traces
                self.pre_synaptic_trace *= np.exp(-self.dt / self.tau_m)
                self.post_synaptic_trace *= np.exp(-self.dt / self.tau_m)
                
                for n in range(self.num_neurons):

                    # Check if neuron is an input neuron
                    if n not in self.input_neuron_idx:
                        spikes = (self.t_since_spike[t - 1, :, l] == 0).astype(int)
                        I_in = np.dot(self.weights[n, :, l], spikes.T)
                        In.append(I_in)

                        # Update equation
                        self.MemPot[t, n, l] = (
                            self.MemPot[t - 1, n, l]
                            + (-self.MemPot[t - 1, n, l] + self.V_rest + self.R * I_in)
                            / self.tau_m
                        )

                        # Update spikes
                        if self.MemPot[t, n, l] > self.V_th:
                            self.pre_synaptic_trace[n, l] += 1
                            self.post_synaptic_trace[n, l] += 1
                            self.t_since_spike[t, n, l] = 0
                            self.MemPot[t, n, l] = self.V_reset
                            spike_counts[n, l] += 1
                        else:
                            self.t_since_spike[t, n, l] = (
                                self.t_since_spike[t - 1, n, l] + 1
                            )
                    # Perform trace-based STDP for hidden neurons
                    for s in range(self.num_neurons):
                        if s != n and self.weights[n, s, l] != 0:

                            # Use the current trace values for STDP calculation
                            pre_trace = self.pre_synaptic_trace[s, l]
                            post_trace = self.post_synaptic_trace[n, l]

                            # Calculate weight change based on traces
                            if self.weights[n, s, l] > 0:  # Excitatory synapse
                                weight_change = self.A_plus * pre_trace * post_trace
                                self.weights[n, s, l] += round(weight_change, 4)

                            elif self.weights[n, s, l] < 0:  # Inhibitory synapse
                                weight_change = self.A_minus * pre_trace * post_trace

                                self.weights[n, s, l] += round(weight_change, 4)

                        # Enforce minimum and maximum synaptic weight
                        self.weights[n, s, l] = np.clip(
                            self.weights[n, s, l],
                            self.min_weight,
                            self.max_weight,
                        )

                        # Add weight normalization
                        norm = np.dot(self.weights[n, :, l].T, self.weights[n, :, l])
                        if norm != 0:
                            self.weights[n, :, l] = np.round(
                                self.weights[n, :, l] / math.sqrt(norm), 4
                            )

            # Update self.MemPot to include the membrane potential from the previous step
            # as the beginning of the next step.
            if l < self.num_items - 1:
                self.t_since_spike[0, :, l + 1] = self.t_since_spike[t, :, l]
                self.MemPot[0, :, l + 1] = self.MemPot[t, :, l]
                self.weights[:, :, l + 1] = self.weights[:, :, l]

            avg_IN.append(np.average(In))
        
                # Convert t_since_spike to spike_array for visualization purposes
        self.spike_array = np.where(self.t_since_spike == 0, 1, 0)

        # Summarize the training
        print(
            f"This training had {count} excitatory and inhibitory changes in weights. On average, each item had the following I_in values: {avg_IN}"
        )

import numpy as np

d = np.array([1,1,0,0,1])
print(np.nonzero(d)[0])