# Create script to train network
import numpy as np


def train_data(
    self,
    N_excit_neurons: int,
    N_inhib_neurons: int,
    N_input_neurons: int,
    MemPot: np.ndarray,
    W_se: np.ndarray,
    W_ee: np.ndarray,
    W_ei: np.ndarray,
    W_ie: np.ndarray,
):
    self.num_neurons = (
        self.N_excit_neurons + self.N_inhib_neurons + self.N_input_neurons
    )
    self.spikes = np.zeros((self.time, self.num_neurons))
    self.pre_synaptic_trace = np.zeros(
        (self.time, self.num_neurons - self.N_input_neurons)
    )
    self.post_synaptic_trace = np.zeros(
        (self.time, self.num_neurons - self.N_input_neurons)
    )

    # Add input data before training for input neurons
    self.spikes[:, : self.N_input_neurons] = self.training_data

    # Loop through time and update membrane potential, spikes and weights
    for t in tqdm(range(1, self.time), desc="Training network"):

        # Decay traces
        self.pre_synaptic_trace *= np.exp(-self.dt / self.tau_m)
        self.post_synaptic_trace *= np.exp(-self.dt / self.tau_m)

        # Update stimulation-excitation, excitation-excitation and inhibitory-excitatory synapses
        for n in range(self.N_excit_neurons):

            # Update incoming spikes as I_in
            I_in = (
                np.dot(
                    self.W_se[:, n],
                    self.spikes[t, : self.N_input_neurons],
                )
                + np.dot(
                    self.W_ee[:, n],
                    self.spikes[
                        t,
                        self.N_input_neurons
                        + 1 : self.N_input_neurons
                        + self.N_excit_neurons,
                    ],
                )
                + np.dot(
                    self.W_ie[:, n],
                    self.spikes[
                        t, n + self.N_input_neurons + self.N_excit_neurons + 1 :
                    ],
                )
            )

            # Update membrane potential based on I_in
            self.MemPot[t, n] = (
                self.MemPot[t - 1, n]
                + (-self.MemPot[t - 1, n] + self.V_rest + self.R * I_in) / self.tau_m
            )

            # Update spikes
            if self.MemPot[t, n] > self.V_th:
                self.spikes[t, n + self.N_input_neurons + 1] = 1
                self.pre_synaptic_trace[t, n + self.N_input_neurons + 1] += 1
                self.post_synaptic_trace[t, n + self.N_input_neurons + 1] += 1
                self.MemPot[t, n] = self.V_reset
            else:
                self.spikes[t, n + self.N_input_neurons + 1] = 0

            # Get all pre-synaptic indices
            pre_syn_indices = np.nonzero(self.W_se[:, n])

            # Loop through each synapse to update strength
            for s in range(pre_syn_indices):

                # Use the current trace values for STDP calculation
                pre_trace = self.pre_synaptic_trace[t, s]
                post_trace = self.post_synaptic_trace[t, n + self.N_input_neurons + 1]

                # Get learning components
                hebb = (
                    self.A * pre_trace * post_trace**2 - self.B * pre_trace * post_trace
                )
                hetero_syn = (
                    -self.beta * (self.W_se[s, n] - self.ideal_w) * post_trace**4
                )
                dopamine_reg = self.delta * pre_trace

                # Assemble components to update weight
                self.W_se[s, n] = hebb + hetero_syn + dopamine_reg

            # Get all pre-synaptic indices
            pre_syn_indices = np.nonzero(self.W_ee[:, n])

            # Loop through each synapse to update strength
            for s in range(pre_syn_indices):
                if s == n:
                    raise UserWarning(
                        "There are self-connections within the W_ee array"
                    )

                # Use the current trace values for STDP calculation
                pre_trace = self.pre_synaptic_trace[t, s + self.N_excit_neurons]
                post_trace = self.post_synaptic_trace[t, n + self.N_excit_neurons]

                # Get learning components
                hebb = (
                    self.A * pre_trace * post_trace**2 - self.B * pre_trace * post_trace
                )
                hetero_syn = (
                    -self.beta * (self.W_ee[s, n] - self.ideal_w) * post_trace**4
                )
                dopamine_reg = self.delta * pre_trace

                # Assemble components to update weight
                self.W_ee[s, n] = hebb + hetero_syn + dopamine_reg

        # Update excitatory-inhibitory weights
        for n in range(self.N_excit_neurons):

            # Update incoming spikes as I_in
            I_in = np.dot(
                self.W_ie[:, n],
                self.spikes[
                    t,
                    self.N_input_neurons
                    + 1 : self.N_input_neurons
                    + self.N_excit_neurons,
                ],
            )
            # Update membrane potential based on I_in
            self.MemPot[t, n + self.N_excit_neurons + 1] = (
                self.MemPot[t - 1, n + self.N_excit_neurons + 1]
                + (
                    -self.MemPot[t - 1, n + self.N_excit_neurons + 1]
                    + self.V_rest
                    + self.R * I_in
                )
                / self.tau_m
            )

            # Update spikes
            if self.MemPot[t, n + self.N_excit_neurons + 1] > self.V_th:
                self.spikes[t, n + self.N_input_neurons + self.N_excit_neurons + 1] = 1
                self.pre_synaptic_trace[
                    t, n + self.N_input_neurons + self.N_excit_neurons + 1
                ] += 1
                self.post_synaptic_trace[
                    t, n + self.N_input_neurons + self.N_excit_neurons + 1
                ] += 1
                self.MemPot[t, n + self.N_excit_neurons + 1] = self.V_reset
            else:
                self.spikes[t, n + self.N_input_neurons + self.N_excit_neurons + 1] = 0

        # Update excitatory-inhibitory weights
        for n in range(self.N_inhib_neurons):

            # Update incoming spikes as I_in
            I_in = np.dot(
                self.W_ei[:, n],
                self.spikes[t, self.N_input_neurons + self.N_excit_neurons + 1 :],
            )
            # Update membrane potential based on I_in
            self.MemPot[t, n + self.N_excit_neurons] = (
                self.MemPot[t - 1, n + self.N_excit_neurons]
                + (
                    -self.MemPot[t - 1, n + self.N_excit_neurons]
                    + self.V_rest
                    + self.R * I_in
                )
                / self.tau_m
            )

            # Update spikes
            if self.MemPot[t, n] > self.V_th:
                self.spikes[t, n + self.N_input_neurons + self.N_excit_neurons] = 1
                self.pre_synaptic_trace[
                    t, n + self.N_input_neurons + self.N_excit_neurons
                ] += 1
                self.post_synaptic_trace[
                    t, n + self.N_input_neurons + self.N_excit_neurons
                ] += 1
                self.MemPot[t, n + self.N_excit_neurons] = self.V_reset
            else:
                self.spikes[t, n + self.N_input_neurons + self.N_excit_neurons] = 0

    return (
        spikes,
        MemPot,
        W_se,
        W_ee,
        W_ei,
        W_ie,
        pre_synaptic_trace,
        post_synaptic_trace,
    )
