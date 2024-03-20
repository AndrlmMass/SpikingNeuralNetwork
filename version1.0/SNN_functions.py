# SNN functions script

# Import libraries
import os
import pickle
import numpy as np
from tqdm import tqdm

# os.chdir("C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork")
os.chdir(
    "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Github\\BONSAI\\SpikingNeuralNetwork\\version1.0"
)

from plot_training import *
from plot_network import *
from gen_weights import *


# Initialize class variable
class SNN_STDP:
    # Initialize neuron parameters
    def __init__(
        self,
        V_th: float,
        V_reset: float,
        C: int,
        R: int,
        tau_m: float,
        num_items: float,
        num_input_neurons: float,
        tau_stdp: float,
        dt: float,
        T: int,
        V_rest: int,
        excit_inhib_ratio: float,
        alpha: float | int,
        max_weight: float | int,
        min_weight: float | int,
        input_scaler: float | int,
        num_epochs: int,
        init_cals: int,
        target_weight: float,
        A: float | int,
        B: float | int,
        beta: float | int,
    ):
        self.V_th = V_th
        self.V_reset = V_reset
        self.C = C
        self.R = R
        self.tau_m = tau_m
        self.tau_stdp = tau_stdp
        self.dt = dt
        self.T = T
        self.A = A
        self.B = B
        self.beta = beta
        self.target_weight = target_weight
        self.num_timesteps = int(T / dt)
        self.time = self.num_timesteps * self.num_items
        self.V_rest = V_rest
        self.leakage_rate = 1 / self.R
        self.num_items = num_items
        self.num_classes = num_input_neurons
        self.excit_inhib_ratio = excit_inhib_ratio
        self.alpha = alpha / (self.num_neurons * 0.1)
        self.init_cals = init_cals
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.input_scaler = input_scaler
        self.num_epochs = num_epochs
        self.max_spike_diff = int(self.num_timesteps * 0.1)

    def initialize_network(
        self,
        N_input_neurons: int,
        N_excit_neurons: int,
        N_inhib_neurons: int,
        radius: int,
        prob: float | int,
        retur: bool,
    ):
        self.N_input_neurons = N_input_neurons
        self.N_excit_neurons = N_excit_neurons
        self.N_inhib_neurons = N_inhib_neurons

        # Generate weights
        gws = gen_weights()

        self.W_se = gws.gen_SE(
            radius=radius,
            N_input_neurons=self.N_input_neurons,
            N_excit_neurons=self.N_excit_neurons,
        )
        self.W_ee = gws.gen_EE(
            N_excit_neurons=self.N_excit_neurons, prob=prob, time=self.time
        )
        self.W_ei = gws.gen_EI(
            N_excit_neurons=self.N_excit_neurons,
            N_inhib_neurons=self.N_inhib_neurons,
            time=self.time,
        )
        self.W_ie = gws.gen_IE(
            N_inhib_neurons=self.N_inhib_neurons,
            N_excit_neurons=self.N_excit_neurons,
            W_ei=self.W_ei,
            prob=prob,
            time=self.time,
        )

        # Generate membrane potential and spikes array
        self.MemPot = np.zeros((self.time, self.num_neurons - self.N_input_neurons))
        self.MemPot[0, :] = self.V_rest
        self.spikes = np.ones((self.time, self.num_neurons))

        if retur:
            return (
                self.MemPot,
                self.t_since_spike,
                self.W_se,
                self.W_ee,
                self.W_ei,
                self.W_ie,
            )

    def load_data(self, rand_lvl, retur):
        with open(
            f"/data/training_data/training_data_{rand_lvl}.pkl", "rb"
        ) as openfile:
            self.training_data = pickle.load(openfile)

        with open(f"/data/testing_data/testing_data_{rand_lvl}.pkl", "rb") as openfile:
            self.testing_data = pickle.load(openfile)

        with open(f"/data/labels_train/labels_train_{rand_lvl}.pkl", "rb") as openfile:
            self.labels_train = pickle.load(openfile)

        with open(f"/data/labels_test/labels_test_{rand_lvl}.pkl", "rb") as openfile:
            self.labels_test = pickle.load(openfile)

        if retur:
            return (
                self.training_data,
                self.testing_data,
                self.labels_train,
                self.labels_test,
            )

    def train_data(self):
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
                    + (-self.MemPot[t - 1, n] + self.V_rest + self.R * I_in)
                    / self.tau_m
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
                    post_trace = self.post_synaptic_trace[
                        t, n + self.N_input_neurons + 1
                    ]

                    # Get learning components
                    hebb = (
                        self.A * pre_trace * post_trace**2
                        - self.B * pre_trace * post_trace
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
                        self.A * pre_trace * post_trace**2
                        - self.B * pre_trace * post_trace
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
                    self.spikes[
                        t, n + self.N_input_neurons + self.N_excit_neurons + 1
                    ] = 1
                    self.pre_synaptic_trace[
                        t, n + self.N_input_neurons + self.N_excit_neurons + 1
                    ] += 1
                    self.post_synaptic_trace[
                        t, n + self.N_input_neurons + self.N_excit_neurons + 1
                    ] += 1
                    self.MemPot[t, n + self.N_excit_neurons + 1] = self.V_reset
                else:
                    self.spikes[
                        t, n + self.N_input_neurons + self.N_excit_neurons + 1
                    ] = 0

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
