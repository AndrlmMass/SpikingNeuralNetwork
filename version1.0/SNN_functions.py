# SNN functions script

# Import libraries
import os
import pickle
import numpy as np
from tqdm import tqdm

# os.chdir("C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork")
os.chdir(
    "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Github\\BONSAI\\SpikingNeuralNetwork"
)

from plot_training import *
from plot_network import *
from gen_weights import *


# Initialize class variable
class SNN_STDP:
    # Initialize neuron parameters
    def __init__(
        self,
        V_th=0.7,
        V_reset=-0.2,
        C=10,
        R=1,
        A_minus=-0.01,
        tau_m=0.002,
        num_items=100,
        num_input_neurons=4,
        tau_stdp=0.1,
        A_plus=1,
        dt=0.001,
        T=1,
        V_rest=0,
        num_neurons=20,
        excit_inhib_ratio=0.8,
        alpha=1,
        max_weight=1,
        min_weight=-1,
        input_scaler=0.25,
        num_epochs=10,
        init_cals=700,
        num_classes=2,
        target_value=3,
    ):
        self.V_th = V_th
        self.V_reset = V_reset
        self.C = C
        self.R = R
        self.tau_m = tau_m
        self.tau_stdp = tau_stdp
        self.dt = dt
        self.T = T
        self.target_value = target_value
        self.num_timesteps = int(T / dt)
        self.time = self.num_timesteps * self.num_items
        self.V_rest = V_rest
        self.A_minus = A_minus
        self.A_plus = A_plus
        self.leakage_rate = 1 / self.R
        self.num_items = num_items
        self.num_neurons = num_neurons
        self.num_classes = num_input_neurons
        self.excit_inhib_ratio = excit_inhib_ratio
        self.alpha = alpha / (self.num_neurons * 0.1)
        self.init_cals = init_cals
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.input_scaler = input_scaler
        self.num_epochs = num_epochs
        self.max_spike_diff = int(self.num_timesteps * 0.1)
        self.pre_synaptic_trace = np.zeros((self.num_neurons, self.time))
        self.post_synaptic_trace = np.zeros((self.num_neurons, self.time))

    def initialize_network(
        self, N_input_neurons, N_excit_neurons, N_inhib_neurons, radius, prob
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
        self.MemPot = np.zeros((self.time, self.num_neurons))
        self.MemPot[0, :, :] = self.V_rest
        self.spikes = np.ones((self.time, self.num_neurons))

        return (
            self.MemPot,
            self.t_since_spike,
            self.W_se,
            self.W_ee,
            self.W_ei,
            self.W_ie,
        )

    def load_data(self, rand_lvl):
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

        return (
            self.training_data,
            self.testing_data,
            self.labels_train,
            self.labels_test,
        )

    def find_prev_spike(self, t, s, l):  # Not sure if this function is still valid
        

    def train_data(self):
        self.spikes = np.zeros((self.time, self.num_neurons))

        # Add input data before training for input neurons
        self.spikes[:, : self.N_input_neurons] = self.training_data

        # Loop through time and update membrane potential, spikes and weights
        for t in tqdm(range(1, self.time), desc="Training network"):

            # Decay traces
            self.pre_synaptic_trace *= np.exp(-self.dt / self.tau_m)
            self.post_synaptic_trace *= np.exp(-self.dt / self.tau_m)

            # Update SE
            for n in range(self.N_excit_neurons):

                # Update incoming spikes as I_in
                I_in = np.dot(self.W_se[:, n], self.spikes[t, : self.N_input_neurons])

                # Update membrane potential based on I_in
                self.MemPot[t, n] = (
                    self.MemPot[t - 1, n]
                    + (-self.MemPot[t - 1, n] + self.V_rest + self.R * I_in)
                    / self.tau_m
                )
                # Update spikes
                if self.MemPot[t, n] > self.V_th:
                    self.spikes[t, n] = 1
                    self.pre_synaptic_trace[n, t] += 1
                    self.post_synaptic_trace[n, t] += 1
                    self.MemPot[t, n] = self.V_reset
                else:
                    self.spikes[t, n] = 0

                # Get all pre-synaptic indices
                pre_syn_indices = np.nonzero(self.W_se[:,n])
                
                # Loop through each synapse to update strength
                for s in range(pre_syn_indices):
                    

    
            # Update EE

            # Update EI

            # Update

    def visualize_network(self, drw_edg=True, drw_netw=True):
        if drw_netw:
            pn.draw_network(self.weights)
        if drw_edg:
            pn.draw_edge_distribution(self.weights)

    def plot_training(self, num_neurons=None, num_items=None, plt_mV=True):
        if num_neurons == None:
            num_neurons = self.num_neurons
        if num_items == None:
            num_items = self.num_items
        if plt_mV:
            pt.plot_membrane_activity(
                MemPot=self.MemPot,
                num_neurons=num_neurons,
                num_items=num_items,
                input_idx=self.input_neuron_idx,
                timesteps=self.num_timesteps,
            )

        pt.plot_spikes(
            num_neurons_to_plot=num_neurons,
            num_items_to_plot=num_items,
            t_since_spike=self.t_since_spike,
            input_indices=self.input_neuron_idx,
        )
        pt.plot_weights(self.weights, dt_items=num_items)

        pt.plot_activity_scatter(
            spikes=self.spike_array, classes=self.classes, num_classes=self.num_classes
        )

        pt.plot_relative_activity(
            spikes=self.spike_array,
            classes=self.classes,
            input_idx=self.input_neuron_idx,
            num_neurons=self.num_neurons,
        )
