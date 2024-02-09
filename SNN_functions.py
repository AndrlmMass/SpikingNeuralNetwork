# SNN functions script

# Import libraries
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

os.chdir("C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork")
#os.chdir(
#    "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Github\\BONSAI\\SpikingNeuralNetwork"
#)

import plot_training as pt
import plot_network as pn
import Gen_weights_nd_data as gwd


# Initialize class variable
class SNN_STDP:
    # Initialize neuron parameters
    def __init__(
        self,
        V_th=0.7,
        V_reset=-0.2,
        C=10,
        R=1,
        A_minus_ex=-0.01,
        A_minus_in=-1,
        tau_m=0.002,
        num_items=100,
        num_input_neurons=4,
        tau_stdp=0.1,
        A_plus_ex=1,
        A_plus_in=0.01,
        dt=0.001,
        T=1,
        V_rest=0,
        num_neurons=20,
        excit_inhib_ratio=0.8,
        alpha=1,
        max_weight=1,
        min_weight=-1,
        input_scaler=100,
        num_epochs=10,
        init_cals=700,
        num_classes=2,
    ):
        self.V_th = V_th
        self.V_reset = V_reset
        self.C = C
        self.R = R
        self.tau_m = tau_m
        self.tau_stdp = tau_stdp
        self.dt = dt
        self.T = T
        self.num_timesteps = int(T / dt)
        self.V_rest = V_rest
        self.A_minus_ex = A_minus_ex
        self.A_plus_ex = A_plus_ex
        self.A_minus_in = A_minus_in
        self.A_plus_in = A_plus_in
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
        self.num_input_neurons = num_input_neurons
        self.num_epochs = num_epochs
        self.max_spike_diff = int(self.num_timesteps * 0.1)
        self.num_classes = num_classes
        self.pre_synaptic_trace = np.zeros((self.num_neurons, self.num_items))
        self.post_synaptic_trace = np.zeros((self.num_neurons, self.num_items))


    def initialize_network(self):
        # Generate weights
        self.weights, self.input_neuron_idx = (
            gwd.generate_small_world_network_power_law(
                num_neurons=self.num_neurons,
                excit_inhib_ratio=self.excit_inhib_ratio,
                alpha=self.alpha,
                num_input_neurons=self.num_input_neurons,
                num_items=self.num_items,
                num_timesteps=self.num_timesteps,
            )
        )
        # Generate membrane potential and spikes array
        self.MemPot = np.zeros((self.num_timesteps, self.num_neurons, self.num_items))
        self.MemPot[0, :, :] = self.V_rest
        self.t_since_spike = np.ones(
            (self.num_timesteps, self.num_neurons, self.num_items)
        )
        print(
            f"Sim between 1 and 2: {np.mean(self.weights[:,:,0]==self.weights[:,:,1])}"
        )
        return self.MemPot, self.t_since_spike, self.weights

    def prepping_data(
        self,
        base_mean,
        mean_increment,
        variance,
    ):
        # Simulate data
        self.data, self.classes = gwd.generate_multidimensional_data(
            self.num_classes,
            base_mean,
            mean_increment,
            variance,
            int(self.num_items / self.num_classes),
            self.num_input_neurons,
            self.num_timesteps,
            self.dt,
            self.input_scaler,
        )
        print(self.data.shape, self.classes.shape)
        return self.data, self.classes

    def find_prev_spike(self, t, s, l):
        for j in range(t, 0, -1):
            if self.t_since_spike[j, s, l] == 0:
                return j
        return 0

    def neuronal_activity(self):
        spike_counts = np.zeros((self.num_neurons, self.num_items))
        # Add input data before training
        for j in range(len(self.input_neuron_idx)):
            for t in range(1,self.num_timesteps):
                for i in range(self.num_items):
                    if self.data[t, j, i] == 1:
                        self.t_since_spike[t, self.input_neuron_idx[j], i] = 0
                    else:
                        self.t_since_spike[t, self.input_neuron_idx[j], i] = (
                            self.t_since_spike[t - 1, self.input_neuron_idx[j], i] + 1
                        )


        cal_consum = 0
        count = [0, 0, 0, 0]
        for l in tqdm(range(self.num_items), desc="Training network"):
            for t in range(1, self.num_timesteps):
                # Decay traces
                self.pre_synaptic_trace *= np.exp(-self.dt / self.tau_m)
                self.post_synaptic_trace *= np.exp(-self.dt / self.tau_m)
                for n in range(self.num_neurons):
                    # Check if neuron is an input neuron
                    if n not in self.input_neuron_idx:
                        spikes = (self.t_since_spike[t - 1, :, l] == 0).astype(int)
                        I_in = np.dot(self.weights[n, :, l], spikes.T)

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
                            cal_consum += 0.1
                            self.t_since_spike[t, n, l] = 0
                            self.MemPot[t, n, l] = self.V_reset
                            spike_counts[n, l] += 1
                        else:
                            self.t_since_spike[t, n, l] = (
                                self.t_since_spike[t - 1, n, l] + 1
                            )

                    # Perform trace-based STDP for hidden neurons
                    for s in range(self.num_neurons):
                        if s != n and self.weights[n,s,l] != 0:
                            
                            # Use the current trace values for STDP calculation
                            pre_trace = self.pre_synaptic_trace[s, l]
                            post_trace = self.post_synaptic_trace[n, l]

                            # Calculate weight change based on traces
                            if self.weights[n, s, l] > 0:  # Excitatory synapse
                                weight_change = self.A_plus_ex * pre_trace * post_trace
                                self.weights[n, s, l] += round(weight_change, 4)
                            elif self.weights[n, s, l] < 0:  # Inhibitory synapse
                                weight_change = self.A_minus_in * pre_trace * post_trace
                                self.weights[n, s, l] -= round(weight_change, 4)

                            # Enforce minimum and maximum synaptic weight
                            self.weights[n, s, l] = np.clip(self.weights[n, s, l], self.min_weight, self.max_weight)

            # Update self.MemPot to include the membrane potential from the previous step
            # as the beginning of the next step.
            if l < self.num_items - 1:
                self.t_since_spike[0, :, l + 1] = self.t_since_spike[t, :, l]
                self.MemPot[0, :, l + 1] = self.MemPot[t, :, l]
                self.weights[:, :, l + 1] = self.weights[:, :, l]

        # Convert t_since_spike to spike_array for visualization purposes
        self.spike_array = np.where(self.t_since_spike == 0, 1, 0)

        # Summarize the training
        print(
            f"This training had {count[0]} excitatory strengthenings and {count[1]} weakenings. While inhibitory connections had {count[3]} strenghtenings and {count[2]} weakenings."
        )

    def visualize_network(self, drw_edg=True, drw_netw=True):
        if drw_netw:
            pn.draw_network(self.weights)
        if drw_edg:
            pn.draw_edge_distribution(self.weights)

    def plot_training(self, num_neurons=None, num_items=None, neuro_id=None, num_items_sec=None):
        if num_neurons == None:
            num_neurons = self.num_neurons
        if num_items == None:
            num_items = self.num_items

        #pt.plot_membrane_activity(MemPot=self.MemPot, neuron_ids=neuro_id, num_items=num_items_sec)

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
        pt.plot_relative_activity(spikes=self.spike_array, classes=self.classes, input_idx=self.input_neuron_idx, num_neurons=self.num_neurons)

