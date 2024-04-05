# SNN functions script

# Import libraries
import os
import pickle
import numpy as np

os.chdir(
    "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.0"
)
#os.chdir(
#    "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Github\\BONSAI\\SpikingNeuralNetwork\\version1.0"
#)

# Import other functions
import sys
sys.path.append('C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.0\\plot')
sys.path.append('C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.0\\gen')
sys.path.append('C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.0\\main')

from plot_training import *
from plot_network import *
from gen_weights import *
from train import *


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
        tau_stdp: float,
        num_neurons: int,
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
        self.num_neurons = num_neurons
        self.target_weight = target_weight
        self.num_timesteps = int(T / dt)
        self.num_items = num_items
        self.time = self.num_timesteps * self.num_items
        self.V_rest = V_rest
        self.leakage_rate = 1 / self.R
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
        radius_: int,
        W_ie_prob: float | int,
        W_ee_prob: float | int,
        retur: bool
    ):
        self.N_input_neurons = N_input_neurons
        self.N_excit_neurons = N_excit_neurons
        self.N_inhib_neurons = N_inhib_neurons

        # Generate weights
        gws = gen_weights()

        self.W_se, self.W_se_ideal = gws.gen_SE(
            radius=radius_,
            N_input_neurons=self.N_input_neurons,
            N_excit_neurons=self.N_excit_neurons,
            time=self.time,
            basenum=1
        )
        self.W_ee, self.W_ee_ideal = gws.gen_EE(
            N_excit_neurons=self.N_excit_neurons, prob=W_ee_prob, time=self.time, basenum=1
        )
        self.W_ei, self.W_ei_ideal = gws.gen_EI(
            N_excit_neurons=self.N_excit_neurons,
            N_inhib_neurons=self.N_inhib_neurons,
            time=self.time,
            weight_val=1
        )
        self.W_ie.self.W_ie_ideal = gws.gen_IE(
            N_inhib_neurons=self.N_inhib_neurons,
            N_excit_neurons=self.N_excit_neurons,
            W_ei=self.W_ei,
            time=self.time,
            N_ws=4,
            weight_val=1,
            radius=1
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

    def train_data(self, retur):
        (
            self.spikes,
            self.MemPot,
            self.W_se,
            self.W_se_ideal,
            self.W_ee,
            self.W_ee_ideal,
            self.W_ei,
            self.W_ei_ideal,
            self.W_ie,
            self.W_ie_ideal,
            self.pre_synaptic_trace,
            self.post_synaptic_trace,
            self.num_neurons,
        ) = train_data(
            R=self.R,
            A=self.A,
            B=self.B,
            beta=self.beta,
            delta=self.delta,
            ideal_w=self.ideal_w,
            time=self.time,
            V_th=self.V_th,
            V_rest=self.V_rest,
            V_reset=self.V_reset,
            dt=self.dt,
            tau_m=self.tau_m,
            training_data=self.training_data,
            N_excit_neurons=self.N_excit_neurons,
            N_inhib_neurons=self.N_inhib_neurons,
            N_input_neurons=self.N_input_neurons,
            MemPot=self.MemPot,
            W_se=self.W_se,
            W_se_ideal=self.W_se_ideal,
            W_ee=self.W_ee,
            W_ee_ideal=self.W_ee_ideal,
            W_ei=self.W_ei,
            W_ei_ideal=self.W_ei_ideal,
            W_ie=self.W_ie,
            W_ie_ideal=self.W_ie_ideal,
        )

        if retur:
            return (
                self.spikes,
                self.MemPot,
                self.W_se,
                self.W_ee,
                self.W_ei,
                self.W_ie,
            )
