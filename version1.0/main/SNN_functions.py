# SNN functions script

# Import libraries
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Set current working directories and add relevant directories to path
if os.path.exists(
    "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork"
):
    os.chdir(
        "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.0"
    )
    sys.path.append(
        "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.0\\gen"
    )
    sys.path.append(
        "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.0\\main"
    )
    sys.path.append(
        "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.0\\plot"
    )
    sys.path.append(
        "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.0\\tool"
    )
else:
    os.chdir(
        "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\version1.0"
    )
    sys.path.append(
        "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\version1.0\\gen"
    )
    sys.path.append(
        "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\version1.0\\main"
    )
    sys.path.append(
        "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\version1.0\\plot"
    )
    sys.path.append(
        "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\version1.0\\tool"
    )

from plot_training import *
from plot_network import *
from gen_weights import *
from train_widget import *
from gen_data import *
from train import *


# Create widget window class
class MainWindow(QMainWindow):
    def __init__(self, main_widget):
        super().__init__()
        self.setCentralWidget(main_widget)
        self.setWindowTitle("Interactive Tool")
        self.setGeometry(100, 100, 800, 600)  # x, y, width, height


# Initialize class variable
class SNN_STDP:
    # Initialize neuron parameters
    def __init__(
        self,
        V_th: float,
        V_reset: float,
        P: int | float,
        C: int,
        R: int,
        tau_plus: float | int,
        tau_minus: float | int,
        tau_slow: float | int,
        tau_m: float | int,
        tau_mm: float | int,
        tau_ht: float | int,
        tau_hom: float | int,
        tau_stdp: float | int,
        tau_H: float | int,
        gamma: float | int,
        learning_rate: float | int,
        num_items: float,
        dt: float,
        T: int,
        wp: float | int,
        V_rest: int,
        max_weight: float | int,
        min_weight: float | int,
        num_epochs: int,
        init_cals: int,
        A: float | int,
        beta: float | int,
        delta: float | int,
        tau_const: float | int,
        euler: int,
    ):
        self.V_th = V_th
        self.V_reset = V_reset
        self.C = C
        self.R = R
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.tau_slow = tau_slow
        self.tau_m = tau_m
        self.tau_mm = tau_mm
        self.tau_ht = tau_ht
        self.tau_hom = tau_hom
        self.tau_stdp = tau_stdp
        self.tau_H = tau_H
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.dt = dt
        self.wp = wp
        self.T = T
        self.A = A
        self.P = P
        self.beta = beta
        self.delta = delta
        self.num_timesteps = int(T / dt)
        self.num_items = num_items
        self.time = self.num_timesteps * self.num_items
        self.V_rest = V_rest
        self.leakage_rate = 1 / self.R
        self.tau_const = tau_const
        self.init_cals = init_cals
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.num_epochs = num_epochs
        self.max_spike_diff = int(self.num_timesteps * 0.1)
        self.euler = euler

    def initialize_network(
        self,
        N_input_neurons: int,
        N_excit_neurons: int,
        N_inhib_neurons: int,
        radius_: int,
        W_ee_prob: float | int,
        retur: bool,
    ):
        self.N_input_neurons = N_input_neurons
        self.N_excit_neurons = N_excit_neurons
        self.N_inhib_neurons = N_inhib_neurons
        self.num_neurons = (
            self.N_excit_neurons + self.N_inhib_neurons + self.N_input_neurons
        )

        # Generate weights
        gws = gen_weights()

        self.W_se, self.W_se_ideal = gws.gen_SE(
            radius=radius_,
            N_input_neurons=self.N_input_neurons,
            N_excit_neurons=self.N_excit_neurons,
            time=self.time,
            basenum=0.1,
        )
        self.W_ee, self.W_ee_ideal = gws.gen_EE(
            N_excit_neurons=self.N_excit_neurons,
            prob=W_ee_prob,
            time=self.time,
            basenum=0.1,
        )
        self.W_ei, self.W_ei_ideal = gws.gen_EI(
            N_excit_neurons=self.N_excit_neurons,
            N_inhib_neurons=self.N_inhib_neurons,
            time=self.time,
            weight_val=0.2,
            prob=0.1,
        )
        self.W_ie, self.W_ie_ideal = gws.gen_IE(
            N_inhib_neurons=self.N_inhib_neurons,
            N_excit_neurons=self.N_excit_neurons,
            W_ei=self.W_ei,
            time=self.time,
            N_ws=4,
            weight_val=0.1,
            radius=10,
        )

        # Generate membrane potential and spikes array
        self.MemPot = np.zeros(
            (self.time, (self.N_excit_neurons + self.N_inhib_neurons))
        )
        self.MemPot[0, :] = self.V_rest
        self.spikes = np.zeros((self.time, self.num_neurons))

        if retur:
            return (
                self.MemPot,
                self.spikes,
                self.W_se,
                self.W_ee,
                self.W_ei,
                self.W_ie,
                self.W_se_ideal,
                self.W_ee_ideal,
                self.W_ei_ideal,
                self.W_ie_ideal,
            )

    def gen_data(
        self,
        N_classes: int,
        noise_rand: bool,
        noise_rand_ls: float | int,
        mean: int | float,
        blank_variance: int | float,
        input_scaler: int | float,
        save: bool,
        retur: bool,
    ):
        # Check if training data already exists
        files = os.listdir("data/training_data/")
        for rand in noise_rand_ls:
            if any(
                f"training_data_{rand}_items_{self.num_items}_.pkl" in file
                for file in files
            ):
                return

        # Create training data since it does not exist already
        for j in range(len(noise_rand_ls)):
            data, labels = gen_float_data_(
                N_classes=N_classes,
                N_input_neurons=self.N_input_neurons,
                items=self.num_items,
                noise_rand=noise_rand,
                noise_variance=noise_rand_ls[j],
                mean=mean,
                blank_variance=blank_variance,
            )

            float_2_pos_spike(
                data=data,
                labels=labels,
                time=self.time,
                timesteps=self.num_timesteps,
                dt=self.dt,
                input_scaler=input_scaler,  # Should be set to 10
                save=save,
                retur=retur,
                rand_lvl=noise_rand_ls[j],
                items=self.num_items,
            )

    def load_data(self, rand_lvl: float | int, retur: bool):
        cur_path = os.getcwd()
        data_path = f"\\data\\training_data\\training_data_{rand_lvl}_items_{self.num_items}_.pkl"
        data_dir = cur_path + data_path

        with open(data_dir, "rb") as openfile:
            self.training_data = pickle.load(openfile)

        label_path = (
            f"\\data\\labels_train\\labels_train_{rand_lvl}_items_{self.num_items}_.pkl"
        )
        label_dir = cur_path + label_path

        with open(label_dir, "rb") as openfile:
            self.labels_train = pickle.load(openfile)

        if retur:
            return self.training_data, self.labels_train

    def train_data(
        self,
        retur: bool,
        save_model: bool,
        item_lim: int,
    ):
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
            self.slow_synaptic_trace,
            self.I_in_ls,
        ) = train_data(
            R=self.R,
            A=self.A,
            P=self.P,
            w_p=self.wp,  # Defines the upper stable point of weight convergence
            beta=self.beta,
            delta=self.delta,
            time=self.time,
            V_th_=self.V_th,
            V_rest=self.V_rest,
            V_reset=self.V_reset,
            dt=self.dt,
            tau_plus=self.tau_plus,
            tau_minus=self.tau_minus,
            tau_slow=self.tau_slow,
            tau_m=self.tau_m,
            tau_ht=self.tau_ht,
            tau_hom=self.tau_hom,
            tau_stdp=self.tau_stdp,
            tau_H=self.tau_H,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            tau_const=self.tau_const,
            training_data=self.training_data,
            N_excit_neurons=self.N_excit_neurons,
            N_inhib_neurons=self.N_inhib_neurons,
            N_input_neurons=self.N_input_neurons,
            MemPot=self.MemPot,
            max_weight=self.max_weight,
            min_weight=self.min_weight,
            W_se=self.W_se,
            W_se_ideal=self.W_se_ideal,
            W_ee=self.W_ee,
            W_ee_ideal=self.W_ee_ideal,
            W_ei=self.W_ei,
            W_ei_ideal=self.W_ei_ideal,
            W_ie=self.W_ie,
            W_ie_ideal=self.W_ie_ideal,
            save_model=save_model,
            item_lim=item_lim,
            items=self.num_items,
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

    def reload_model(self):
        self.W_se = np.load("model/W_se.npy")
        self.W_ee = np.load("model/W_ee.npy")
        self.W_ie = np.load("model/W_ie.npy")
        self.W_ei = np.load("model/W_ei.npy")
        self.spikes = np.load("model/spikes.npy")
        self.MemPot = np.load("model/MemPot.npy")
        self.pre_trace = np.load("model/pre_synaptic_trace.npy")
        self.post_trace = np.load("model/post_synaptic_trace.npy")
        self.slow_trace = np.load("model/slow_pre_synaptic_trace.npy")

    def plot_training(
        self,
        ws_nd_spikes: bool,
        idx_start: int,
        idx_stop: int,
        mv: bool,
        overlap: bool,
        traces: bool,
    ):
        if ws_nd_spikes:
            plot_weights_and_spikes(
                spikes=self.spikes,
                W_se=self.W_se,
                W_ee=self.W_ee,
                W_ie=self.W_ie,
            )
        if mv:
            plot_membrane_activity(
                MemPot=self.MemPot,
                idx_start=idx_start,
                idx_stop=idx_stop,
            )
        if overlap:
            plot_clusters(
                spikes=self.spikes,
                labels=self.labels_train,
                N_input_neurons=self.N_input_neurons,
                N_excit_neurons=self.N_excit_neurons,
                N_inhib_neurons=self.N_inhib_neurons,
            )
        if traces:
            plot_traces(
                pre_synaptic_trace=self.pre_trace,
                post_synaptic_trace=self.post_trace,
                slow_pre_synaptic_trace=self.slow_trace,
                N_input_neurons=self.N_input_neurons,
            )

    def plot_I_in(self):
        d = np.arange(0, len(self.I_in_ls))
        plt.plot(d, self.I_in_ls)
        plt.show()
