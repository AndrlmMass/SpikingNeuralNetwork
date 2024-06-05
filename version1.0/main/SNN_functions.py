# SNN functions script

# Import libraries
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Set the current directory based on the existence of a specific path
if os.path.exists(
    "C:\\Users\\Bruker\\OneDrive\\Documents\\NMBU_\\BONSAI\\SNN\\SpikingNeuralNetwork\\version1.0"
):
    base_path = "C:\\Users\\Bruker\\OneDrive\\Documents\\NMBU_\\BONSAI\\SNN\\SpikingNeuralNetwork\\version1.0"
else:
    base_path = "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\version1.0"

os.chdir(base_path)
sys.path.append(os.path.join(base_path, "gen"))
sys.path.append(os.path.join(base_path, "main"))
sys.path.append(os.path.join(base_path, "plot"))
sys.path.append(os.path.join(base_path, "tool"))

from plot_training import *
from plot_network import *
from gen_weights import *
from train_widget import *
from plot_data import *
from gen_data import *
from train import *


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
        tau_thr: float | int,
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
        self.tau_thr = tau_thr
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

        self.W_se, self.W_se_ideal, self.W_se_2d, self.W_se_plt_idx = gws.gen_SE(
            radius=radius_,
            N_input_neurons=self.N_input_neurons,
            N_excit_neurons=self.N_excit_neurons,
            time=self.time,
            basenum=0.1,
        )
        self.W_ee, self.W_ee_ideal, self.W_ee_2d, self.W_ee_plt_idx = gws.gen_EE(
            N_excit_neurons=self.N_excit_neurons,
            prob=W_ee_prob,
            time=self.time,
            basenum=0.1,
        )
        self.W_ei, self.W_ei_ideal, self.W_ei_2d, self.W_ei_plt_idx = gws.gen_EI(
            N_excit_neurons=self.N_excit_neurons,
            N_inhib_neurons=self.N_inhib_neurons,
            time=self.time,
            prob=0.1,
        )
        self.W_ie, self.W_ie_ideal, self.W_ie_2d, self.W_ie_plt_idx = gws.gen_IE(
            N_inhib_neurons=self.N_inhib_neurons,
            N_excit_neurons=self.N_excit_neurons,
            W_ei=self.W_ei,
            radius=10,
            time=self.time,
            N_ws=4,
            weight_val=0.1,
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

    def vis_network(self, heatmap: bool, weight_layer: bool):
        if heatmap:
            plot_input_space(self.W_se)
        if weight_layer:
            W_se_ = np.sum(self.W_se, axis=1).reshape(
                int(np.sqrt(self.N_input_neurons)), int(np.sqrt(self.N_input_neurons))
            )
            draw_weights_layer(
                weights=W_se_,
                title="Input space",
                xlabel="Input Neurons",
                ylabel="Input Neurons",
            )
            W_es_ = np.sum(self.W_se, axis=0).reshape(
                int(np.sqrt(self.N_input_neurons)), int(np.sqrt(self.N_input_neurons))
            )
            draw_weights_layer(
                weights=W_es_,
                title="Excitatory space",
                xlabel="Excitatory Neurons",
                ylabel="Excitatory Neurons",
            )
            W_ee_ = np.sum(self.W_ee, axis=1).reshape(
                int(np.sqrt(self.N_excit_neurons)), int(np.sqrt(self.N_excit_neurons))
            )
            draw_weights_layer(
                weights=W_ee_,
                title="Excitatory space",
                xlabel="Excitatory neuron",
                ylabel="Excitatory neuron",
            )
            W_ie_ = np.sum(self.W_ie, axis=0).reshape(
                int(np.sqrt(self.N_excit_neurons)), int(np.sqrt(self.N_excit_neurons))
            )
            draw_weights_layer(
                weights=W_ie_,
                title="Excitatory space",
                xlabel="Excitatory neuron",
                ylabel="Excitatory neuron",
            )
            W_

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
        self.N_classes = N_classes
        # Check if training data already exists
        files = os.listdir("data/training_data/")
        for rand in noise_rand_ls:
            if any(
                f"training_data_{rand}_items_{self.num_items}_.npy" in file
                for file in files
            ):
                return

        # Create training data since it does not exist already
        for j in range(len(noise_rand_ls)):
            self.data, self.labels = gen_float_data_(
                N_classes=N_classes,
                N_input_neurons=self.N_input_neurons,
                items=self.num_items,
                noise_rand=noise_rand,
                noise_variance=noise_rand_ls[j],
                mean=mean,
                blank_variance=blank_variance,
                save=save,
            )

            float_2_pos_spike(
                data=self.data,
                labels=self.labels,
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
        self.training_data = np.load(
            f"data\\training_data\\training_data_{rand_lvl}_items_{self.num_items}_.npy"
        )
        self.data = np.load(
            f"data\\training_data_float\\training_data_items_{self.num_items}_.npy"
        )
        self.labels_train = np.load(
            f"data\\labels_train\\labels_train_{rand_lvl}_items_{self.num_items}_.npy"
        )
        self.labels = np.load(
            f"data\\labels_train_float\\labels_train_{self.num_items}_.npy"
        )

        if retur:
            return (
                self.training_data,
                self.labels_train,
            )

    def visualize_data(self, run):
        if run:
            input_space_plotted_single(self.data[0])

    def train_data(
        self,
        retur: bool,
        save_model: bool,
        item_lim: int,
    ):
        (
            self.spikes,
            self.MemPot,
            self.W_se_2d,
            self.W_se_ideal,
            self.W_ee_2d,
            self.W_ee_ideal,
            self.W_ei_2d,
            self.W_ei_ideal,
            self.W_ie_2d,
            self.W_ie_ideal,
            self.pre_synaptic_trace,
            self.post_synaptic_trace,
            self.slow_synaptic_trace,
            self.z_istdp,
            self.I_in_ls,
            self.V_th_array,
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
            tau_thr=self.tau_thr,
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
            W_se_2d=self.W_se_2d,
            W_se_plt_idx=self.W_se_plt_idx,
            W_ee=self.W_ee,
            W_ee_ideal=self.W_ee_ideal,
            W_ee_2d=self.W_ee_2d,
            W_ee_plt_idx=self.W_ee_plt_idx,
            W_ei=self.W_ei,
            W_ei_ideal=self.W_ei_ideal,
            W_ei_2d=self.W_ei_2d,
            W_ei_plt_idx=self.W_ei_plt_idx,
            W_ie=self.W_ie,
            W_ie_ideal=self.W_ie_ideal,
            W_ie_2d=self.W_ie_2d,
            W_ie_plt_idx=self.W_ie_plt_idx,
            save_model=save_model,
            item_lim=item_lim,
            items=self.num_items,
        )

        if retur:
            return (
                self.spikes,
                self.MemPot,
                self.W_se_2d,
                self.W_ee_2d,
                self.W_ei_2d,
                self.W_ie_2d,
            )

    def plot_training(
        self,
        ws_nd_spikes: bool,
        idx_start: int,
        idx_stop: int,
        mv: bool,
        overlap: bool,
        traces: bool,
        tsne: bool,
    ):
        if ws_nd_spikes:
            plot_weights_and_spikes(
                spikes=self.spikes,
                W_se=self.W_se_2d,
                W_ee=self.W_ee_2d,
                W_ie=self.W_ie_2d,
            )
        if mv:
            plot_membrane_activity(
                MemPot=self.MemPot,
                MemPot_th=self.V_th_array,
                idx_start=idx_start,
                idx_stop=idx_stop,
                time=self.time,
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
                pre_synaptic_trace=self.pre_synaptic_trace,
                post_synaptic_trace=self.post_synaptic_trace,
                slow_pre_synaptic_trace=self.slow_synaptic_trace,
                N_input_neurons=self.N_input_neurons,
            )
        if tsne:
            t_SNE(
                self.N_classes,
                self.spikes,
                self.labels,
                self.labels_train,
                self.num_timesteps,
                self.N_input_neurons,
            )

    def plot_I_in(self):
        d = np.arange(0, len(self.I_in_ls))
        plt.plot(d, self.I_in_ls)
        plt.show()
