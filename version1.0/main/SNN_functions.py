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
        U: float | int,
        tau_plus: float | int,
        tau_minus: float | int,
        tau_slow: float | int,
        tau_m: float | int,
        tau_ht: float | int,
        tau_hom: float | int,
        tau_istdp: float | int,
        tau_H: float | int,
        tau_thr: float | int,
        tau_ampa: float | int,
        tau_nmda: float | int,
        tau_gaba: float | int,
        tau_a: float | int,
        tau_b: float | int,
        tau_d: float | int,
        tau_f: float | int,
        delta_a: float | int,
        delta_b: float | int,
        U_exc: float | int,
        U_inh: float | int,
        alpha_exc: float | int,
        alpha_inh: float | int,
        learning_rate: float | int,
        gamma: float | int,
        num_items: float,
        dt: float,
        T: int,
        wp: float | int,
        V_rest: int,
        min_weight: float | int,
        max_weight: float | int,
        num_epochs: int,
        A: float | int,
        B: float | int,
        beta: float | int,
        delta: float | int,
        tau_cons: float | int,
        euler: int,
        U_cons: float | int,
    ):
        self.V_th = V_th
        self.V_reset = V_reset
        self.P = P
        self.C = C
        self.U = U
        self.A = A
        self.B = B
        self.T = T
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.tau_slow = tau_slow
        self.tau_m = tau_m
        self.tau_ht = tau_ht
        self.tau_hom = tau_hom
        self.tau_istdp = tau_istdp
        self.tau_H = tau_H
        self.tau_thr = tau_thr
        self.tau_ampa = tau_ampa
        self.tau_nmda = tau_nmda
        self.tau_gaba = tau_gaba
        self.tau_a = tau_a
        self.tau_b = tau_b
        self.tau_d = tau_d
        self.tau_f = tau_f
        self.delta_a = delta_a
        self.delta_b = delta_b
        self.U_exc = U_exc
        self.U_inh = U_inh
        self.alpha_exc = alpha_exc
        self.alpha_inh = alpha_inh
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.num_items = num_items
        self.dt = dt
        self.wp = wp
        self.V_rest = V_rest
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.num_epochs = num_epochs
        self.beta = beta
        self.delta = delta
        self.tau_cons = tau_cons
        self.euler = euler
        self.num_timesteps = int(T / dt)
        self.time = self.num_timesteps * self.num_items
        self.max_spike_diff = int(self.num_timesteps * 0.1)  # what does this do?
        self.U_cons = U_cons

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
        self.W_inh = gws.gen_EI(
            N_excit_neurons=self.N_excit_neurons,
            N_inhib_neurons=self.N_inhib_neurons,
            time=self.time,
            prob=0.1,
        )
        self.W_ie, self.W_ie_ideal, self.W_ie_2d, self.W_ie_plt_idx = gws.gen_IE(
            N_inhib_neurons=self.N_inhib_neurons,
            N_excit_neurons=self.N_excit_neurons,
            W_ei=self.W_inh,
            radius=10,
            time=self.time,
            N_ws=4,
            weight_val=0.1,
        )

        # Concatenate excitatory weights
        self.W_exc = np.concatenate((self.W_se, self.W_ee, self.W_ie), axis=0)
        self.W_exc_ideal = np.concatenate(
            (self.W_se_ideal, self.W_ee_ideal, self.W_ie_ideal), axis=0
        )
        self.W_exc_2d = np.concatenate(
            (self.W_se_2d, self.W_ee_2d, self.W_ie_2d), axis=1
        )
        self.W_exc_plt_idx = np.concatenate(
            (self.W_se_plt_idx, self.W_ee_plt_idx, self.W_ie_plt_idx), axis=0
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
                self.W_exc,
                self.W_inh,
                self.W_exc_ideal,
            )

    def vis_network(self, heatmap: bool, weight_layer: bool):
        if heatmap:
            plot_input_space(self.W_se)
        if weight_layer:
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

    def gen_data(
        self,
        N_classes: int,
        noise_rand: bool,
        noise_rand_ls: float | int,
        mean: int | float,
        blank_variance: int | float,
        save: bool,
        retur: bool,
        avg_high_freq: float | int,
        avg_low_freq: float | int,
        var_high_freq: float | int,
        var_low_freq: float | int,
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
                save=save,
                retur=retur,
                rand_lvl=noise_rand_ls[j],
                items=self.num_items,
                avg_high_freq=avg_high_freq,
                avg_low_freq=avg_low_freq,
                var_high_freq=var_high_freq,
                var_low_freq=var_low_freq,
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

        spikes_indices = [
            np.where(self.training_data[:, n])[0]
            for n in range(self.training_data.shape[1])
        ]
        plt.eventplot(spikes_indices, linelengths=0.5)
        plt.show()

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
            self.W_exc_2d,
            self.pre_synaptic_trace,
            self.post_synaptic_trace,
            self.slow_synaptic_trace,
            self.z_istdp,
            self.V_th_array,
        ) = train_data(
            A=self.A,
            P=self.P,
            w_p=self.wp,
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
            tau_cons=self.tau_cons,
            tau_H=self.tau_H,
            tau_istdp=self.tau_istdp,
            tau_ampa=self.tau_ampa,
            tau_nmda=self.tau_nmda,
            tau_gaba=self.tau_gaba,
            tau_thr=self.tau_thr,
            tau_d=self.tau_d,
            tau_f=self.tau_f,
            tau_a=self.tau_a,
            tau_b=self.tau_b,
            delta_a=self.delta_a,
            delta_b=self.delta_b,
            U_exc=self.U_exc,
            U_inh=self.U_inh,
            learning_rate=self.learning_rate,
            training_data=self.training_data,
            N_excit_neurons=self.N_excit_neurons,
            N_inhib_neurons=self.N_inhib_neurons,
            N_input_neurons=self.N_input_neurons,
            MemPot=self.MemPot,
            max_weight=self.max_weight,
            min_weight=self.min_weight,
            W_exc=self.W_exc,
            W_inh=self.W_inh,
            W_exc_ideal=self.W_exc_ideal,
            W_exc_2d=self.W_exc_2d,
            W_exc_plt_idx=self.W_exc_plt_idx,
            gamma=self.gamma,
            alpha_exc=self.alpha_exc,
            alpha_inh=self.alpha_inh,
            save_model=save_model,
            item_lim=item_lim,
            items=self.num_items,
            U_cons=self.U_cons,
        )

        if retur:
            return (
                self.spikes,
                self.MemPot,
                self.W_exc_2d,
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
                W_se=self.W_exc_2d[:, :10],
                W_ee=self.W_exc_2d[:, 10:-10],
                W_ie=self.W_exc_2d[:, -10:],
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
