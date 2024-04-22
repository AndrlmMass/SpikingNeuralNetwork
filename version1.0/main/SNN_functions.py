# SNN functions script

# Import libraries
import os
import sys
import pickle
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow

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
        tau_m: float,
        num_items: float,
        tau_stdp: float,
        dt: float,
        T: int,
        V_rest: int,
        alpha: float | int,
        max_weight: float | int,
        min_weight: float | int,
        num_epochs: int,
        init_cals: int,
        A: float | int,
        B: float | int,
        beta: float | int,
        delta: float | int,
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
        self.P = P
        self.beta = beta
        self.delta = delta
        self.num_timesteps = int(T / dt)
        self.num_items = num_items
        self.time = self.num_timesteps * self.num_items
        self.V_rest = V_rest
        self.leakage_rate = 1 / self.R
        self.alpha = alpha
        self.init_cals = init_cals
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.num_epochs = num_epochs
        self.max_spike_diff = int(self.num_timesteps * 0.1)

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
            basenum=1,
        )
        self.W_ee, self.W_ee_ideal = gws.gen_EE(
            N_excit_neurons=self.N_excit_neurons,
            prob=W_ee_prob,
            time=self.time,
            basenum=1,
        )
        self.W_ei, self.W_ei_ideal = gws.gen_EI(
            N_excit_neurons=self.N_excit_neurons,
            N_inhib_neurons=self.N_inhib_neurons,
            time=self.time,
            weight_val=1,
        )
        self.W_ie, self.W_ie_ideal = gws.gen_IE(
            N_inhib_neurons=self.N_inhib_neurons,
            N_excit_neurons=self.N_excit_neurons,
            W_ei=self.W_ei,
            time=self.time,
            N_ws=4,
            weight_val=0.5,
            radius=radius_,
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
        run: bool,
        N_classes: int,
        noise_rand: bool,
        noise_rand_ls: float | int,
        mean: int | float,
        blank_variance: int | float,
        input_scaler: int | float,
        save: bool,
        retur: bool,
    ):
        if not run:
            return
        for j in range(len(noise_rand_ls)):
            data, labels = gen_float_data_(
                run=run,
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
        data_path = f"\\data\\training_data\\training_data_{rand_lvl}.pkl"
        data_dir = cur_path + data_path

        with open(data_dir, "rb") as openfile:
            self.training_data = pickle.load(openfile)

        label_path = f"\\data\\labels_train\\labels_train_{rand_lvl}.pkl"
        label_dir = cur_path + label_path

        with open(label_dir, "rb") as openfile:
            self.labels_train = pickle.load(openfile)

        if retur:
            return self.training_data, self.labels_train

    def train_data(
        self,
        w_p: float | int,
        retur: bool,
        update_frequency: int,
        interactive_tool: bool,
    ):
        if interactive_tool:
            app = QApplication(sys.argv)

            MW = MainWidget()
            MW.init_gui(
                V_th=self.V_th,
                V_reset=self.V_reset,
                P=self.P,
                R=self.R,
                A=self.A,
                B=self.B,
                w_p=w_p,
                beta=self.beta,
                delta=self.delta,
                time=self.time,
                V_rest=self.V_rest,
                dt=self.dt,
                tau_m=self.tau_m,
                tau_const=1,
                update_frequency=update_frequency,
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
                train_data=self.training_data,
            )

            main_window = MainWindow(MW)
            main_window.show()
            sys.exit(app.exec_())
        else:
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
            ) = train_data(
                R=self.R,
                A=self.A,
                B=self.B,
                P=1,
                w_p=w_p,  # Defines the upper stable point of weight convergence
                beta=self.beta,
                delta=self.beta,
                time=self.time,
                V_th=self.V_th,
                V_rest=self.V_rest,
                V_reset=self.V_reset,
                dt=self.dt,
                tau_m=self.tau_m,
                tau_const=0.001,  # Defines the rate of convergence, e.g., 20 minutes
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
                update_frequency=update_frequency,
                callback=None,
            )
        self.update_frequency = update_frequency

        if retur:
            return (
                self.spikes,
                self.MemPot,
                self.W_se,
                self.W_ee,
                self.W_ei,
                self.W_ie,
            )

    def plot_training(
        self,
        ws_nd_spikes: bool,
        idx_start: int,
        idx_stop: int,
        mv: bool,
        time_start: int,
        time_stop: int,
    ):
        if ws_nd_spikes:
            plot_weights_and_spikes(
                spikes=self.spikes,
                W_se=self.W_se,
                W_ee=self.W_ee,
                dt=self.dt,
                update_interval=self.update_frequency,
            )
        if mv:
            plot_membrane_activity(
                MemPot=self.MemPot,
                idx_start=idx_start,
                idx_stop=idx_stop,
                update_interval=self.update_frequency,
                time_start=time_start,
                time_stop=time_stop,
            )
