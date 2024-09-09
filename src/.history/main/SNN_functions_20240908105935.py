# SNN functions script

# Import libraries
import os
import sys
import numpy as np
import json

# Set the current directory based on the existence of a specific path
if os.path.exists(
    "C:\\Users\\Bruker\\OneDrive\\Documents\\NMBU_\\BONSAI\\SNN\\SpikingNeuralNetwork\\src"
):
    base_path = "C:\\Users\\Bruker\\OneDrive\\Documents\\NMBU_\\BONSAI\\SNN\\SpikingNeuralNetwork\\src"
else:
    base_path = "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\src"

os.chdir(base_path)
sys.path.append(os.path.join(base_path, "gen"))
sys.path.append(os.path.join(base_path, "main"))
sys.path.append(os.path.join(base_path, "plot"))
sys.path.append(os.path.join(base_path, "tool"))

from plot_training import *
from plot_network import *
from gen_weights import *
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
        tau_ht: float | int,
        tau_m: float | int,
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

    def process(
        self,
        data: bool = False,
        model: bool = False,
        args: dict = None,
        load: bool = False,
        save: bool = False,
    ):

        # Add checks
        if model and data:
            raise ValueError("model and data variables cannot both be True")

        if not args or args == None:
            raise ValueError(
                "args argument needs to be a dictionary with values. Current variable is either an empty dict or a none-value"
            )

        if not data and not model and not load and not save:
            raise UserWarning(
                "All boolean variables are False and no operations will be performed."
            )

        ########## model save or load ##########
        if model:
            print(args)

            output = input("Should we continue?")
            if output != "y":
                raise ValueError("ur mama fat")

            # Remove irrelevant parameters from args dict
            del args["some_key"] # Fill in here

            # Update dictionary
            args.update()

            if save:
                # Generate a random number for model folder
                rand_num = np.random.randint(0, 1000)

                # Check if random number folder exists
                while os.path.exists(f"model/model_{rand_num}"):
                    rand_num = np.random.randint(0, 1000)

                os.makedirs(f"model/model_{rand_num}")

                # Save main path
                save_path = f"model/model_{rand_num}"

                # Create a dictionary of file names and variables
                vars_to_save = {
                    "W_exc_2d": self.W_exc_2d,
                    "spikes": self.spikes,
                    "MemPot": self.MemPot,
                    "pre_synaptic_trace": self.pre_synaptic_trace,  # why is this not a variable?
                    "post_synaptic_trace": self.post_synaptic_trace,
                    "slow_pre_synaptic_trace": self.slow_pre_synaptic_trace,
                    "C": self.C,
                    "z_ht": self.z_ht,
                    "x": self.x,
                    "u": self.u,
                    "H": self.H,
                    "z_i": self.z_i,
                    "z_j": self.z_j,
                    "config": self.filtered_locs,
                    "V_th_array": self.V_th_array,
                    "exc_weights": self.W_exc,
                    "inh_weights": self.W_inh,
                    "V_th": self.V_th,
                    "g_nmda": self.g_nmda,
                    "g_ampa": self.g_ampa,
                    "g_gaba": self.g_gaba,
                    "g_a": self.g_a,
                    "g_b": self.g_b,
                }

                # Loop through the dictionary and save each variable
                for filename, var in vars_to_save.items():
                    np.save(f"{save_path}/{filename}.npy", var)

                # Save model parameters
                filepath = f"model\\{rand_nums}\\model_parameters.json"

                with open(filepath, "w") as outfile:
                    json.dump(args, outfile)

                print("model saved", end="\r")
                return

            if load:
                print(args)

                output = input("Should we continue?")
                if output != "y":
                    raise ValueError("ur mama fat")

                folders = os.listdir("model")

                # Search for existing models
                if len(folders) > 0:
                    for folder in folders:
                        ex_params = json.load(open(f"model\\{folder}\\model_parameters.json"))

                        # Check if parameters are the same as the current ones
                        if ex_params == args:
                            # Load the model (this will run in the main thread)
                            save_path = f"model/model_{folder}"

                            # Now you can access the variables like this:
                            self.W_exc_2d = np.load(save_path + "/W_exc_2d")
                            self.spikes = np.load(save_path + "/spikes")
                            self.MemPot = np.load(save_path + "/MemPot")
                            self.pre_synaptic_trace = np.load(
                                save_path + "/pre_synaptic_trace"
                            )
                            self.post_synaptic_trace = np.load(
                                save_path + "/post_synaptic_trace"
                            )
                            self.slow_pre_synaptic_trace = np.load(
                                save_path + "/slow_pre_synaptic_trace"
                            )
                            self.C = np.load(save_path + "/C")
                            self.z_ht = np.load(save_path + "/z_ht")
                            self.x = np.load(save_path + "/x")
                            self.u = np.load(save_path + "/u")
                            self.H = np.load(save_path + "/H")
                            self.z_i = np.load(save_path + "/z_i")
                            self.z_j = np.load(save_path + "/z_j")
                            self.filtered_locs = np.load(save_path + "/config")
                            self.V_th_array = np.load(save_path + "/V_th_array")
                            self.W_exc = np.load(save_path + "/W_exc")
                            self.W_inh = np.load(save_path + "/W_inh")
                            self.V_th = np.load(save_path + "/V_th")
                            self.g_nmda = np.load(save_path + "/g_nmda")
                            self.g_ampa = np.load(save_path + "/g_ampa")
                            self.g_gaba = np.load(save_path + "/g_gaba")
                            self.g_a = np.load(save_path + "/g_a")
                            self.g_b = np.load(save_path + "/g_b")
                            self.model_params = ex_params
                            print("model loaded", end="\r")
                            return 1
                return 0

        ########## data save or load ##########
        if data:
            if save:
                rand_nums = np.random.randint(low=0, high=9, size=5)

                # Check if name is taken
                while any(item in os.listdir("data") for item in rand_nums):
                    rand_nums = np.random.randint(low=0, high=9, size=5)

                # Create folder to store data
                os.makedirs(f"data\\{rand_nums}")

                # Save training data and labels
                np.save(f"data\\{rand_nums}\\data_bin.npy", self.training_data)
                np.save(f"data\\{rand_nums}\\labels_bin.npy", self.labels_train)
                filepath = f"data\\{rand_nums}\\data_parameters.json"

                with open(filepath, "w") as outfile:
                    json.dump(args, outfile)

                print("data saved", end="\r")
                return 1

            if load:
                # Define folder to load data
                folders = os.listdir("data")

                # Search for existing data gens
                if len(folders) > 0:
                    for folder in folders:
                        ex_params = json.load(
                            open(f"data\\{folder}\\data_parameters.json")
                        )

                        # Check if parameters are the same as the current ones
                        if ex_params == args:
                            self.training_data = np.load(
                                f"data\\{folder}\\data_bin.npy"
                            )
                            self.labels_train = np.load(
                                f"data\\{folder}\\labels_bin.npy"
                            )
                            self.data_params = ex_params
                            print("data loaded", end="\r")
                            return

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
                weights=self.W_se_2d,
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
        noise_variance: float | int,
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

        # Create training data since it does not exist already
        gd = gen_data_cl(
            N_classes=N_classes,
            N_input_neurons=self.N_input_neurons,
            items=self.num_items,
            noise_rand=noise_rand,
            noise_variance=noise_variance,
            mean=mean,
            blank_variance=blank_variance,
            time=self.time,
            timesteps=self.num_timesteps,
            dt=self.dt,
            retur=retur,
            avg_high_freq=avg_high_freq,
            avg_low_freq=avg_low_freq,
            var_high_freq=var_high_freq,
            var_low_freq=var_low_freq,
        )

        gd.gen_float_data_()

        # Convert to binary poisson sequences
        self.training_data, self.labels_train, self.basic_data, self.labels_seq = (
            gd.float_2_pos_spike()
        )

        # Save data
        self.process(data=True, args=)

    def visualize_data(self, single_data, raster_plot_, alt_raster_plot):
        if single_data:
            input_space_plotted_single(self.training_data[0])
        if raster_plot_:
            raster_plot(self.training_data)
        if alt_raster_plot:
            raster_plot_other(self.training_data, self.labels_train, False)

    def train_data(
        self,
        retur: bool,
        save_model: bool,
        force_retrain: bool,
    ):
        (
            self.W_exc_2d,
            self.spikes,
            self.MemPot,
            self.post_synaptic_trace,
            self.slow_pre_synaptic_trace,
            self.C,
            self.z_ht,
            self.x,
            self.u,
            self.H,
            self.z_i,
            self.z_j,
            self.filtered_locs,
            self.V_th_array,
            self.W_exc,
            self.W_inh,
            self.V_th,
            self.g_nmda,
            self.g_ampa,
            self.g_gaba,
            self.g_a,
            self.g_b,
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
            U_cons=self.U_cons,
            force_retrain=force_retrain,
        )

        if retur:
            return (
                self.W_exc_2d,
                self.spikes,
                self.MemPot,
                self.post_synaptic_trace,
                self.slow_pre_synaptic_trace,
                self.C,
                self.z_ht,
                self.x,
                self.u,
                self.H,
                self.z_i,
                self.z_j,
                self.filtered_locs,
                self.V_th_array,
                self.W_exc,
                self.W_inh,
                self.V_th,
                self.g_nmda,
                self.g_ampa,
                self.g_gaba,
                self.g_a,
                self.g_b,
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
                self.labels_seq,
                self.num_timesteps,
                self.N_input_neurons,
            )
