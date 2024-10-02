# SNN functions script
import os
import numpy as np
import random
import json

# Importing local scripts
from plot.plot_training import *
from plot.plot_network import *
from gen.gen_weights import *
from plot.plot_data import *
from gen.gen_data import *
from main.train import *


# Initialize class variable
class SNN:
    # Initialize neuron parameters
    def __init__(
        self,
        V_th: int = -50,  # Spiking threshold
        V_rest: int = -60,  # Resting potential
        P: int | float = 20,  # Potential strength
        C: int | float = 1,  # Where does it say that this should be 1?
        U: float | int = 0.2,  # Initial release probability parameter
        tau_plus: float | int = 20,  # presynaptic excitatory synapse
        tau_minus: float | int = 20,  # postsynaptic excitatory synapse
        tau_slow: float | int = 100,  # slowsynaptic excitatory synapse
        tau_ht: float | int = 100,  # Spiking threshold time constant
        tau_m: float | int = 20,  # Membrane time constant
        tau_hom: float | int = 30000,  # metaplasticity time constant (5 minutes)
        tau_istdp: float | int = 20,  # Inhibitory weight update constant
        tau_H: float | int = 1 * 10**4,  # Global secreted factor time constant
        tau_thr: float | int = 2,  #
        tau_ampa: float | int = 5,
        tau_nmda: float | int = 100,
        tau_gaba: float | int = 10,
        tau_a: float | int = 100,  # 100ms
        tau_b: float | int = 20 * 10**3,  # 20s
        tau_d: float | int = 200,
        tau_f: float | int = 600,
        delta_a: float | int = 0.1,  # decay unit
        delta_b: float | int = 5 * 10**-4,  # seconds,
        U_exc: int = 0,
        U_inh: int = -80,
        alpha_exc: float | int = 0.2,
        alpha_inh: float | int = 0.3,
        learning_rate: float | int = 2 * 10**-5,
        gamma: float | int = 4 * 10**-3,  # Target population rate in Hz
        num_items: int = 4,  # Num of items,
        dt: int | float = 1,  # time unit for modelling,
        T: int = 1000,  # total time each item will appear
        wp: float | int = 0.5,
        num_epochs: int = 1,  # number of epochs -> not currently in use
        A: float | int = 1 * 10**-3,  # LTP rate,
        B: float | int = 1 * 10**-3,  # LTD rate,
        beta: float | int = 0.05,
        delta: float | int = 2 * 10**-5,  # Transmitter triggered plasticity
        tau_cons: float | int = 6 * 10**4,  # 5 minutes until weight convergence
        euler: int = 5,
        U_cons: float | int = 0.2,
        th_rest: int = -50,
        th_refact: int = 50,
    ):
        self.V_th = V_th
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
        self.num_epochs = num_epochs
        self.beta = beta
        self.delta = delta
        self.tau_cons = tau_cons
        self.euler = euler
        self.num_timesteps = int(T / dt)
        self.time = self.num_timesteps * self.num_items
        self.U_cons = U_cons
        self.th_rest = th_rest
        self.th_refact = th_refact
        self.data_loaded = False
        self.model_loaded = False

    def process(
        self,
        data: bool = False,
        model: bool = False,
        load: bool = False,
        save: bool = False,
    ):

        # Add checks
        if model and data:
            raise ValueError("model and data variables cannot both be True")

        if not model and not data:
            raise UserWarning(
                "No processing will occur. model and data variables are False."
            )
        if load and save:
            raise ValueError("load and save variables cannot both be True")

        if not load and not save:
            raise UserWarning(
                "No processing will occur. load and save variables are False."
            )

        ########## load or save model ##########
        if model:
            if not os.path.exists("model"):
                os.makedirs("model")
            if save:
                # Generate a random number for model folder
                rand_num = random.sample((0, 9), 5)

                # Check if random number folder exists
                while os.path.exists(os.path.join("model", f"model_{rand_num}")):
                    rand_num = random.sample((0, 9), 5)

                # Save main path
                save_path = os.path.join("model", f"model_{rand_num}")
                os.makedirs(os.path.join(save_path))

                # Create a dictionary of file names and variables
                vars_to_save = {
                    "W_plastic_plt": self.W_plastic_plt,
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
                    "V_th_array": self.V_th_array,
                    "plastic_weights": self.W_plastic,
                    "static_weights": self.W_static,
                    "V_th": self.V_th,
                    "g_nmda": self.g_nmda,
                    "g_ampa": self.g_ampa,
                    "g_gaba": self.g_gaba,
                    "g_a": self.g_a,
                    "g_b": self.g_b,
                }

                # Loop through the dictionary and save each variable
                for filename, var in vars_to_save.items():
                    np.save(os.path.join(save_path, filename), var)

                # Save model parameters
                filepath = os.path.join(save_path, "model_parameters.json")

                with open(filepath, "w") as outfile:
                    json.dump(self.model_parameters, outfile)

                print("model saved", end="\r")
                return

            if load:
                folders = os.listdir("model")

                # Search for existing models
                if len(folders) > 0:
                    for folder in folders:
                        ex_params = json.load(
                            open(os.path.join("model", folder, "model_parameters.json"))
                        )

                        # Check if parameters are the same as the current ones
                        if ex_params == self.model_parameters:
                            # Load the modelll(dir != "data" for dir in os.listdir()):
                            save_path = os.path.join("model", folder)

                            # Now you can access the variables like this:
                            self.W_plastic_plt = np.load(
                                os.path.join(save_path, "W_plastic_plt.npy")
                            )
                            self.spikes = np.load(os.path.join(save_path, "spikes.npy"))
                            self.MemPot = np.load(os.path.join(save_path, "MemPot.npy"))
                            self.pre_synaptic_trace = np.load(
                                os.path.join(save_path, "pre_synaptic_trace.npy")
                            )
                            self.post_synaptic_trace = np.load(
                                os.path.join(save_path, "post_synaptic_trace.npy")
                            )
                            self.slow_pre_synaptic_trace = np.load(
                                os.path.join(save_path, "slow_pre_synaptic_trace.npy")
                            )
                            self.C = np.load(os.path.join(save_path, "C.npy"))
                            self.z_ht = np.load(os.path.join(save_path, "\\z_ht.npy"))
                            self.x = np.load(os.path.join(save_path, "\\x.npy"))
                            self.u = np.load(os.path.join(save_path, "\\u.npy"))
                            self.H = np.load(os.path.join(save_path, "\\H.npy"))
                            self.z_i = np.load(os.path.join(save_path, "\\z_i.npy"))
                            self.z_j = np.load(os.path.join(save_path, "\\z_j.npy"))
                            self.V_th_array = np.load(
                                os.path.join(save_path, "\\V_th_array.npy")
                            )
                            self.W_plastic = np.load(
                                os.path.join(save_path, "\\plastic_weights.npy")
                            )
                            self.W_static = np.load(
                                os.path.join(save_path, "\\static_weights.npy")
                            )
                            self.V_th = np.load(os.path.join(save_path, "\\V_th.npy"))
                            self.g_nmda = np.load(
                                os.path.join(save_path, "\\g_nmda.npy")
                            )
                            self.g_ampa = np.load(
                                os.path.join(save_path, "\\g_ampa.npy")
                            )
                            self.g_gaba = np.load(
                                os.path.join(save_path, "\\g_gaba.npy")
                            )
                            self.g_a = np.load(os.path.join(save_path, "\\g_a.npy"))
                            self.g_b = np.load(os.path.join(save_path, "\\g_b.npy"))
                            print("model loaded", end="\r")
                            self.model_loaded = True
                            return
                return

        ########## load or save data ##########
        if data:
            if not os.path.exists("data"):
                os.makedirs("data")
            if save:
                rand_nums = np.random.randint(low=0, high=9, size=5)

                while any(item in os.listdir("data") for item in rand_nums):
                    rand_nums = random.randint(low=0, high=9, size=5)[0]

                # Create folder to store data
                data_dir = os.path.join("data", str(rand_nums))
                os.makedirs(data_dir)

                # Save training data and labels
                np.save(os.path.join(data_dir, "data_bin.npy"), self.training_data)
                np.save(os.path.join(data_dir, "labels_bin.npy"), self.labels_train)
                np.save(os.path.join(data_dir, "basic_data.npy"), self.basic_data)
                np.save(os.path.join(data_dir, "labels_seq.npy"), self.labels_seq)
                filepath = os.path.join(data_dir, "data_parameters.json")

                with open(filepath, "w") as outfile:
                    json.dump(self.data_parameters, outfile)

                print("data saved", end="\r")
                return

            if load:
                # Define folder to load data
                folders = os.listdir("data")

                # Search for existing data gens
                if len(folders) > 0:
                    for folder in folders:
                        json_file_path = os.path.join(
                            "data", folder, "data_parameters.json"
                        )

                        with open(json_file_path, "r") as j:
                            ex_params = json.loads(j.read())

                        # Check if parameters are the same as the current ones
                        if ex_params == self.data_parameters:
                            self.training_data = np.load(
                                os.path.join("data", folder, "data_bin.npy")
                            )
                            self.labels_train = np.load(
                                os.path.join("data", folder, "labels_bin.npy")
                            )
                            self.basic_data = np.load(
                                os.path.join("data", folder, "basic_data.npy")
                            )
                            self.labels_seq = np.load(
                                os.path.join("data", folder, "labels_seq.npy")
                            )

                            self.data_loaded = True

                            print("data loaded", end="\r")

                            return

    def build(
        self,
        radius: int = 4,
        N_input_neurons: int = 484,
        N_excit_neurons: int = 484,
        N_inhib_neurons: int = 121,
        w_prob_se: float | int = 0.05,
        w_prob_ee: float | int = 0.1,
        w_prob_ei: float | int = 0.1,
        w_prob_ii: float | int = 0.1,
        w_prob_ie: float | int = 0.1,
        w_val_se: float | int = 0.2,
        w_val_ee: float | int = 0.1,
        w_val_ei: float | int = 0.2,
        w_val_ii: float | int = 0.2,
        w_val_ie: float | int = 0.1,
        load_model_if_available: bool = True,
    ):
        self.N_input_neurons = N_input_neurons
        self.N_excit_neurons = N_excit_neurons
        self.N_inhib_neurons = N_inhib_neurons
        self.num_neurons = (
            self.N_excit_neurons + self.N_inhib_neurons + self.N_input_neurons
        )

        # Get current variables in use
        self.model_parameters = {**locals()}

        # Copy and remove class element to dict
        self_dict = self.__dict__.copy()
        del self.model_parameters["self"]

        # Combine dicts and update
        self.model_parameters.update(self_dict)

        # Remove irrelevant arguments
        del self.model_parameters["model_parameters"]
        del self.model_parameters["load_model_if_available"]

        # Update model
        self.model_parameters.update()

        if load_model_if_available:

            # Load model if possible
            self.process(model=True, load=True)

            # Check if previous model has been loaded
            if self.model_loaded == True:
                return

        # Generate weights
        gws = gen_weights()

        self.W_se, self.W_se_ideal = gws.gen_SE(
            N_input_neurons=self.N_input_neurons,
            N_excit_neurons=self.N_excit_neurons,
            w_prob=w_prob_se,
            w_val=w_val_se,
            radius=radius,
        )
        self.W_ee, self.W_ee_ideal = gws.gen_EE(
            N_excit_neurons=self.N_excit_neurons,
            w_prob=w_prob_ee,
            w_val=w_val_ee,
        )
        self.W_ei = gws.gen_EI(
            N_excit_neurons=self.N_excit_neurons,
            N_inhib_neurons=self.N_inhib_neurons,
            w_prob=w_prob_ei,
            w_val=w_val_ei,
        )

        self.W_ii = gws.gen_II(
            N_inhib_neurons=self.N_inhib_neurons,
            w_prob=w_prob_ii,
            w_val=w_val_ii,
        )

        self.W_ie, self.W_ie_plt = gws.gen_IE(
            N_inhib_neurons=self.N_inhib_neurons,
            N_excit_neurons=self.N_excit_neurons,
            time=self.time,
            w_prob=w_prob_ie,
            w_val=w_val_ie,
        )

        # Concatenate plastic weights
        self.W_plastic = np.concatenate((self.W_se, self.W_ee, self.W_ie), axis=0)
        self.W_plastic_ideal = np.concatenate(
            (self.W_se_ideal, self.W_ee_ideal), axis=0
        )
        self.W_plastic_plt = np.zeros(shape=(self.time, 9))

        # Concatenate static weights
        self.W_static = np.concatenate((self.W_ei, self.W_ii), axis=0)

    def gen_data(
        self,
        N_classes: int = 4,
        noise_rand: bool = True,
        noise_variance: float | int = 0.05,
        mean: int | float = 0,
        blank_variance: int | float = 0.01,
        save: bool = True,
        avg_high_freq: float | int = 30,
        avg_low_freq: float | int = 10,
        var_high_freq: float | int = 0.05,
        var_low_freq: float | int = 0.05,
        force_new_data: bool = False,
    ):
        self.N_classes = N_classes
        self.data_parameters = {**locals()}

        self_dict = self.__dict__.copy()
        del self.data_parameters["self"]
        del self.data_parameters["save"]

        # Add relevant parameters
        self.data_parameters["num_epochs"] = self_dict["num_epochs"]
        self.data_parameters["num_timesteps"] = self_dict["num_timesteps"]
        self.data_parameters["num_items"] = self_dict["num_items"]
        self.data_parameters["time"] = self_dict["time"]

        # Check if training data exists and load if it does
        self.process(data=True, load=True)

        if self.data_loaded == True and force_new_data == False:
            return

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
        if save:
            self.process(data=True, save=save)

    def visualize_data(
        self,
        single_data: bool = False,
        raster_plot_: bool = True,
        alt_raster_plot: bool = False,
    ):
        if single_data:
            input_space_plotted_single(self.training_data[0])
        if raster_plot_:
            raster_plot(self.training_data)
        if alt_raster_plot:
            raster_plot_other(self.training_data, self.labels_train, False)

    def train_(
        self,
        run_njit: bool = True,
        save_model: bool = True,
        force_retrain: bool = False,
    ):
        if self.model_loaded == True and force_retrain == False:
            return

        (
            self.W_plastic_plt,
            self.spikes,
            self.MemPot,
            self.pre_synaptic_trace,
            self.post_synaptic_trace,
            self.slow_pre_synaptic_trace,
            self.C,
            self.z_ht,
            self.x,
            self.u,
            self.H,
            self.z_i,
            self.z_j,
            self.V_th_array,
            self.W_plastic,
            self.W_static,
            self.V_th,
            self.g_nmda,
            self.g_ampa,
            self.g_gaba,
            self.g_a,
            self.g_b,
        ) = train_model(
            A=self.A,
            P=self.P,
            w_p=self.wp,
            beta=self.beta,
            delta=self.delta,
            euler=self.euler,
            time=self.time,
            V_th_=self.V_th,
            V_rest=self.V_rest,
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
            W_plastic=self.W_plastic,
            W_static=self.W_static,
            W_plastic_ideal=self.W_plastic_ideal,
            W_plastic_plt=self.W_plastic_plt,
            gamma=self.gamma,
            alpha_exc=self.alpha_exc,
            alpha_inh=self.alpha_inh,
            U_cons=self.U_cons,
            th_rest=self.th_rest,
            th_refact=self.th_refact,
            run_njit=run_njit,
        )

        if save_model:
            self.process(model=True, save=True)

    def plot_training(
        self,
        t_stop: int = None,
        t_start: int = None,
        items: int = None,
        ws_nd_spikes: bool = True,
        mv: bool = True,
        overlap: bool = True,
        traces: bool = True,
        tsne: bool = True,
    ):
        if t_stop == None:
            t_stop = self.time

        if items == None:
            items = self.num_items

        if t_start == None:
            t_start = self.time - int(self.time * 0.2)
        print("t_start:", t_start, "t_stop:", t_stop)

        if ws_nd_spikes:
            plot_weights_and_spikes(
                spikes=self.spikes,
                weights=self.W_plastic_plt,
                t_start=t_start,
                t_stop=t_stop,
            )
        if mv:
            plot_membrane_activity(
                MemPot=self.MemPot,
                MemPot_th=self.V_th,
                t_start=t_start,
                t_stop=t_stop,
                N_excit_neurons=self.N_excit_neurons,
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
                slow_pre_synaptic_trace=self.slow_pre_synaptic_trace,
                N_input_neurons=self.N_input_neurons,
            )
        if tsne:
            t_SNE(
                self.N_classes,
                self.spikes,
                self.labels_train,
                self.labels_seq,
                self.num_timesteps,
                self.N_input_neurons,
            )
