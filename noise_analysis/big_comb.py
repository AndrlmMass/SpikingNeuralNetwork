import numpy as np
import os
import json
import random
from train import train_network
from get_data import create_data
from plot import (
    spike_plot,
    heat_map,
    mp_plot,
    weights_plot,
    spike_threshold_plot,
    plot_traces,
    plot_accuracy,
    top_responders_plotted,
)
from analysis import t_SNE, PCA_analysis, Clustering_estimation
from create_network import create_weights, create_arrays


class snn_sleepy:
    def __init__(
        self,
        N_exc=200,
        N_inh=50,
        N_x=225,
        classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        supervised=False,
        unsupervised=False,
    ):
        self.N_exc = N_exc
        self.N_inh = N_inh
        self.N_x = N_x
        self.N_classes = len(classes)
        self.classes = classes
        self.supervised = supervised
        self.unsupervised = unsupervised
        if self.unsupervised or self.supervised:
            self.st = N_x  # stimulation
            self.ex = self.st + N_exc  # excitatory
            self.ih = self.ex + N_inh  # inhibitory
            self.pp = self.ih + self.N_classes  # predicted positive
            self.pn = self.pp + self.N_classes  # predicted negative
            self.tp = self.pn + self.N_classes  # true positive
            self.tn = self.tp + self.N_classes  # true negative
            self.fp = self.tn + self.N_classes  # false positive
            self.fn = self.fp + self.N_classes  # false negative
        else:
            self.st = N_x  # stimulation
            self.ex = self.st + N_exc  # excitatory
            self.ih = self.ex + N_inh  # inhibitory

        if self.unsupervised and supervised:
            raise ValueError("Unsupervised and supervised cannot both be true.")

        if self.supervised:
            self.N = N_exc + N_inh + N_x + self.N_classes * 8
        elif self.unsupervised:
            self.N = N_exc + N_inh + N_x + self.N_classes * 8
        else:
            self.N = N_exc + N_inh + N_x

    def process(
        self,
        model_dir_: str = None,
        load_data: bool = False,
        save_data: bool = False,
        load_model: bool = False,
        save_model: bool = False,
        save_test_model: bool = False,
        load_test_model: bool = False,
        data_parameters: dict = None,
        model_parameters: dict = None,
    ):

        # Add checks
        if load_data and save_data:
            raise ValueError("load and save data cannot both be True")
        self.data_loaded = False
        self.model_loaded = False
        self.test_data_loaded = False

        ########## load or save data ##########
        if save_data:
            # generate random number to create unique folder
            rand_nums = np.random.randint(low=0, high=9, size=5)

            # Check if folder already exists
            while any(item in os.listdir("data") for item in rand_nums):
                rand_nums = random.randint(low=0, high=9, size=5)[0]

            # Create folder to store data
            data_dir = os.path.join("data/sdata", str(rand_nums))
            os.makedirs(data_dir)

            # Save training data and labels
            np.save(os.path.join(data_dir, "data_train.npy"), self.data_train)
            np.save(os.path.join(data_dir, "data_test.npy"), self.data_test)
            np.save(os.path.join(data_dir, "labels_train.npy"), self.labels_train)
            np.save(os.path.join(data_dir, "data_test.npy"), self.data_test)
            np.save(os.path.join(data_dir, "labels_test.npy"), self.labels_test)
            np.save(
                os.path.join(data_dir, "labels_true_train.npy"), self.labels_true_train
            )
            np.save(
                os.path.join(data_dir, "labels_true_test.npy"), self.labels_true_test
            )
            filepath = os.path.join(data_dir, "data_parameters.json")

            with open(filepath, "w") as outfile:
                json.dump(data_parameters, outfile)

            print("\rdata saved", end="")
            return

        if load_data:
            # Define folder to load data
            folders = os.listdir("data/sdata")

            # Search for existing data gens
            if len(folders) > 0:
                for folder in folders:
                    json_file_path = os.path.join(
                        "data/sdata", folder, "data_parameters.json"
                    )

                    with open(json_file_path, "r") as j:
                        ex_params = json.loads(j.read())

                    # Check if parameters are the same as the current ones
                    if ex_params == data_parameters:
                        self.data_train = np.load(
                            os.path.join("data/sdata", folder, "data_train.npy")
                        )
                        self.labels_train = np.load(
                            os.path.join("data/sdata", folder, "labels_train.npy")
                        )
                        self.data_test = np.load(
                            os.path.join("data/sdata", folder, "data_test.npy")
                        )
                        self.labels_test = np.load(
                            os.path.join("data/sdata", folder, "labels_test.npy")
                        )
                        if self.unsupervised or self.supervised:
                            self.labels_true_train = np.load(
                                os.path.join(
                                    "data/sdata", folder, "labels_true_train.npy"
                                )
                            )

                            self.labels_true_test = np.load(
                                os.path.join(
                                    "data/sdata", folder, "labels_true_test.npy"
                                )
                            )
                        else:
                            self.labels_true_test = None
                            self.labels_true_train = None

                        print("data loaded", end="\r")
                        self.data_loaded = True
                        return

        ########## load or save model ##########
        if save_model and load_model:
            raise ValueError("load and save model cannot both be True")

        if save_model:
            if save_test_model:
                np.save(
                    os.path.join(model_dir_, "spikes_test.npy"),
                    self.spikes_test,
                )
                print("\rtest saved", end="")
                return

            if not os.path.exists("model"):
                os.makedirs("model")

            # generate random number to create unique folder
            rand_nums = np.random.randint(low=0, high=9, size=5)

            # Check if folder already exists
            while any(item in os.listdir("data") for item in rand_nums):
                rand_nums = random.randint(low=0, high=9, size=5)[0]

            # Create folder to store data
            model_dir = os.path.join("model", str(rand_nums))
            os.makedirs(model_dir)

            # Save training data and labels
            np.save(os.path.join(model_dir, "weights.npy"), self.weights)
            np.save(os.path.join(model_dir, "spikes_train.npy"), self.spikes_train)
            np.save(os.path.join(model_dir, "pre_trace.npy"), self.pre_trace)
            np.save(os.path.join(model_dir, "post_trace.npy"), self.post_trace)
            np.save(os.path.join(model_dir, "mp_train.npy"), self.mp_train)
            np.save(
                os.path.join(model_dir, "weights2plot_exc.npy"),
                self.weights2plot_exc,
            )
            np.save(
                os.path.join(model_dir, "weights2plot_inh.npy"),
                self.weights2plot_inh,
            )
            np.save(os.path.join(model_dir, "pre_trace_plot.npy"), self.pre_trace_plot)
            np.save(
                os.path.join(model_dir, "post_trace_plot.npy"), self.post_trace_plot
            )
            np.save(
                os.path.join(model_dir, "spike_threshold.npy"), self.spike_threshold
            )
            np.save(os.path.join(model_dir, "weight_mask.npy"), self.weight_mask)
            np.save(
                os.path.join(model_dir, "max_weight_sum_inh.npy"),
                self.max_weight_sum_inh,
            )
            np.save(
                os.path.join(model_dir, "max_weight_sum_exc.npy"),
                self.max_weight_sum_exc,
            )
            np.save(os.path.join(model_dir, "labels_train"), self.labels_train)
            np.save(os.path.join(model_dir, "labels_test"), self.labels_test)

            filepath = os.path.join(model_dir, "model_parameters.json")

            with open(filepath, "w") as outfile:
                json.dump(model_parameters, outfile)

            print("\rmodel saved", end="")
            return model_dir

        if load_model:
            if load_test_model:
                path = os.path.join(model_dir_, "spikes_test.npy")
                if os.path.exists(path):
                    self.spikes_test = np.load(
                        os.path.join(model_dir_, "spikes_test.npy")
                    )
                    self.test_data_loaded = True
                    print("\rTest spikes found.", end="")
                else:
                    print("\rTest spikes not found. Testing again.", end="")

            # Define folder to load data
            folders = os.listdir("model")

            # Search for existing data gens
            if len(folders) > 0:
                for folder in folders:
                    json_file_path = os.path.join(
                        "model", folder, "model_parameters.json"
                    )

                    with open(json_file_path, "r") as j:
                        ex_params = json.loads(j.read())

                    # Check if parameters are the same as the current ones
                    if ex_params == model_parameters:
                        self.weights = np.load(
                            os.path.join("model", folder, "weights.npy")
                        )
                        self.spikes_train = np.load(
                            os.path.join("model", folder, "spikes_train.npy")
                        )
                        self.pre_trace = np.load(
                            os.path.join("model", folder, "pre_trace.npy")
                        )
                        self.post_trace = np.load(
                            os.path.join("model", folder, "post_trace.npy")
                        )
                        self.mp_train = np.load(
                            os.path.join("model", folder, "mp_train.npy")
                        )
                        self.weights2plot_exc = np.load(
                            os.path.join("model", folder, "weights2plot_exc.npy")
                        )
                        self.weights2plot_inh = np.load(
                            os.path.join("model", folder, "weights2plot_inh.npy")
                        )
                        self.pre_trace_plot = np.load(
                            os.path.join("model", folder, "pre_trace_plot.npy")
                        )
                        self.post_trace_plot = np.load(
                            os.path.join("model", folder, "post_trace_plot.npy")
                        )
                        self.spike_threshold = np.load(
                            os.path.join("model", folder, "spike_threshold.npy")
                        )
                        self.weight_mask = np.load(
                            os.path.join("model", folder, "weight_mask.npy")
                        )
                        self.max_weight_sum_inh = np.load(
                            os.path.join("model", folder, "max_weight_sum_inh.npy")
                        )
                        self.max_weight_sum_exc = np.load(
                            os.path.join("model", folder, "max_weight_sum_exc.npy")
                        )
                        self.labels_train = np.load(
                            os.path.join("model", folder, "labels_train.npy")
                        )
                        self.labels_test = np.load(
                            os.path.join("model", folder, "labels_test.npy")
                        )

                        print("\rmodel loaded", end="")
                        self.model_loaded = True
                        return os.path.join("model", folder)
            else:
                print("No model found to load. Will train new model from scratch.")

    # acquire data
    def prepare_data(
        self,
        num_images=50,
        force_recreate=False,
        plot_comparison=False,
        inspect_spike_plot=False,
        plot_spikes=False,
        noisy_data=False,
        noise_level=0.005,
        add_breaks=False,
        break_lengths=[500, 1500, 1000],
        gain=1.0,
        test_data_ratio=0.5,
        max_time=2000,
        plot_heat_map=False,
        retur=False,
        num_steps=1000,
        train_=True,
        offset=0,
        first_spike_time=0,
        time_var_input=False,
        min_time=None,
        gain_labels=0.5,
    ):
        if num_images <= self.N_classes:
            raise ValueError(
                f"num images {num_images} needs to be more than number of classes ({self.N_classes})"
            )
        # Save current parameters
        self.data_parameters = {**locals()}

        # Copy and remove class element to dict
        list = [
            "plot_spikes",
            "plot_heat_map",
            "plot_comparison",
            "retur",
            "force_recreate",
            "self",
        ]

        # Remove elements from data_parameters
        for element in list:
            del self.data_parameters[element]
        self.data_parameters["supervised"] = self.supervised
        self.data_parameters["unsupervised"] = self.unsupervised

        # Update model
        self.data_parameters.update()

        # set parameters
        self.num_steps = num_steps
        self.num_items = num_images

        # check correct num_images
        if self.num_items % self.N_classes != 0:
            raise ValueError(
                f"The number of images are not divisible by the number classes without a remaining fraction. It should be divisble by {self.N_classes}"
            )

        # create data
        if not force_recreate:
            self.process(load_data=True, data_parameters=self.data_parameters)

        if force_recreate or not self.data_loaded:
            # Define data parameters
            data_parameters = {"pixel_size": int(np.sqrt(self.N_x)), "train_": train_}

            # Define folder to load data
            folders = os.listdir("data/mdata")

            # Search for existing data
            if len(folders) > 0:
                for folder in folders:
                    json_file_path = os.path.join(
                        "data", "mdata", folder, "data_parameters.json"
                    )

                    with open(json_file_path, "r") as j:
                        ex_params = json.loads(j.read())

                    # Check if parameters are the same as the current ones
                    if ex_params == data_parameters:
                        data_dir = os.path.join("data/mdata", folder)
                        download = False
                        break
                else:
                    download = True
            else:
                download = True

            # get dataset with progress bar
            print("Downloading MNIST dataset...")
            if download == True:
                # generate random number to create unique folder
                rand_nums = np.random.randint(low=0, high=9, size=5)

                # Check if folder already exists
                while any(item in os.listdir("data") for item in rand_nums):
                    rand_nums = np.random.randint(low=0, high=9, size=5)

                # Create folder to store data
                data_dir = os.path.join("data/mdata", str(rand_nums))
                os.makedirs(data_dir)

                # Save data parameters
                filepath = os.path.join(data_dir, "data_parameters.json")

                with open(filepath, "w") as outfile:
                    json.dump(data_parameters, outfile)

            if self.unsupervised or self.supervised:
                true_labels = True
            else:
                true_labels = False

            (
                self.data_train,
                self.labels_train,
                self.data_test,
                self.labels_test,
                self.labels_true_train,
                self.labels_true_test,
            ) = create_data(
                pixel_size=int(np.sqrt(self.N_x)),
                num_steps=num_steps,
                plot_comparison=plot_comparison,
                gain=gain,
                gain_labels=gain_labels,
                train_=train_,
                offset=offset,
                download=download,
                data_dir=data_dir,
                true_labels=true_labels,
                N_classes=self.N_classes,
                first_spike_time=first_spike_time,
                time_var_input=time_var_input,
                num_images=num_images,
                add_breaks=add_breaks,
                break_lengths=break_lengths,
                noisy_data=noisy_data,
                noise_level=noise_level,
                classes=self.classes,
                test_data_ratio=test_data_ratio,
            )
            self.process(save_data=True, data_parameters=self.data_parameters)

        # get data shape
        self.T_train = self.data_train.shape[0]
        self.T_test = self.data_test.shape[0]

        # plot spikes
        if plot_spikes:
            if min_time == None:
                min_time = 0
            if max_time == None:
                max_time = self.T_train
            spike_plot(
                self.data_train[min_time:max_time], self.labels_train[min_time:max_time]
            )

        # inspect spike labels
        if inspect_spike_plot:
            spike_plot(data=self.labels_true_train, labels=None)

        # plot heatmap of activity
        if plot_heat_map:
            heat_map(self.data_train, pixel_size=10)

        # return data and labels if needed
        if retur:
            return self.data_train, self.labels_train

    def prepare_training(
        self,
        plot_weights=False,
        plot_network=False,
        w_dense_ee=0.01,
        w_dense_se=0.05,
        w_dense_ei=0.05,
        w_dense_ie=0.1,
        weight_affinity_output=0.33,
        resting_membrane=-70,
        max_time=100,
        retur=False,
        epp_weight=1,
        epn_weight=1,
        pp_weight=1,
        pn_weight=-1,
        tp_weight=1,
        tn_weight=1,
        fp_weight=-1,
        fn_weight=-1,
        se_weights=0.1,
        ee_weights=0.3,
        ei_weights=0.3,
        ie_weights=-0.2,
    ):
        # create weights
        self.w_dense_ee = w_dense_ee
        self.w_dense_ei = w_dense_ei
        self.w_dense_ie = w_dense_ie
        self.w_dense_se = w_dense_se
        self.se_weights = se_weights
        self.ee_weights = ee_weights
        self.ei_weights = ei_weights
        self.ie_weights = ie_weights
        self.weight_affinity_output = weight_affinity_output
        self.weights = create_weights(
            N_exc=self.N_exc,
            N_inh=self.N_inh,
            unsupervised=self.unsupervised,
            N=self.N,
            N_x=self.N_x,
            N_classes=self.N_classes,
            supervised=self.supervised,
            w_dense_ee=w_dense_ee,
            w_dense_ei=w_dense_ei,
            w_dense_se=w_dense_se,
            w_dense_ie=w_dense_ie,
            se_weights=se_weights,
            ee_weights=ee_weights,
            ei_weights=ei_weights,
            ie_weights=ie_weights,
            weight_affinity_output=weight_affinity_output,
            plot_weights=plot_weights,
            plot_network=plot_network,
            epp_weight=epp_weight,
            epn_weight=epn_weight,
            pp_weight=pp_weight,
            pn_weight=pn_weight,
            tp_weight=tp_weight,
            tn_weight=tn_weight,
            fp_weight=fp_weight,
            fn_weight=fn_weight,
        )
        self.resting_potential = resting_membrane
        self.max_time = max_time

        # create other arrays
        (
            self.mp_train,
            self.mp_test,
            self.pre_trace,
            self.post_trace,
            self.spikes_train,
            self.spikes_test,
            self.spike_times,
        ) = create_arrays(
            N=self.N,
            N_exc=self.N_exc,
            N_inh=self.N_inh,
            resting_membrane=self.resting_potential,
            total_time_train=self.T_train,
            total_time_test=self.T_test,
            data_train=self.data_train,
            data_test=self.data_test,
            supervised=self.supervised,
            unsupervised=self.unsupervised,
            N_classes=self.N_classes,
            N_x=self.N_x,
            max_time=self.max_time,
            labels_true=self.labels_true_train,
        )
        # return results if retur == True
        if retur:
            return (
                self.weights,
                self.spikes_train,
                self.spikes_test,
                self.mp_train,
                self.mp_test,
                self.pre_trace,
                self.post_trace,
                self.spike_times,
                self.resting_potential,
                self.max_time,
            )

    def train(
        self,
        plot_spikes_train=False,
        plot_spikes_test=False,
        plot_mp_train=False,
        plot_mp_test=False,
        plot_weights=True,
        plot_threshold=False,
        plot_traces_=False,
        train_weights=False,
        learning_rate_exc=0.0008,
        learning_rate_inh=0.005,
        w_target_exc=0.1,
        w_target_inh=-0.1,
        var_noise=1,
        min_weight_inh=-25,
        max_weight_inh=0,
        max_weight_exc=25,
        min_weight_exc=0,
        spike_threshold_default=-55,
        check_sleep_interval=10000,
        interval=1000,
        min_mp=-100,
        sleep=True,
        force_train=False,
        save_model=True,
        weight_decay=False,
        weight_decay_rate_exc=0.9999,
        weight_decay_rate_inh=0.9999,
        noisy_potential=True,
        noisy_threshold=False,
        noisy_weights=False,
        spike_adaption=True,
        delta_adaption=0.5,
        tau_adaption=100,
        trace_update=False,
        timing_update=True,
        vectorized_trace=False,
        clip_exc_weights=False,
        clip_inh_weights=False,
        alpha=1.25,
        beta=0.8,
        A_plus=0.25,
        A_minus=0.5,
        test=True,
        tau_LTD=10,
        tau_LTP=10,
        dt=1,
        tau_m=30,
        membrane_resistance=100,
        reset_potential=-80,
        spike_slope=-0.1,
        spike_intercept=-4,
        w_interval=5,
        start_time_spike_plot=None,
        stop_time_spike_plot=None,
        start_index_mp=None,
        stop_index_mp=None,
        time_start_mp=None,
        time_stop_mp=None,
        mean_noise=0,
        max_mp=40,
        tau_pre_trace_exc=1,
        tau_pre_trace_inh=1,
        tau_post_trace_exc=1,
        tau_post_trace_inh=1,
        weight_mean_noise=0.05,
        weight_var_noise=0.005,
        random_selection_weight_plot=True,
        num_inh=10,
        num_exc=50,
        plot_accuracy_train=True,
        plot_accuracy_test=True,
        save_test_data=True,
        narrow_top=0.05,
        wide_top=0.15,
        smoothening=350,
        plot_top_response_train=False,
        plot_top_response_test=False,
    ):
        self.dt = dt

        # Save current parameters
        self.model_parameters = {**locals()}
        remove = [
            "self",
            "force_train",
            "save_model",
            "test",
            "plot_mp_train",
            "plot_mp_test",
            "plot_spikes_train",
            "plot_spikes_test",
            "plot_weights",
            "plot_threshold",
            "plot_traces_",
            "random_selection_weight_plot",
            "train_weights",
            "plot_accuracy_train",
            "plot_accuracy_test",
            "start_time_spike_plot",
            "stop_time_spike_plot",
            "start_index_mp",
            "stop_index_mp",
            "time_start_mp",
            "time_stop_mp",
            "save_test_data",
        ]

        # Remove elements from model_parameters
        for element in remove:
            self.model_parameters.pop(element)

        self.model_parameters["num_items"] = self.num_items
        self.model_parameters["supervised"] = self.supervised
        self.model_parameters["unsupervised"] = self.unsupervised
        self.model_parameters["w_dense_ee"] = self.w_dense_ee
        self.model_parameters["w_dense_se"] = self.w_dense_se
        self.model_parameters["w_dense_ei"] = self.w_dense_ei
        self.model_parameters["w_dense_ie"] = self.w_dense_ie
        self.model_parameters["ie_weights"] = self.ie_weights
        self.model_parameters["ee_weights"] = self.ee_weights
        self.model_parameters["ei_weights"] = self.ei_weights
        self.model_parameters["se_weights"] = self.se_weights
        self.model_parameters["weight_affinity_output"] = self.weight_affinity_output

        if not force_train and self.data_loaded:
            model_dir = self.process(
                load_model=True,
                model_parameters=self.model_parameters,
                load_test_model=False,
            )

        if not self.model_loaded or force_train:
            # train model
            (
                self.weights,
                self.spikes_train,
                self.pre_trace,
                self.post_trace,
                self.mp_train,
                self.weights2plot_exc,
                self.weights2plot_inh,
                self.pre_trace_plot,
                self.post_trace_plot,
                self.spike_threshold,
                self.weight_mask,
                self.max_weight_sum_inh,
                self.max_weight_sum_exc,
                self.labels_train,
            ) = train_network(
                weights=self.weights,
                spike_labels=self.labels_train,
                N_classes=self.N_classes,
                supervised=self.supervised,
                mp=self.mp_train,
                sleep=sleep,
                alpha=alpha,
                timing_update=timing_update,
                spikes=self.spikes_train,
                pre_trace=self.pre_trace,
                post_trace=self.post_trace,
                check_sleep_interval=check_sleep_interval,
                tau_pre_trace_exc=tau_pre_trace_exc,
                tau_pre_trace_inh=tau_pre_trace_inh,
                tau_post_trace_exc=tau_post_trace_exc,
                tau_post_trace_inh=tau_post_trace_inh,
                resting_potential=self.resting_potential,
                membrane_resistance=membrane_resistance,
                unsupervised=self.unsupervised,
                min_weight_exc=min_weight_exc,
                max_weight_exc=max_weight_exc,
                min_weight_inh=min_weight_inh,
                max_weight_inh=max_weight_inh,
                N_inh=self.N_inh,
                N_exc=self.N_exc,
                beta=beta,
                num_exc=num_exc,
                num_inh=num_inh,
                weight_decay=weight_decay,
                weight_decay_rate_exc=weight_decay_rate_exc,
                weight_decay_rate_inh=weight_decay_rate_inh,
                train_weights=train_weights,
                learning_rate_exc=learning_rate_exc,
                learning_rate_inh=learning_rate_inh,
                w_interval=w_interval,
                interval=interval,
                w_target_exc=w_target_exc,
                w_target_inh=w_target_inh,
                tau_LTP=tau_LTP,
                tau_LTD=tau_LTD,
                tau_m=tau_m,
                max_mp=max_mp,
                min_mp=min_mp,
                dt=self.dt,
                N=self.N,
                clip_exc_weights=clip_exc_weights,
                clip_inh_weights=clip_inh_weights,
                A_plus=A_plus,
                A_minus=A_minus,
                trace_update=trace_update,
                spike_adaption=spike_adaption,
                delta_adaption=delta_adaption,
                tau_adaption=tau_adaption,
                spike_threshold_default=spike_threshold_default,
                spike_intercept=spike_intercept,
                spike_slope=spike_slope,
                noisy_threshold=noisy_threshold,
                reset_potential=reset_potential,
                spike_times=self.spike_times,
                noisy_potential=noisy_potential,
                noisy_weights=noisy_weights,
                weight_mean_noise=weight_mean_noise,
                weight_var_noise=weight_var_noise,
                vectorized_trace=vectorized_trace,
                N_x=self.N_x,
                T=self.T_train,
                mean_noise=mean_noise,
                var_noise=var_noise,
            )

        if save_model and not self.model_loaded:
            model_dir = self.process(
                save_model=True,
                model_parameters=self.model_parameters,
                load_test_model=False,
            )
        """
        For some reason, labels_train has no "sleep" labels when I load it in. This makes no sense. 
        
        """
        if plot_top_response_train:
            top_responders_plotted(
                spikes=self.spikes_train,
                labels=self.labels_train,
                ih=self.ih,
                st=self.st,
                num_classes=self.N_classes,
                narrow_top=narrow_top,
                smoothening=smoothening,
                train=True,
                wide_top=wide_top,
            )

        if plot_accuracy_train and (self.supervised or self.unsupervised):
            self.accuracy_train = plot_accuracy(
                spikes=self.spikes_train,
                ih=self.ih,
                pp=self.pp,
                pn=self.pn,
                tp=self.tp,
                labels=self.labels_train,
                num_steps=self.num_steps,
                num_classes=self.N_classes,
                test=False,
            )

        if plot_spikes_train:
            if start_time_spike_plot == None:
                start_time_spike_plot = int(self.spikes_train.shape[0] * 0.95)
            if stop_time_spike_plot == None:
                stop_time_spike_plot = self.spikes_train.shape[0]

            spike_plot(
                self.spikes_train[
                    start_time_spike_plot:stop_time_spike_plot, self.st :
                ],
                self.labels_train[start_time_spike_plot:stop_time_spike_plot],
            )

        if plot_threshold:
            spike_threshold_plot(self.spike_threshold, self.N_exc)

        if plot_mp_train:
            if start_index_mp == None:
                start_index_mp = self.ex
            if stop_index_mp == None:
                stop_index_mp = self.ih
            if time_start_mp == None:
                time_start_mp = int(self.T_train * 0.95)
            if time_stop_mp == None:
                time_stop_mp = self.T_train

            mp_plot(
                mp=self.mp_train[time_start_mp:time_stop_mp],
                N_exc=self.N_exc,
            )

        if plot_weights:
            weights_plot(
                weights_exc=self.weights2plot_exc[
                    int(self.weights2plot_exc.shape[0] * 0.80) :
                ],
                weights_inh=self.weights2plot_inh[
                    int(self.weights2plot_inh.shape[0] * 0.80) :
                ],
                N=self.N,
                N_x=self.N_x,
                N_exc=self.N_exc,
                N_inh=self.N_inh,
                max_weight_sum_inh=self.max_weight_sum_inh,
                max_weight_sum_exc=self.max_weight_sum_exc,
                random_selection=random_selection_weight_plot,
            )

        if plot_traces_:
            plot_traces(
                N_exc=self.N_exc,
                N_inh=self.N_inh,
                pre_traces=self.pre_trace_plot,
                post_traces=self.post_trace_plot,
            )

        if test:
            # test model

            # check for previous test models saved
            self.process(load_model=True, load_test_model=True, model_dir_=model_dir)

            # train if no model were found
            if not self.test_data_loaded:
                (
                    self.weights,
                    self.spikes_test,
                    self.pre_trace,
                    self.post_trace,
                    self.mp_test,
                    self.weights2plot_exc,
                    self.weights2plot_inh,
                    self.pre_trace_plot,
                    self.post_trace_plot,
                    self.spike_threshold,
                    self.weight_mask,
                    self.max_weight_sum_inh,
                    self.max_weight_sum_exc,
                    self.labels_test,
                ) = train_network(
                    weights=self.weights,
                    spike_labels=self.labels_test,
                    N_classes=self.N_classes,
                    supervised=self.supervised,
                    mp=self.mp_test,
                    sleep=False,
                    alpha=alpha,
                    timing_update=timing_update,
                    spikes=self.spikes_test,
                    pre_trace=self.pre_trace,
                    post_trace=self.post_trace,
                    unsupervised=self.unsupervised,
                    check_sleep_interval=check_sleep_interval,
                    tau_pre_trace_exc=tau_pre_trace_exc,
                    tau_pre_trace_inh=tau_pre_trace_inh,
                    tau_post_trace_exc=tau_post_trace_exc,
                    tau_post_trace_inh=tau_post_trace_inh,
                    resting_potential=self.resting_potential,
                    membrane_resistance=membrane_resistance,
                    min_weight_exc=min_weight_exc,
                    max_weight_exc=max_weight_exc,
                    min_weight_inh=min_weight_inh,
                    max_weight_inh=max_weight_inh,
                    N_inh=self.N_inh,
                    N_exc=self.N_exc,
                    beta=beta,
                    num_exc=num_exc,
                    num_inh=num_inh,
                    weight_decay=weight_decay,
                    weight_decay_rate_exc=weight_decay_rate_exc,
                    weight_decay_rate_inh=weight_decay_rate_inh,
                    train_weights=False,
                    learning_rate_exc=learning_rate_exc,
                    learning_rate_inh=learning_rate_inh,
                    w_interval=w_interval,
                    interval=interval,
                    w_target_exc=w_target_exc,
                    w_target_inh=w_target_inh,
                    tau_LTP=tau_LTP,
                    tau_LTD=tau_LTD,
                    tau_m=tau_m,
                    max_mp=max_mp,
                    min_mp=min_mp,
                    dt=self.dt,
                    N=self.N,
                    clip_exc_weights=clip_exc_weights,
                    clip_inh_weights=clip_inh_weights,
                    A_plus=A_plus,
                    A_minus=A_minus,
                    trace_update=trace_update,
                    spike_adaption=spike_adaption,
                    delta_adaption=delta_adaption,
                    tau_adaption=tau_adaption,
                    spike_threshold_default=spike_threshold_default,
                    spike_intercept=spike_intercept,
                    spike_slope=spike_slope,
                    noisy_threshold=noisy_threshold,
                    reset_potential=reset_potential,
                    spike_times=self.spike_times,
                    noisy_potential=noisy_potential,
                    noisy_weights=noisy_weights,
                    weight_mean_noise=weight_mean_noise,
                    weight_var_noise=weight_var_noise,
                    vectorized_trace=vectorized_trace,
                    N_x=self.N_x,
                    T=self.T_test,
                    mean_noise=mean_noise,
                    var_noise=var_noise,
                )
                if save_test_data:
                    self.process(
                        save_model=True, save_test_model=True, model_dir_=model_dir
                    )
            if plot_top_response_test:
                top_responders_plotted(
                    spikes=self.spikes_test,
                    labels=self.labels_test,
                    ih=self.ih,
                    st=self.st,
                    num_classes=self.N_classes,
                    narrow_top=narrow_top,
                    smoothening=smoothening,
                    train=False,
                    wide_top=wide_top,
                )

            if plot_accuracy_test and (self.unsupervised or self.supervised):
                self.spikes_test[:, self.pn : self.tn] = self.labels_true_test
                self.accuracy_test = plot_accuracy(
                    spikes=self.spikes_test,
                    ih=self.ih,
                    pp=self.pp,
                    pn=self.pn,
                    tp=self.tp,
                    labels=self.labels_test,
                    num_steps=self.num_steps,
                    num_classes=self.N_classes,
                    test=True,
                )

            if plot_spikes_test:
                if start_time_spike_plot == None:
                    start_time_spike_plot = int(self.T_test * 0.95)
                if stop_time_spike_plot == None:
                    stop_time_spike_plot = self.T_test

                spike_plot(
                    self.spikes_test[start_time_spike_plot:stop_time_spike_plot],
                    self.labels_test[start_time_spike_plot:stop_time_spike_plot],
                )

            if plot_mp_test:
                if start_index_mp == None:
                    start_index_mp = self.N_x
                if stop_index_mp == None:
                    stop_index_mp = self.N_exc + self.N_inh
                if time_start_mp == None:
                    time_start_mp = 0
                if time_stop_mp == None:
                    time_stop_mp = self.T_test

                mp_plot(
                    mp=self.mp_test[time_start_mp:time_stop_mp],
                    N_exc=self.N_exc,
                )

    def analysis(
        self,
        perplexity=8,
        max_iter=1000,
        random_state=48,
        n_components=2,
        t_sne=False,
        t_sne_train=False,
        t_sne_test=True,
        pca=False,
        pca_train=False,
        pca_test=True,
        clustering_estimation=True,
    ):
        if t_sne:
            if t_sne_train:
                t_SNE(
                    spikes=self.spikes_train[:, self.st : self.ex],
                    labels_spike=self.labels_train,
                    n_components=n_components,
                    perplexity=perplexity,
                    max_iter=max_iter,
                    random_state=random_state,
                    accuracy=self.accuracy_train,
                )
            if t_sne_test:
                t_SNE(
                    spikes=self.spikes_test[:, self.st : self.ex],
                    labels_spike=self.labels_test,
                    n_components=n_components,
                    perplexity=perplexity,
                    max_iter=max_iter,
                    random_state=random_state,
                    accuracy=self.accuracy_test,
                )
        if pca:
            if pca_train:
                PCA_analysis(
                    spikes=self.spikes_train[:, self.N_x :],
                    labels_spike=self.labels_train,
                    n_components=n_components,
                    random_state=random_state,
                )
            if pca_test:
                PCA_analysis(
                    spikes=self.spikes_test[:, self.N_x :],
                    labels_spike=self.labels_test,
                    n_components=n_components,
                    random_state=random_state,
                )
        if clustering_estimation:
            Clustering_estimation(
                spikes_train=self.spikes_train,
                spikes_test=self.spikes_test,
                labels_train=self.labels_train,
                labels_test=self.labels_test,
                num_steps=self.num_steps,
                n_components=n_components,
                random_state=random_state,
                num_classes=self.N_classes,
            )
