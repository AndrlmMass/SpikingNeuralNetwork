import numpy as np
import os
import gc
from tqdm import tqdm
import json
from datetime import datetime
from src.core.trainer import Trainer
from src.plot.spikes import (
    gif_spike_rate_by_label,
)
from src.plot.training import PCAScatterDisplay, plot_accuracy
from src.evaluation.evaluation import Evaluator
from src.utils.performance import start_plot_worker, stop_plot_worker
from src.network.init_network import create_weights, create_arrays
from src.plot.training import plot_accuracy


class snn_sleepy:
    def __init__(
        self,
        N_exc=200,
        N_inh=50,
        N_x=225,
        ts_spec=None,
        random_state=0,
        classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    ):
        self.N_exc = N_exc
        self.N_inh = N_inh
        self.N_x = N_x
        self.pixel_size = int(np.sqrt(N_x))
        self.rng = np.random.default_rng(random_state)
        self.N_classes = len(classes)
        self.classes = classes
        self.st = N_x  # stimulation
        self.ex = self.st + N_exc  # excitatory
        self.ih = self.ex + N_inh  # inhibitory
        self.N = N_exc + N_inh + N_x
        # One-time plotting guard
        self._did_plot_spectrograms = False
        self._image_preview_done = False
        # Accuracy tracking
        self.acc_history = {"train": {}, "val": {}, "test": {}}
        self._acc_log_dir = None
        self._acc_log_file = None
        self.ts_spec = ts_spec
        self.ts = datetime.now().strftime("%Y.%m.%d")

    def spikes_per_item(self, spikes, labels):
        T = spikes.shape[0]
        t = self.num_steps
        N = spikes.shape[1]
        spikes = spikes.reshape(T // t, t, N).mean(axis=1)
        labels = labels.reshape(T // t, t).mean(axis=1).astype(int)
        return spikes, labels

    def pad_to_match(self, a, b, pad_value=0):
        len_a, len_b = len(a), len(b)

        if len_a < len_b:
            a = np.pad(a, (0, len_b - len_a), constant_values=pad_value)
        elif len_b < len_a:
            b = np.pad(b, (0, len_a - len_b), constant_values=pad_value)

        return a, b

    def _save_streaming_parameters(self):
        """Save streaming data parameters for future reference."""
        if not hasattr(self, "data_parameters"):
            return

        # Create a combined parameters file for streaming data
        streaming_params = {
            "streaming_mode": True,
            "audio_streamer": self.audio_streamer is not None,
            "image_streamer": self.image_streamer is not None,
            "N_x": self.N_x,
            "N_exc": self.N_exc,
            "N_inh": self.N_inh,
            "batch_train": self.batch_train,
            "batch_test": self.batch_test,
            "batch_val": self.batch_val,
            "all_train": self.all_train,
            "all_test": self.all_test,
            "all_val": self.all_val,
            "epochs": self.batches,
        }

        # Save to a streaming parameters file
        streaming_file = "data/mdata/streaming_parameters.json"
        os.makedirs("data/mdata", exist_ok=True)

        with open(streaming_file, "w") as f:
            json.dump(streaming_params, f, indent=2)

        print(f"Streaming parameters saved to {streaming_file}")

    # Removed inline audio visualization; use plot.plot_audio_preview_from_streamer instead

    def get_training_mode(self):
        """Determine the current training mode based on active streamers."""
        if (
            hasattr(self, "audio_streamer")
            and self.audio_streamer is not None
            and hasattr(self, "image_streamer")
            and self.image_streamer is not None
        ):
            return "multimodal"
        elif hasattr(self, "audio_streamer") and self.audio_streamer is not None:
            return "audio_only"
        elif hasattr(self, "image_streamer") and self.image_streamer is not None:
            return "image_only"
        else:
            return "unknown"

    def prepare_network(
        self,
        plot_weights=False,
        w_dense_ee=0.01,
        w_dense_se=0.05,
        w_dense_ei=0.05,
        w_dense_ie=0.05,
        resting_membrane=-70,
        max_time=100,
        retur=False,
        se_weights=0.1,
        ee_weights=0.3,
        ei_weights=0.3,
        ie_weights=-0.2,
        random_weights=False,
        create_network=False,
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
        self.resting_potential = resting_membrane
        self.max_time = max_time

        self.weights = create_weights(
            N_exc=self.N_exc,
            N_inh=self.N_inh,
            N=self.N,
            N_x=self.N_x,
            w_dense_ee=w_dense_ee,
            w_dense_ei=w_dense_ei,
            w_dense_se=w_dense_se,
            w_dense_ie=w_dense_ie,
            se_weights=se_weights,
            ee_weights=ee_weights,
            ei_weights=ei_weights,
            ie_weights=ie_weights,
            plot_weights=plot_weights,
            random_weights=random_weights,
        )

        if create_network:
            # create other arrays
            (
                self.mp_train,
                self.mp_test,
                self.spikes_train,
                self.spikes_test,
                self.spike_trace,
            ) = create_arrays(
                N=self.N,
                N_exc=self.N_exc,
                N_inh=self.N_inh,
                resting_membrane=self.resting_potential,
                total_time_train=self.T_train,
                total_time_test=self.T_test,
                data_train=None,  # Streaming mode - data loaded on-demand
                data_test=None,  # Streaming mode - data loaded on-demand
                N_x=self.N_x,
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
                    self.resting_potential,
                    self.max_time,
                )

    def train_network(
        self,
        train_weights=False,
        tau_trace=25,
        learning_rate=0.0008,
        var_noise=1,
        min_weight_inh=-25,
        track_weights=False,
        max_weight_inh=-0.01,
        max_weight_exc=25,
        min_weight_exc=0.01,
        spike_threshold_default=-55,
        min_mp=-100,
        sleep=False,
        normalize_weights=False,  # Alternative to sleep: maintain initial weight sum
        save_model=True,
        spike_adaption=True,
        delta_adaption=1,
        tau_adaption=100,
        clip_weights=False,
        A_plus=0.5,
        A_minus=0.5,
        tau_LTD=10,
        tau_LTP=10,
        w_max=10,
        early_stopping=False,  # reimplement this!
        early_stopping_patience_pct=0.1,  # Patience as percentage of total epochs (0.1 = 10%)
        dt=1,
        membrane_resistance_exc=30,
        membrane_resistance_inh=30,
        reset_potential=-80,
        pca_variance=0.95,
        mean_noise=0,
        max_mp=40,
        heatmap_plot=False,
        get_giffed=False,
        mu_weight=0.6,
        tau_syn_exc=30,
        tau_syn_inh=30,
        tau_m_exc=30,
        tau_m_inh=30,
        use_validation_data=False,  # passed as false - WHY?
        accuracy_method="top",
        use_QDA=False,
        use_LR=True,
        reg_mode="static",
        track_stats=False,
        use_phi=True,
        use_pca=True,
        PCA_plot=True,
        gif_pca_plot=True,
        gif_spikes_plot=True,
        profile=False,
        reg_frequency=1050,  # need to ensure that this is divisible by the number of samples so that regularization doesnt split batches
        sleep_duration=300,
        stat_tracking_frequency=1000,
        update_weights_freq=100,
    ):
        self.dt = dt
        self.pca_variance = pca_variance
        self.use_validation_data = use_validation_data
        self.use_QDA = use_QDA
        self.use_LR = use_LR
        self.normalize_weights = normalize_weights

        # ensure conditions are met
        if accuracy_method != "pca_lr" and PCA_plot:
            raise ValueError("PCA_plot can only be True if accuracy_method is 'pca_lr'")

        # Save current parameters
        self.model_parameters = {**locals()}
        self.model_parameters["all_train"] = self.all_train
        self.model_parameters["all_test"] = self.all_test
        self.model_parameters["all_val"] = self.all_val
        self.model_parameters["batch_train"] = self.batch_train
        self.model_parameters["batch_test"] = self.batch_test
        self.model_parameters["batch_val"] = self.batch_val
        self.model_parameters["epochs"] = self.batches
        self.model_parameters["w_dense_ee"] = self.w_dense_ee
        self.model_parameters["w_dense_ei"] = self.w_dense_ei
        self.model_parameters["w_dense_se"] = self.w_dense_se
        self.model_parameters["w_dense_ie"] = self.w_dense_ie

        self.model_parameters["ie_weights"] = self.ie_weights
        self.model_parameters["ee_weights"] = self.ee_weights
        self.model_parameters["ei_weights"] = self.ei_weights
        self.model_parameters["se_weights"] = self.se_weights
        self.model_parameters["classes"] = self.classes
        del self.model_parameters[
            "self"
        ]  # Remove self reference to avoid issues with JSON serialization

        # prepare variables for training
        if reg_mode == "static":
            initial_sums_se = np.zeros(1)
            initial_sums_ee = np.zeros(1)
        elif reg_mode == "post":
            initial_sums_se = self.weights[: self.st, self.st : self.ex].sum(axis=0)
            initial_sums_ee = self.weights[self.st : self.ex, self.st : self.ex].sum(
                axis=0
            )
        elif reg_mode == "layer":
            initial_sums_se = self.weights[: self.st, self.st : self.ex].sum()
            initial_sums_ee = self.weights[self.st : self.ex, self.st : self.ex].sum()

        mp_new = np.zeros(
            (self.ih - self.st)
        )  # Create generic mp_new for all uses (train, val and test)
        nonzero_pre_idx = []
        for i in range(self.st, self.ex):  # loop over postsynapses that receive STDP
            nonzero_pre_idx.append(np.nonzero(self.weights[: self.ex, i])[0])
        nz_rows_se, nz_cols_se = np.nonzero(self.weights[: self.st, self.st : self.ex])
        nz_rows_ee, nz_cols_ee = np.nonzero(
            self.weights[self.st : self.ex, self.st : self.ex]
        )
        nz_rows_exc, nz_cols_exc = np.nonzero(
            self.weights[: self.ex, self.st : self.ex]
        )
        nz_cols_exc += self.st  # → global

        # initiate the evaluator
        eval = Evaluator(
            xp_var_or_comps=pca_variance,
            num_classes=self.N_classes,
            do_phi=use_phi,
            do_LR=use_LR,
            do_pca=use_pca,
        )
        # initiate the trainer
        trainer = Trainer(
            resting_potential=self.resting_potential,
            membrane_resistance_exc=membrane_resistance_exc,
            membrane_resistance_inh=membrane_resistance_inh,
            update_weights_freq=update_weights_freq,
            reg_frequency=reg_frequency,
            sleep_duration=sleep_duration,
            stat_tracking_frequency=stat_tracking_frequency,
            track_stats=track_stats,
            track_weights=track_weights,
            min_weight_exc=min_weight_exc,
            max_weight_exc=max_weight_exc,
            min_weight_inh=min_weight_inh,
            max_weight_inh=max_weight_inh,
            train_weights=train_weights,
            N_x=self.N_x,
            N_inh=self.N_inh,
            N_exc=self.N_exc,
            learning_rate=learning_rate,
            tau_LTP=tau_LTP,
            tau_LTD=tau_LTD,
            max_mp=max_mp,
            min_mp=min_mp,
            w_target_se=self.se_weights,
            w_target_ee=self.ee_weights,
            w_max=w_max,
            clip_weights=clip_weights,
            dt=self.dt,
            run=self.ts_spec,
            A_plus=A_plus,
            A_minus=A_minus,
            tau_m_exc=tau_m_exc,
            tau_m_inh=tau_m_inh,
            sleep=sleep,
            spike_adaption=spike_adaption,
            tau_adaption=tau_adaption,
            delta_adaption=delta_adaption,
            spike_threshold_default=spike_threshold_default,
            save_plots=heatmap_plot,
            reset_potential=reset_potential,
            initial_sums_se=initial_sums_se,
            initial_sums_ee=initial_sums_ee,
            dataset=self.image_dataset,
            plot_iterations=self.num_steps,
            tau_trace=tau_trace,
            tau_syn_exc=tau_syn_exc,
            tau_syn_inh=tau_syn_inh,
            mean_noise=mean_noise,
            var_noise=var_noise,
            mu_weight=mu_weight,
            st=self.st,
            ex=self.ex,
            ih=self.ih,
            mp_new=mp_new,
            time_per_item=self.num_steps,
            normalize_weights=normalize_weights,
            nonzero_pre_idx=nonzero_pre_idx,
            reg_mode=reg_mode,
            nz_rows_se=nz_rows_se,
            nz_cols_se=nz_cols_se,
            nz_rows_ee=nz_rows_ee,
            nz_cols_ee=nz_cols_ee,
            nz_cols_exc=nz_cols_exc,
            nz_rows_exc=nz_rows_exc,
        )

        # Always attempt to load a matching model if not forcing a fresh run,
        # regardless of data_loaded (streaming modes don't set data_loaded).
        # In test-only inference with frozen weights, skip loading any saved model
        if not self.model_loaded:
            # Handle data parameter checking for streaming data
            data_dir = None
            download = False

            # Check if we're using streaming data
            if self.audio_streamer is not None or self.image_streamer is not None:
                # For streaming data, we don't need to check data_parameters.json
                # as the streamers handle their own data loading
                print("Using streaming data - no data parameter checking needed")

                # Log which streamers are active
                if self.audio_streamer is not None and self.image_streamer is not None:
                    print("Multimodal streaming: Audio + Image")
                elif self.audio_streamer is not None:
                    print("Audio streaming only")
                elif self.image_streamer is not None:
                    print("Image streaming only")

                # Save training parameters for streaming data
                self._save_streaming_parameters()
            else:
                # For pre-loaded data, check data_parameters.json
                data_parameters = {"pixel_size": self.pixel_size, "train_": True}

                # Ensure data/mdata directory exists
                if not os.path.exists("data/mdata"):
                    os.makedirs("data/mdata", exist_ok=True)

                # Define folder to load data
                folders = os.listdir("data/mdata")

                # Search for existing data
                if len(folders) > 0:
                    for folder in folders:
                        json_file_path = os.path.join("data", "mdata", folder)
                        if os.path.exists(json_file_path):
                            with open(json_file_path, "r") as j:
                                ex_params = json.loads(j.read())
                            # Check if parameters are the same as the current ones
                            if ex_params == data_parameters:
                                data_dir = os.path.join("data/mdata", folder)
                                download = True
                                break
                if data_dir is None:
                    print("Could not find the mdata directory.")
                    download = False
                    data_dir = None

            # pre-define performance tracking array
            if not train_weights:
                # single pass; tracker will be (1,2) later
                self.performance_tracker = np.zeros((1, 2))
            else:
                self.performance_tracker = np.zeros((self.batches, 2))

            # define progress bar
            pbar_total = 1 if (not train_weights) else self.batches
            pbar = tqdm(
                total=pbar_total,
                desc=(
                    "Test-only" if (not train_weights) else f"Epoch 0/{self.batches}:"
                ),
                unit="it",
                ncols=80,
                bar_format="{desc} [{bar}] ETA: {remaining} |{postfix}",
            )
            # Predefine outputs to avoid UnboundLocalError if loop exits early
            spikes_tr_out = None
            labels_tr_out = None
            thresh_tr = None
            spikes_te_out = None
            labels_te_out = None

            # create missing arrays
            I_syn_exc = np.zeros(self.N_exc)
            I_syn_inh = np.zeros(self.N_inh)
            a = np.zeros(self.N_exc + self.N_inh)

            # create spike threshold array
            spike_threshold = np.full(
                shape=(self.ih - self.st),
                fill_value=spike_threshold_default,
                dtype=float,
            )

            # Track sleep percentages across epochs
            best_val = 0.0

            # loop over self.batches
            for e in range(self.batches):
                # Reset test/val indices at the beginning of each epoch
                self.current_test_idx = 0
                # Reset streamer validation/train pointers (streamers ignore start_idx)
                try:
                    if (
                        hasattr(self, "image_streamer")
                        and self.image_streamer is not None
                    ):
                        self.image_streamer.reset_partition("val")
                except Exception:
                    pass

                if self.image_streamer is not None:
                    # Image only streaming mode (missing branch)
                    from data.get_data import load_image_batch

                    train_start_idx = self.current_train_idx
                    data_train, labels_train = load_image_batch(
                        self.image_streamer,
                        train_start_idx,
                        self.batch_train,
                        self.num_steps,
                        int(np.sqrt(self.N_x)) ** 2,  # Image-only uses full N_x
                    )
                    if data_train is None:
                        # Wrap around: reset pointer and re-fetch
                        self.image_streamer.reset_partition("train")
                        self.current_train_idx = 0
                        data_train, labels_train = load_image_batch(
                            self.image_streamer,
                            0,
                            self.batch_train,
                            self.num_steps,
                            int(np.sqrt(self.N_x)) ** 2,
                        )
                        if data_train is None:
                            print(f"No image data available at epoch {e}")
                            break
                    # Advance training index for streaming
                    self.current_train_idx += self.batch_train
                else:
                    # Use pre-loaded data (for image data)
                    if self.data_train is not None:
                        data_train = self.data_train
                        labels_train = self.labels_train
                    else:
                        print("No data available")
                        break

                # Update T_train and T_test to match the actual data shapes
                self.T_train = data_train.shape[0]

                # Debug: Check data dimensions and training mode
                training_mode = self.get_training_mode()
                print(f"Training mode: {training_mode}")
                print(f"Loaded data shape: {data_train.shape}")
                print(f"Expected N_x: {self.N_x}")
                print(f"Expected st (stimulation neurons): {self.N_x}")

                # Create & fetch necessary arrays (only if not pre-allocated)
                if not hasattr(self, "_arrays_pre_allocated"):
                    (
                        mp_train,
                        __,
                        spikes_train,
                        __,
                        spike_trace,
                    ) = create_arrays(
                        N=self.N,
                        N_exc=self.N_exc,
                        N_inh=self.N_inh,
                        resting_membrane=self.resting_potential,
                        total_time_train=self.T_train,
                        total_time_test=0,
                        data_train=data_train,
                        data_test=None,
                        N_x=self.N_x,
                    )
                    self._arrays_pre_allocated = True
                    print("inside!")
                else:
                    # add input data to array
                    spikes_train = np.zeros((self.T_train, self.N), dtype=np.int8)
                    spikes_train[:, : self.N_x] = data_train
                    if mp_train is None:
                        # Reuse pre-allocated arrays
                        mp_train = np.zeros(self.N - self.N_x)
                        mp_train[:] = self.resting_potential
                    if spike_trace is None:
                        spike_trace = np.zeros(self.N - self.N_inh)
                del data_train
                gc.collect()  # Force garbage collection to free memory
                if profile:
                    import cProfile

                    profiler = cProfile.Profile()
                    profiler.enable()
                if heatmap_plot:
                    start_plot_worker()
                # train network
                (
                    self.weights,
                    spikes_tr_out,  # delete after
                    mp_train,
                    thresh_tr,
                    labels_tr_out,  # delete after
                    I_syn_exc,
                    I_syn_inh,
                    a,
                    spike_trace,
                    x_tar_se,
                    x_tar_ee,
                ) = trainer.step(
                    weights=self.weights,
                    mp=mp_train,
                    spikes=spikes_train,
                    spike_labels=labels_train,
                    spike_trace=spike_trace,
                    training_mode="train",
                    spike_threshold=spike_threshold,
                    I_syn_exc=I_syn_exc,
                    I_syn_inh=I_syn_inh,
                    a=a,
                    x_tar_se=x_tar_se,
                    x_tar_ee=x_tar_ee,
                )
                if heatmap_plot:
                    stop_plot_worker()

                if profile:
                    profiler.disable()
                    dir = os.path.join("profile", self.image_dataset, self.ts)
                    os.makedirs(dir, exist_ok=True)
                    final_dir = os.path.join(dir, f"{self.ts_spec}.prof")
                    profiler.dump_stats(final_dir)
                spike_threshold = thresh_tr

                # plot gif
                if get_giffed:
                    self.plot_spikes()

                # Calculate training accuracy for current epoch
                if spikes_tr_out is not None and labels_tr_out is not None:
                    print(
                        f"\n--- Epoch {e+1} Training Accuracy ({accuracy_method.upper()}) ---"
                    )

                    if accuracy_method == "pca_lr":
                        # Debug: Check network activity
                        input_spikes = spikes_tr_out[:, : self.st]
                        exc_spikes = spikes_tr_out[:, self.st : self.ex]
                        inh_spikes = spikes_tr_out[:, self.ex : self.ih]
                        total_input_spikes = np.sum(input_spikes)
                        total_exc_spikes = np.sum(exc_spikes)
                        total_inh_spikes = np.sum(inh_spikes)

                        input_spike_rate = (
                            total_input_spikes / input_spikes.size
                            if input_spikes.size > 0
                            else 0
                        )
                        exc_spike_rate = (
                            total_exc_spikes / exc_spikes.size
                            if exc_spikes.size > 0
                            else 0
                        )
                        inh_spike_rate = (
                            total_inh_spikes / inh_spikes.size
                            if inh_spikes.size > 0
                            else 0
                        )
                        print(f"Network Activity Check:")
                        print(
                            f"  Input spikes: {total_input_spikes} ({input_spike_rate*100:.4f}% rate)"
                        )
                        print(
                            f"  Excitatory spikes: {total_exc_spikes} ({exc_spike_rate*100:.4f}% rate)"
                        )
                        print(
                            f"  Inhibitory spikes: {total_inh_spikes} ({inh_spike_rate*100:.4f}% rate)"
                        )

                        # get nonzero weights
                        nz_st_ws = self.weights[: self.st, self.st : self.ex][
                            self.weights[: self.st, self.st : self.ex] != 0
                        ]
                        nz_ex_ws = self.weights[self.st : self.ex, self.st : self.ih][
                            self.weights[self.st : self.ex, self.st : self.ih] != 0
                        ]
                        nz_ih_ws = self.weights[self.ex : self.ih, self.st : self.ex][
                            self.weights[self.ex : self.ih, self.st : self.ex] != 0
                        ]
                        print(f"Mean input weights: {np.mean(nz_st_ws)}")
                        print(f"Mean excitatory weights: {np.mean(nz_ex_ws)}")
                        print(f"Mean inhibitory weights: {np.mean(nz_ih_ws)}")

                        X_tr, y_tr = self.spikes_per_item(
                            spikes=exc_spikes,
                            labels=labels_tr_out,
                        )

                        if X_tr.size > 0:
                            # Check if we have enough samples for reliable PCA+LR
                            min_samples_needed = max(
                                10, self.N_classes * 5
                            )  # At least 5 samples per class
                            recommended_samples = (
                                self.N_classes * 20
                            )  # Recommended: 20 samples per class

                            print(
                                f"Training samples: {len(X_tr)} (min needed: {min_samples_needed}, recommended: {recommended_samples})"
                            )

                            if len(X_tr) < min_samples_needed:
                                print(
                                    f"⚠️  WARNING: Only {len(X_tr)} samples - PCA+LR may be unreliable (need ≥{min_samples_needed})"
                                )
                            elif len(X_tr) < recommended_samples:
                                print(
                                    f"ℹ️  NOTE: {len(X_tr)} samples available - consider using more for better PCA+LR estimation"
                                )

                            eval.fit(X=X_tr, Y=y_tr)
                            if e == 0 and PCA_plot:
                                from copy import deepcopy

                                scaler = deepcopy(eval.scaler)
                                pca = deepcopy(eval.pca)
                                pca_plotter = PCAScatterDisplay(scaler=scaler, pca=pca)
                            acc, phi = eval.score(X=X_tr, Y=y_tr)

                            print(f"Training Accuracy (PCA+LR): {acc:.4f}")
                            print(f"Training clustering (Phi): {phi:.4f}")

                            self._record_accuracy(
                                "train",
                                acc,
                                epoch=e + 1,
                                method="pca_lr",
                            )
                            self._record_phi(
                                "train",
                                phi,
                                epoch=e + 1,
                            )

                            # Show training data distribution
                            train_dist = np.bincount(y_tr, minlength=self.N_classes)
                            print(f"\nTraining Data Distribution:")
                            for i in range(self.N_classes):
                                count = train_dist[i]
                                percentage = (
                                    (count / len(y_tr)) * 100 if len(y_tr) > 0 else 0
                                )
                                print(
                                    f"  Class {i}: {count:3d} samples ({percentage:5.1f}%)"
                                )
                        else:
                            print("No training features available for PCA+LR")

                    print(f"{'-'*50}")

                    # Rinse memory
                # Clean up training data
                del labels_train
                del spikes_tr_out, labels_tr_out, spikes_train

                total_num_vals = (
                    self.all_val // self.batch_val
                )  # Use test dataset size for validation batching
                acc_LR_sum = 0.0
                phi_val_sum = 0.0

                # Predefine variables used in the test loop so cleanup never fails
                data_val = None
                labels_val = None
                mp_val = None
                spikes_val_out = None
                labels_val_out = None
                x_tar_se_val = np.zeros(self.N_x)
                x_tar_ee_val = np.zeros(self.N_exc)

                for val_batch_idx in range(total_num_vals):
                    # Image only streaming mode
                    from data.get_data import load_image_batch

                    val_start_idx = val_batch_idx * self.batch_val
                    data_val, labels_val = load_image_batch(
                        self.image_streamer,
                        val_start_idx,
                        self.batch_val,
                        self.num_steps,
                        int(np.sqrt(self.N_x)) ** 2,  # Image-only mode uses full N_x
                        partition="val",  # Use validation data
                    )
                    if data_val is None:
                        print(f"No more validation image data available")
                        break

                    # Update T_test for this batch
                    # Ensure test data width matches expected N_x
                    data_val = data_val[:, : self.st]
                    T_val_batch = data_val.shape[0]

                    mp_val = np.zeros((self.N - self.st))
                    mp_val[:] = self.resting_potential

                    spikes_val = np.zeros((T_val_batch, self.N), dtype=np.int8)
                    spikes_val[:, : self.st] = data_val
                    del data_val
                    gc.collect()
                    spike_trace_val = np.zeros(self.N - self.N_inh)
                    a_val = np.zeros(self.N_exc + self.N_inh)
                    I_syn_exc_val = np.zeros(self.N_exc)
                    I_syn_inh_val = np.zeros(self.N_inh)
                    spike_threshold_val = np.full(
                        self.N_exc + self.N_inh, fill_value=spike_threshold_default
                    )

                    # run validation of network
                    (
                        weights_val_out,  # delete after
                        spikes_val_out,  # delete after
                        mp_val_out,  # delete after
                        spike_threshold_val_out,  # delete after
                        labels_val_out,  # delete after
                        I_syn_exc_val_out,  # delete after
                        I_syn_inh_val_out,  # delete after
                        a_val_out,  # delete after
                        spike_trace_val_out,  # delete after
                        x_tar_se_val_out,  # delete after
                        x_tar_ee_val_out,  # delete after
                    ) = trainer.step(
                        weights=self.weights,
                        mp=mp_val,
                        spikes=spikes_val,
                        spike_labels=labels_val,
                        spike_trace=spike_trace_val,
                        training_mode="val",
                        spike_threshold=spike_threshold_val,
                        I_syn_exc=I_syn_exc_val,
                        I_syn_inh=I_syn_inh_val,
                        a=a_val,
                        x_tar_se=x_tar_se_val,
                        x_tar_ee=x_tar_ee_val,
                    )

                    # Prepare features from validation data for testing
                    X_val, y_val = self.spikes_per_item(
                        spikes=spikes_val_out[:, self.st : self.ex],
                        labels=labels_val_out,
                    )

                    # PCA+LR accuracy estimation
                    if accuracy_method == "pca_lr" and X_val.size > 0:
                        acc, phi = eval.score(X=X_val, Y=y_val)

                    # if PCA_plot
                    if PCA_plot and X_val.size > 0:
                        pca_plotter.plot(
                            X=X_val,
                            Y=y_val,
                            epoch=e,
                            run=self.ts_spec,
                            dataset=self.image_dataset,
                            phi=phi,
                        )

                    # update summed scores
                    if accuracy_method == "pca_lr":
                        acc_LR_sum += acc
                        phi_val_sum += phi

                # estimate the mean accuracy and phi
                if accuracy_method == "pca_lr":
                    final_acc_LR = acc_LR_sum / max(1, total_num_vals)
                    final_val_phi = phi_val_sum / max(1, total_num_vals)

                    # save weights if val improved
                    if final_acc_LR > best_val:
                        best_val = final_acc_LR
                        if save_model:
                            self.process(
                                save_model=True,
                                model_parameters=self.model_parameters,
                            )
                        new_best = "New best "
                    else:
                        new_best = ""

                    # print results
                    print(f"{new_best}Validation accuracy (PCA+LR): {final_acc_LR}")
                    print(f"Validation phi: {final_val_phi}")

                    # record the test accuracy
                    if final_acc_LR is not None:
                        self._record_accuracy(
                            "val",
                            final_acc_LR,
                            epoch=e + 1,
                            method="pca_lr",
                        )
                    if final_val_phi is not None:
                        self._record_phi("val", final_val_phi, epoch=e + 1)

                    # plot validation and training accuracy progress
                    plot_accuracy(mcc=False, pca=True, wta=False, phi=True)
                else:
                    final_acc_LR = None
                    final_val_phi = None

                # Rinse val memory
                if total_num_vals > 0:
                    del spikes_val, spikes_val_out, labels_val_out, labels_val
                    del (
                        weights_val_out,
                        mp_val_out,
                        spike_threshold_val_out,
                        I_syn_exc_val_out,
                        I_syn_inh_val_out,
                        a_val_out,
                        spike_trace_val_out,
                        x_tar_se_val_out,
                        x_tar_ee_val_out,
                    )
                    gc.collect()

                pbar.set_description(f"Epoch {e+1}/{self.batches}")
                # Handle None valuTruees safely
                acc_str = f"{final_acc_LR:.3f}" if final_acc_LR is not None else "N/A"
                phi_str = f"{final_val_phi:.2f}" if final_val_phi is not None else "N/A"
                pbar.set_postfix(acc=acc_str, phi=phi_str)
                pbar.update(1)
            pbar.close()

            # Clean up main training loop variables
            del (
                mp_train,
                I_syn_exc,
                I_syn_inh,
                a,
                spike_threshold,
                spike_trace,
                x_tar_se,
                x_tar_ee,
            )
            gc.collect()

        # create gif if wanted after finishing training THIS CAN BE DONE SEPARATELY WITHOUT INCLUDING IN THE CURRENT LOOP
        if gif_pca_plot and PCA_plot:
            from src.plot.spikes import GenerateGif

            output_filename = f"{self.ts_spec}.gif"
            gif = GenerateGif(
                frame_folder=pca_plotter.dir, output_filename=output_filename
            )
            gif.create()

        if gif_spikes_plot and heatmap_plot:
            from src.plot.spikes import GenerateGif

            output_filename = "evolution.gif"
            directory = os.path.join(
                "plots", "spikes", self.image_dataset, "all", self.ts, self.ts_spec
            )
            os.makedirs(directory, exist_ok=True)

            gif = GenerateGif(frame_folder=directory, output_filename=output_filename)

            gif.create()

        # Final test pass even if early-stopped (evaluate on test partition)
        from data.get_data import load_image_batch

        ################## THIS IS WHERE WE TEST THE MODEL ######################
        # Reset test pointers:
        self.image_streamer.reset_partition("test")

        # Reuse test-only evaluator logic with partition="test"
        total_num_tests = max(1, self.all_images_test // self.batch_image_test)
        T_test_batch = int(self.batch_image_test * self.num_steps)
        test_sample_count = 0
        acc_te_sum = 0.0
        phi_te_sum = 0.0

        for idx_test in range(total_num_tests):
            # make sure we are actually generating content from the test-set of MNIST
            test_start_idx = idx_test * self.batch_image_test
            data_test, labels_test = load_image_batch(
                self.image_streamer,
                test_start_idx,
                self.batch_image_test,
                self.num_steps,
                self.pixel_size,
                partition="test",
            )

            if data_test is None:
                raise ValueError("data_test is None")

            mp_test = np.zeros((self.ih - self.st))
            mp_test[:] = self.resting_potential
            spikes_test = np.zeros((T_test_batch, self.N), dtype=np.int8)
            spikes_test[:, : self.st] = data_test
            del data_test
            gc.collect()
            a_test = np.zeros(self.N_exc + self.N_inh)
            I_syn_exc_test = np.zeros(self.N_exc)
            I_syn_inh_test = np.zeros(self.N_inh)
            spike_threshold_test = np.full(
                self.N_exc + self.N_inh, fill_value=spike_threshold_default, dtype=float
            )
            spike_trace_te = np.zeros(self.N - self.N_inh)
            x_tar_se_test = np.zeros(self.N_x)
            x_tar_ee_test = np.zeros(self.N_exc)

            # Build common args locally in case outer scope wasn't initialized in this path
            (
                weights_te_out,
                spikes_te_out,
                mp_te_out,
                spike_threshold_te_out,
                labels_te_out,
                I_syn_exc_te_out,
                I_syn_inh_te_out,
                a_te_out,
                spike_trace_te_out,
                x_tar_se_te_out,
                x_tar_ee_te_out,
            ) = trainer.step(
                weights=self.weights,
                mp=mp_test,
                spikes=spikes_test,
                spike_labels=labels_test,
                spike_trace=spike_trace_te,
                training_mode="test",
                spike_threshold=spike_threshold_test,
                I_syn_exc=I_syn_exc_test,
                I_syn_inh=I_syn_inh_test,
                a=a_test,
                x_tar_se=x_tar_se_test,
                x_tar_ee=x_tar_ee_test,
            )

            # Increase sample index
            test_sample_count += self.batch_image_test

            # Prepare features from testing
            X_te, y_te = self.spikes_per_item(
                spikes=spikes_te_out[:, self.st : self.ex],
                labels=labels_te_out,
            )

            # PCA+LR accuracy estimation
            if accuracy_method == "pca_lr" and X_te.size > 0:
                acc, phi = eval.score(X=X_te, Y=y_te)
                # update summed scores
                acc_te_sum += acc
                phi_te_sum += phi

            # Clean up final test arrays and variables
            del spikes_test, spikes_te_out, labels_te_out, labels_test
            del (
                weights_te_out,
                mp_te_out,
                spike_threshold_te_out,
                I_syn_exc_te_out,
                I_syn_inh_te_out,
                a_te_out,
                spike_trace_te_out,
                x_tar_se_te_out,
                x_tar_ee_te_out,
            )
            gc.collect()

        # estimate the mean accuracy and phi
        if accuracy_method == "pca_lr":
            final_acc_LR = acc_te_sum / max(1, total_num_tests)
            final_phi = phi_te_sum / max(1, total_num_tests)

            # print results
            print(f"Testing accuracy (PCA+LR): {final_acc_LR}")
            print(f"Testing phi: {final_phi}")

            if final_acc_LR is not None:
                self._record_accuracy(
                    "test",
                    final_acc_LR,
                    epoch=e + 1,
                    method="pca_lr",
                )
            if final_phi is not None:
                self._record_phi("test", final_phi, epoch=e + 1)
