import numpy as np
import os
import gc
import torch
from tqdm import tqdm
import json
from datetime import datetime
import hashlib


from .train import train_network
from .layers import create_weights, create_arrays

# Use absolute imports from src when crossing package boundaries
# (relative imports with ... can fail when src is the top-level package in module execution)
from src.datasets.load import (
    DataStreamer,
)
from src.plot.live import (
    spike_plot,
    plot_epoch_training,
    weight_trajectories,
    weight_evolution,
    top_responders_plotted,
)
from src.evaluation.classifiers import (
    pca_logistic_regression,
    fit_model,
    accuracy,
    pca_quadratic_discriminant,
    Phi,
)


class snn_sleepy:
    def __init__(
        self,
        N_exc=200,
        N_inh=50,
        N_x=225,
        seed=1,
        which_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    ):
        self.N_exc = N_exc
        self.N_inh = N_inh
        self.N_x = N_x
        self.N_classes = len(which_classes)
        self.which_classes = which_classes
        self.st = N_x  # stimulation
        self.ex = self.st + N_exc  # excitatory
        self.ih = self.ex + N_inh  # inhibitory
        self.N = N_exc + N_inh + N_x
        # One-time plotting guard
        self._image_preview_done = False
        # Accuracy tracking
        self.model_loaded = False
        self.acc_history = {"train": [], "val": [], "test": []}
        self._acc_log_dir = os.path.join("results", "accuracy")
        self._training_run_id = None  # Will be set when training starts
        self._acc_log_file_cache = {}  # Cache log file paths per epoch
        # Initiate seed
        self.seed = seed
        self.rng_numpy = np.random.Generator(np.random.PCG64(seed))
        self.rng_torch = torch.Generator()
        self.rng_torch.manual_seed(seed)

    def _ensure_acc_logger(self, epoch: int | None = None):
        # Initialize training run ID on first call if not set
        if self._training_run_id is None:
            self._training_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Use cache key for this epoch
        epoch_key = f"epoch_{epoch}" if epoch is not None else "epoch_unknown"

        # Return cached file path if it exists
        if epoch_key in self._acc_log_file_cache:
            self._acc_log_file = self._acc_log_file_cache[epoch_key]
            return

        # Fetch dataset name
        ds = getattr(self, "image_dataset", "unknown")

        # Create directory structure: {dataset}/{run_id}/epoch_{epoch}/
        dataset_dir = os.path.join(self._acc_log_dir, ds)
        run_dir = os.path.join(dataset_dir, self._training_run_id)
        epoch_dir = os.path.join(run_dir, epoch_key)

        # Create directories if they don't exist
        os.makedirs(epoch_dir, exist_ok=True)

        # Create one log file per epoch (batches will append to this file)
        batch_ts = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )  # Timestamp for this batch/epoch
        log_file = os.path.join(epoch_dir, f"{batch_ts}_acc.jsonl")

        # Cache the file path for this epoch
        self._acc_log_file_cache[epoch_key] = log_file
        self._acc_log_file = log_file

    def _record_accuracy(self, split: str, value: float, epoch: int | None = None):
        try:
            if value is not None:
                self.acc_history.setdefault(split, []).append(value)
        except Exception:
            pass
        # Persist incrementally
        try:
            self._ensure_acc_logger(epoch=epoch)
            rec = {
                "timestamp": datetime.now().isoformat(),
                "split": str(split),
                "epoch": int(epoch) if epoch is not None else None,
                "accuracy": value,
            }
            with open(self._acc_log_file, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:
            raise ValueError(f"Failed to persist accuracy record ({e})")
        finally:
            gc.collect()

    def _hash_parameters(self, params: dict) -> str:
        """Generate a hash from model parameters dictionary."""
        # Sort keys and convert to JSON string for consistent hashing
        params_str = json.dumps(params, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(params_str.encode("utf-8")).hexdigest()[
            :12
        ]  # Use first 12 chars

    def process(
        self,
        load_model: bool = False,
        save_model: bool = False,
    ):
        ########## load or save model ##########
        if save_model and load_model:
            raise ValueError("load and save model cannot both be True")

        # get dataset
        dataset = getattr(self, "image_dataset", "unknown")

        # create model base directory
        model_base_dir = f"models/SNN_sleepy/{dataset}"
        os.makedirs(model_base_dir, exist_ok=True)

        if save_model:
            if self.model_parameters is None:
                raise ValueError(
                    "model_parameters must be provided when save_model=True"
                )

            # Generate hash from parameters
            param_hash = self._hash_parameters(self.model_parameters)

            # Create timestamp-based folder name: YYYYMMDD_HHMMSS_hash
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"{timestamp}_{param_hash}"
            model_dir = os.path.join(model_base_dir, folder_name)

            # Create model directory
            os.makedirs(model_dir, exist_ok=True)

            # Save model parameters
            with open(os.path.join(model_dir, "model_parameters.json"), "w") as outfile:
                json.dump(self.model_parameters, outfile, indent=2)

            # Save weights
            self._save_model_dir(model_dir)

            return model_dir

        if load_model:
            if not os.path.exists(model_base_dir):
                return
            folders = os.listdir(model_base_dir)

            if self.model_parameters is not None:
                # Search for exact parameter match using hash
                param_hash = self._hash_parameters(self.model_parameters)
                matched_folder = None

                for folder in folders:
                    # Check if folder name contains the hash
                    if param_hash in folder:
                        json_file_path = os.path.join(
                            model_base_dir, folder, "model_parameters.json"
                        )
                        if os.path.exists(json_file_path):
                            # Verify exact match
                            with open(json_file_path, "r") as j:
                                ex_params = json.load(j)
                            if ex_params == self.model_parameters:
                                matched_folder = folder
                                break

                if matched_folder is not None:
                    model_dir = os.path.join(model_base_dir, matched_folder)
                    self._load_model_dir(model_dir)
                    print("\rmodel loaded", end="")
                    self.model_loaded = True
                    return model_dir
                else:
                    self._log(
                        "No model found with matching parameters. Will train new model from scratch."
                    )
                    return None
            else:
                # No parameters provided - cannot safely load without knowing which model to use
                self._log(
                    "No model parameters provided. Cannot load model without parameters."
                )
                return None

    def _save_model_dir(self, model_dir):
        """Save essentials only: weights as compressed .npz file."""
        try:
            weights = getattr(self, "weights", None)
            if weights is not None:
                np.savez_compressed(
                    os.path.join(model_dir, "weights.npz"), weights=weights
                )
        except Exception as e:
            print(f"Warning: model save failed ({e})")

    def _load_model_dir(self, model_dir):
        """Load essentials only from model directory."""
        weights_path = os.path.join(model_dir, "weights.npz")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        data = np.load(weights_path)
        self.weights = data["weights"]

    def save_checkpoint(self, epoch, batch=None, suffix="", weights=None, dataset=None):
        """
        Save model weights as a checkpoint during training.

        Args:
            epoch: Current epoch number
            batch: Current batch number (optional)
            suffix: Additional suffix for checkpoint name (optional)
            weights: Weights to save (if None, uses self.weights)
            dataset: Dataset name (if None, uses self.image_dataset)

        Returns:
            Path to saved checkpoint file, or None if save failed
        """

        # Create checkpoint directory
        checkpoint_base_dir = f"models/SNN_sleepy/{dataset}/checkpoints"
        os.makedirs(checkpoint_base_dir, exist_ok=True)

        # Create checkpoint filename
        if batch is not None:
            checkpoint_name = f"checkpoint_epoch_{epoch:04d}_batch_{batch:04d}"
        else:
            checkpoint_name = f"checkpoint_epoch_{epoch:04d}"

        if suffix:
            checkpoint_name += f"_{suffix}"

        checkpoint_path = os.path.join(checkpoint_base_dir, f"{checkpoint_name}.npz")

        # Save weights
        np.savez_compressed(checkpoint_path, weights=weights)

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        """
        Load model weights from a checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint .npz file

        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            data = np.load(checkpoint_path)
            self.weights = data["weights"]
            return True
        except Exception as e:
            print(f"Warning: Checkpoint load failed ({e})")
            return False

    # prepare data
    def prepare_data(
        self,
        force_recreate=False,
        noisy_data=False,
        noise_level=0.005,
        add_breaks=False,
        total_data=10000,
        break_lengths=[500, 1500, 1000],
        gain=1.0,
        num_steps=100,
        offset=0,
        first_spike_time=0,
        time_var_input=False,
        val_split=0.1,
        train_split=0.7,
        test_split=0.2,
        batch_size=100,
        preview_data=False,
        dataset="mnist",
        geom_jitter=False,
        geom_jitter_amount=0.05,
        geom_noise_var=0.02,
        geom_workers=None,
        tri_size=0.5,
        tri_thick=2,
        cir_size=0.5,
        cir_thick=2,
        sqr_size=0.5,
        sqr_thick=2,
        x_size=0.5,
        x_thick=2,
        clamp_min=0.0,
        clamp_max=1.0,
    ):
        # Save important data generation parameters for reproducibility
        self.data_parameters = {
            # Dataset configuration
            "image_dataset": dataset,
            "total_data": total_data,
            "train_split": train_split,
            "val_split": val_split,
            "test_split": test_split,
            "batch_size": batch_size,
            # Spike conversion parameters
            "num_steps": num_steps,
            "gain": gain,
            "offset": offset,
            "first_spike_time": first_spike_time,
            "time_var_input": time_var_input,
            # Data augmentation/noise
            "noisy_data": noisy_data,
            "noise_level": noise_level,
            "add_breaks": add_breaks,
            "break_lengths": break_lengths,
            # Geomfig-specific parameters
            "geom_jitter": geom_jitter,
            "geom_jitter_amount": geom_jitter_amount,
            "geom_noise_var": geom_noise_var,
            "geom_workers": geom_workers,
            # Geometric shape parameters (for geomfig)
            "tri_size": tri_size,
            "tri_thick": tri_thick,
            "cir_size": cir_size,
            "cir_thick": cir_thick,
            "sqr_size": sqr_size,
            "sqr_thick": sqr_thick,
            "x_size": x_size,
            "x_thick": x_thick,
            "clamp_min": clamp_min,
            "clamp_max": clamp_max,
        }

        # image parameters
        self.dataset = dataset
        self.pixel_size = int(np.sqrt(self.N_x))
        self.num_steps = num_steps
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.num_data = total_data
        self.num_train = int(self.num_data * self.train_split)
        self.num_val = int(self.num_data * self.val_split)
        self.num_test = int(self.num_data * self.test_split)
        self.T_train = int(self.num_train * self.num_steps)
        self.T_val = int(self.num_val * self.num_steps)
        self.T_test = int(self.num_test * self.num_steps)
        self.batch_test = int(self.num_test // batch_size)
        self.batch_val = int(self.num_val // batch_size)
        self.batch_train = int(self.num_train // batch_size)
        self.epochs_train = int(self.num_train // batch_size) + 1
        self.epochs_val = int(self.num_val // batch_size) + 1
        self.epochs_test = int(self.num_test // batch_size) + 1

        # spike parameters
        self.num_steps = num_steps
        self.gain = gain
        self.offset = offset
        self.first_spike_time = first_spike_time
        self.time_var_input = time_var_input
        self.add_breaks = add_breaks
        self.break_lengths = break_lengths
        self.noise_level = noise_level
        # Geomfig-specific knobs (optional)
        self.geom_jitter = geom_jitter
        self.geom_jitter_amount = geom_jitter_amount
        self.geom_noise_var = geom_noise_var
        self.geom_workers = geom_workers
        # Initialize label attributes
        self.labels_train = None
        self.labels_test = None
        self.data_dir = f"data/datasets/{dataset}"

        # Instantiate DataStreamer to eagerly load and preprocess data
        # Use torch generator for MNIST-family datasets, numpy for geomfig
        rng_to_use = (
            self.rng_torch
            if dataset.lower() in ["mnist", "kmnist", "fmnist", "notmnist"]
            else self.rng_numpy
        )

        self.data_streamer = DataStreamer(
            data_dir=self.data_dir,
            pixel_size=self.pixel_size,
            num_steps=num_steps,
            num_classes=self.N_classes,
            which_classes=self.which_classes,
            num_samples=total_data,
            gain=gain,
            offset=offset,
            first_spike_time=first_spike_time,
            time_var_input=time_var_input,
            train_ratio=train_split,
            val_ratio=val_split,
            test_ratio=test_split,
            rng=rng_to_use,
            geom_noise_var=geom_noise_var,
            jitter=geom_jitter,
            jitter_amount=geom_jitter_amount,
            force_recreate=force_recreate,
            num_workers=geom_workers,
            noise_var=noise_level,
            dataset=dataset,
            tri_size=tri_size,
            tri_thick=tri_thick,
            cir_size=cir_size,
            cir_thick=cir_thick,
            sqr_size=sqr_size,
            sqr_thick=sqr_thick,
            x_size=x_size,
            x_thick=x_thick,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        )
        # preview data if True
        if preview_data:
            if dataset.lower() == "geomfig" and not self._image_preview_done:
                # Use preview_loaded_data for geomfig
                from src.plot.live import preview_loaded_data

                preview_loaded_data(
                    save_path="plots/data/geomfig_preview.png",
                    images=self.data_streamer.train_images,
                    labels=self.data_streamer.train_labels,
                )
                self._image_preview_done = True

            elif (
                dataset.lower() in ["mnist", "kmnist", "fmnist", "notmnist"]
                and not self._image_preview_done
            ):
                # Use plot_floats_and_spikes for MNIST-family datasets
                from src.plot.live import plot_floats_and_spikes

                # Get images and labels directly from streamer
                preview_images = self.data_streamer.train_images
                preview_img_labels = self.data_streamer.train_labels
                preview_spikes_all = self.data_streamer.train_spikes

                # Select one sample per unique label for preview (first 4 classes only)
                unique_labels = np.unique(preview_img_labels)
                num_preview = min(len(unique_labels), 4)  # Limit to first 4 classes

                selected_indices = []
                selected_labels = []
                for label in unique_labels[:num_preview]:
                    # Find first occurrence of this label
                    label_idx = np.where(preview_img_labels == label)[0][0]
                    selected_indices.append(label_idx)
                    selected_labels.append(int(label))

                if len(selected_indices) > 0:
                    # Convert images to numpy if they're torch tensors
                    if hasattr(preview_images, "numpy"):
                        preview_images_np = preview_images[selected_indices].numpy()
                    elif hasattr(preview_images, "cpu"):
                        preview_images_np = (
                            preview_images[selected_indices].cpu().numpy()
                        )
                    else:
                        preview_images_np = np.array(preview_images[selected_indices])

                    # Remove channel dimension if present: (N, 1, H, W) -> (N, H, W)
                    if preview_images_np.ndim == 4 and preview_images_np.shape[1] == 1:
                        preview_images_np = preview_images_np.squeeze(1)

                    # Extract spikes for selected samples directly from train_spikes
                    # Each sample has num_steps rows in the spike array
                    selected_spikes_list = []
                    selected_spike_labels_list = []
                    for img_idx in selected_indices:
                        start_row = int(img_idx) * num_steps
                        end_row = start_row + num_steps
                        selected_spikes_list.append(
                            preview_spikes_all[start_row:end_row]
                        )
                        # Repeat label for num_steps
                        selected_spike_labels_list.extend(
                            [preview_img_labels[img_idx]] * num_steps
                        )

                    # Stack spikes: shape (num_samples * num_steps, features)
                    selected_spikes = np.vstack(selected_spikes_list)
                    selected_spike_labels = np.array(selected_spike_labels_list)

                    plot_floats_and_spikes(
                        images=preview_images_np,
                        spikes=selected_spikes,
                        spike_labels=selected_spike_labels,
                        img_labels=np.array(selected_labels),
                        num_steps=num_steps,
                    )

    def prepare_network(
        self,
        resting_membrane=-70,
        retur=False,
        w_dense_ee=0.15,
        w_dense_se=0.1,
        w_dense_ei=0.2,
        w_dense_ie=0.25,
        se_weights=0.15,
        ee_weights=0.3,
        ei_weights=0.3,
        ie_weights=-0.3,
        spike_threshold_default=-55,
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
        self.spike_threshold_default = spike_threshold_default

        self.weights = create_weights(
            N_exc=self.N_exc,
            N_inh=self.N_inh,
            N=self.N,
            rng=self.rng_numpy,
            N_x=self.N_x,
            w_dense_ee=w_dense_ee,
            w_dense_ei=w_dense_ei,
            w_dense_se=w_dense_se,
            w_dense_ie=w_dense_ie,
            se_weights=se_weights,
            ee_weights=ee_weights,
            ei_weights=ei_weights,
            ie_weights=ie_weights,
        )

        if create_network:
            # create other arrays
            (
                self.mp_train,
                self.spikes_train,
                self.I_syn,
                self.spike_times,
                self.a,
                self.spike_threshold,
            ) = create_arrays(
                N=self.N,
                st=self.st,
                ih=self.ih,
                spike_threshold_default=self.spike_threshold_default,
                resting_membrane=self.resting_potential,
                runtime=self.T_train,
                data=None,
            )
            # return results if retur == True
            if retur:
                return (
                    self.weights,
                    self.spikes_train,
                    self.mp_train,
                    self.resting_potential,
                )
        if retur:
            return (self.weights,)

    def create_learning_bounds(self, weights, ex, st, ih, beta):
        # Calculate initial weight sums for normalization (used in train_network)
        sum_weights_exc = np.sum(np.abs(weights[:ex, st:ih])) * beta
        sum_weights_inh = np.sum(np.abs(weights[ex:ih, st:ex])) * beta
        sum_weights_total = np.sum(np.abs(weights)) * beta

        return sum_weights_exc, sum_weights_inh, sum_weights_total

    def train_network(
        self,
        plot_spikes_train=False,
        plot_spikes_test=False,
        plot_mp_train=False,
        plot_mp_test=False,
        plot_threshold=False,
        plot_traces_=False,
        train_weights=True,
        learning_rate_exc=0.0008,  # default learning rate for excitatory neurons
        learning_rate_inh=0.0008,  # default learning rate for inhibitory neurons
        w_target_exc=0.2,
        w_target_inh=-0.2,  # default target weight for inhibitory neurons
        var_noise=2.0,  # default noise variance
        min_mp=-100,  # default minimum membrane potential
        sleep=False,  # default sleep mode is off
        sleep_ratio=0.0,  # Sleep percentage per interval (e.g., 0.1 = 10%)
        normalize_weights=False,  # Alternative to sleep: maintain initial weight sum
        force_train=False,  # default force train is off
        save_checkpoints=False,  # Save intermediate checkpoints during training
        checkpoint_frequency="epoch",  # "epoch", "batch", or int (save every N batches)
        keep_checkpoints=5,  # Number of recent checkpoints to keep
        weight_decay=False,
        weight_decay_rate_exc=0.99997,
        weight_decay_rate_inh=0.99997,
        noisy_potential=True,
        noisy_threshold=False,
        noisy_weights=False,
        spike_adaption=True,
        delta_adaption=3,
        tau_adaption=100,
        beta=1.0,  # default beta for weight normalization
        A_plus=1.0,
        A_minus=1.0,
        tau_LTD=10,  # default tau for long-term depression
        tau_LTP=10,  # default tau for long-term potentiation
        early_stopping=True,
        early_stopping_patience_pct=0.2,  # Patience as percentage of total epochs (0.1 = 10%)
        dt=1,
        tau_m=30,  # default tau for membrane potential
        membrane_resistance=30,  # default membrane resistance
        reset_potential=-80,  # default reset potential
        spike_slope=-0.1,  # default spike slope
        spike_intercept=-4,  # default spike intercept
        pca_variance=0.95,  # default PCA variance
        start_time_spike_plot=None,  # default start time for spike plot
        stop_time_spike_plot=None,  # default stop time for spike plot
        start_index_mp=None,  # default start index for membrane potential plot
        stop_index_mp=None,  # default stop index for membrane potential plot
        time_start_mp=None,  # default start time for membrane potential plot
        time_stop_mp=None,  # default stop time for membrane potential plot
        mean_noise=0,  # default mean noise
        max_mp=40,  # default maximum membrane potential
        sleep_synchronized=False,  # default sleep synchronization is on
        weight_mean_noise=0.05,  # default mean noise for weights
        weight_var_noise=0.005,  # default variance noise for weights
        track_weights=False,  # Enable weight tracking (separate from plotting) - set to True to track weight changes
        plot_epoch_performance=True,  # default plot epoch performance is on
        plot_weight_trajectories=False,  # Plot weights after each epoch (for debugging)
        plot_weight_evolution=False,  # plot evolution of weights across epochs (mean, min/max)
        weight_track_samples=32,  # default weight track samples
        weight_track_interval=0,  # default weight track interval
        weight_track_sleep_interval=0,  # default weight track sleep interval
        narrow_top=0.2,  # Increased from 0.05 to 0.2 (20% of neurons)
        wide_top=0.15,
        tau_syn=30,  # default tau for synaptic time constant
        smoothening=350,  # default smoothening
        plot_top_response_train=False,
        plot_top_response_test=False,
        plot_tsne_during_training=True,  # New parameter for t-SNE plotting
        tsne_plot_interval=1,  # Plot t-SNE every N epochs (1 = every epoch)
        use_validation_data=False,
        accuracy_method="pca_lr",
        test_only=False,
        test_batch_size=None,
        patience=None,  # Can override percentage with explicit epoch count
        use_QDA=False,
        use_LR=True,
        sleep_max_iters=5000,
        on_timeout="give_up",
        sleep_tol_frac=1e-3,
        sleep_mode="static",
        epochs=10,
    ):
        self.dt = dt
        self.pca_variance = pca_variance
        self.use_validation_data = use_validation_data
        self.use_QDA = use_QDA
        self.use_LR = use_LR
        self.sleep_ratio = sleep_ratio
        self.normalize_weights = normalize_weights
        self.learning_rate_exc = learning_rate_exc
        self.learning_rate_inh = learning_rate_inh
        self.w_target_exc = w_target_exc
        self.w_target_inh = w_target_inh
        self.var_noise = var_noise
        self.min_mp = min_mp
        self.sleep = sleep
        self.sleep_ratio = sleep_ratio
        self.weight_decay = weight_decay
        self.weight_decay_rate_exc = weight_decay_rate_exc
        self.weight_decay_rate_inh = weight_decay_rate_inh
        self.noisy_potential = noisy_potential
        self.noisy_threshold = noisy_threshold
        self.noisy_weights = noisy_weights
        self.spike_adaption = spike_adaption
        self.delta_adaption = delta_adaption
        self.tau_adaption = tau_adaption
        self.save_checkpoints = save_checkpoints
        self.checkpoint_frequency = checkpoint_frequency
        self.keep_checkpoints = keep_checkpoints
        self.beta = beta
        self.epochs = epochs
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_LTD = tau_LTD
        self.tau_LTP = tau_LTP
        self.early_stopping = early_stopping
        self.early_stopping_patience_pct = early_stopping_patience_pct
        self.tau_m = tau_m
        self.membrane_resistance = membrane_resistance
        self.reset_potential = reset_potential
        self.spike_slope = spike_slope
        self.spike_intercept = spike_intercept
        self.pca_variance = pca_variance
        self.start_time_spike_plot = start_time_spike_plot
        self.stop_time_spike_plot = stop_time_spike_plot
        self.start_index_mp = start_index_mp
        self.stop_index_mp = stop_index_mp
        self.time_start_mp = time_start_mp
        self.time_stop_mp = time_stop_mp
        self.mean_noise = mean_noise
        self.max_mp = max_mp
        self.sleep_synchronized = sleep_synchronized
        self.weight_mean_noise = weight_mean_noise
        self.weight_var_noise = weight_var_noise
        self.plot_epoch_performance = plot_epoch_performance
        self.plot_weight_evolution = plot_weight_evolution
        self.plot_weight_trajectory = plot_weight_trajectories
        self.plot_spikes_train = plot_spikes_train
        self.plot_spikes_test = plot_spikes_test
        self.track_weights = (
            track_weights  # Separate from plotting - enables weight tracking
        )
        self.weight_track_samples = weight_track_samples
        self.weight_track_interval = weight_track_interval
        self.weight_track_sleep_interval = weight_track_sleep_interval
        self.narrow_top = narrow_top
        self.wide_top = wide_top
        self.tau_syn = tau_syn
        self.smoothening = smoothening
        self.plot_top_response_train = plot_top_response_train
        self.plot_top_response_test = plot_top_response_test
        self.plot_tsne_during_training = plot_tsne_during_training
        self.tsne_plot_interval = tsne_plot_interval
        self.accuracy_method = accuracy_method
        self.test_only = test_only
        self.test_batch_size = test_batch_size
        self.patience = patience
        self.sleep_max_iters = sleep_max_iters
        self.on_timeout = on_timeout
        self.sleep_tol_frac = sleep_tol_frac
        self.sleep_mode = sleep_mode
        self.force_train = force_train

        def initiate_trackers(self, mode, T):
            if mode == "train":
                # create learning bounds
                self.baseline_sum_exc, self.baseline_sum_inh, self.baseline_sum = (
                    self.create_learning_bounds(
                        self.weights, self.ex, self.st, self.ih, self.beta
                    )
                )
                # prepare_early_stopping needs these parameters from train_network scope
                self.prepare_early_stopping(
                    test_only=test_only,
                    train_weights=train_weights,
                    early_stopping=early_stopping,
                    early_stopping_patience_pct=early_stopping_patience_pct,
                    patience=patience,
                )

            # prepare common arguments
            self.prepare_common_args()

            # Total timesteps across all epochs
            self.total_timesteps = self.epochs * T

            # Generate sleep schedule for training (random timesteps where sleep activates)
            if mode == "train" and self.sleep and self.sleep_ratio > 0.0:
                # Calculate total sleep timesteps needed
                sleep_timesteps_total = int(self.sleep_ratio * self.total_timesteps)

                # Generate random timesteps across entire training period (no early learning boundary)
                # Use seed for reproducibility
                rng = np.random.RandomState(self.seed)
                # Generate random timesteps from 0 to total_timesteps-1
                sleep_schedule = rng.choice(
                    self.total_timesteps,
                    size=min(sleep_timesteps_total, self.total_timesteps),
                    replace=False,  # No duplicates
                )
                sleep_schedule = np.sort(sleep_schedule)  # Sort for efficient lookup

                # Store as set for O(1) lookup
                self.sleep_schedule = set(sleep_schedule.tolist())
                self.sleep_timesteps_total = sleep_timesteps_total
                self.sleep_timesteps_used = 0

                # Calculate sleep iterations budget: 2% of total timesteps = computational budget for sleep
                # This represents the total number of sleep iterations we can afford across all training
                self.sleep_iterations_budget = int(
                    self.sleep_ratio * self.total_timesteps
                )
                self.sleep_iterations_used = (
                    0  # Track actual sleep iterations used (not just triggers)
                )

                print(
                    f"Generated sleep schedule: {sleep_timesteps_total} sleep triggers out of {self.total_timesteps} total timesteps ({self.sleep_ratio*100:.1f}%)"
                )
                print(
                    f"Sleep iterations budget: {self.sleep_iterations_budget} iterations ({self.sleep_ratio*100:.1f}% of total computational time)"
                )
            else:
                self.sleep_schedule = set()
                self.sleep_timesteps_total = 0
                self.sleep_timesteps_used = 0
                self.sleep_iterations_budget = 0
                self.sleep_iterations_used = 0

            # Initialize global timestep counter (tracks position across all batches/epochs)
            self.global_timestep = 0

            # Progress bar updates every 1000 timesteps
            pbar_total = self.total_timesteps // 1000
            pbar_desc = f"{mode} Epoch 0/{self.epochs}:"

            # prepare progress bar
            self.pbar = tqdm(
                total=pbar_total,
                desc=pbar_desc,
                unit="1k steps",
                ncols=80,
                bar_format="{desc} [{bar}] ETA: {remaining} |{postfix}",
            )

            # Track sleep percentages across epochs
            self.sleep_percent_sum = 0.0
            self.sleep_percent_count = 0

            # Track weight changes during sleep across epochs
            self.all_weight_tracking_sleep = {
                "exc_mean": [],
                "exc_min": [],
                "exc_max": [],
                "exc_samples": [],
                "inh_mean": [],
                "inh_min": [],
                "inh_max": [],
                "inh_samples": [],
                "times": [],
                "sleep_segments": [],
            }
            self._tracking_time_offset = 0.0  # what does this actually do?

            # Track weight evolution over epochs for plotting
            self.weight_evolution = {
                "epochs": [],
                "exc_mean": [],
                "exc_min": [],
                "exc_max": [],
                "inh_mean": [],
                "inh_min": [],
                "inh_max": [],
            }

            # prepare performance tracker
            self.performance_tracker = {
                "train_accuracy": [0],
                "train_clustering": [0],
                "val_accuracy": [0],
                "val_clustering": [0],
                "test_accuracy": [0],
                "test_clustering": [0],
            }

        def reset_persistent_arrays(self):
            """Reset persistent state arrays between epochs.

            NOTE: We don't create full T_train-sized arrays anymore - they're too large!
            Instead, we only store final state values to carry over between batches.
            Batch-sized arrays are created on-demand in process_batch().
            """
            # Store only final state values (scalars/small arrays), not full T_train arrays
            # This avoids creating huge arrays (e.g., 600k timesteps × 250 neurons = 600 MB)
            # Initialize mp_final_state to resting potential for first batch of epoch
            self.mp_final_state = np.full(
                shape=(self.ih - self.st),
                fill_value=self.resting_potential,
                dtype=float,
            )

            # Create small persistent arrays for state (not full T_train size)
            # These are 1D arrays for single neuron states, not time-series
            self.I_syn = np.zeros(self.N - self.st)  # Current synaptic current state
            self.spike_times = np.zeros(self.N)  # Current spike times state
            self.a = np.zeros(self.N - self.st)  # Current adaptation state
            self.spike_threshold = np.full(
                shape=(self.ih - self.st),
                fill_value=self.spike_threshold_default,
                dtype=float,
            )

        def process_batch(self, data, labels, mode, epoch, global_timestep_offset=0):
            """
            Process a single batch of data through the network.

            Args:
                data: Input data array (T_batch, N_x)
                labels: Labels array (T_batch,)
                mode: "train", "val", or "test"
                global_timestep_offset: Global timestep offset for this batch (for sleep schedule lookup)

            Returns:
                Dictionary with:
                    - spikes: Output spikes
                    - labels: Labels (same as input)
                    - weight_tracking_sleep: Weight tracking data if available
            """
            # Get batch size
            T_batch = data.shape[0]

            # For validation/test: reset arrays at start of each batch
            # For training: arrays persist between batches

            mp, spikes, I_syn, spike_times, a, spike_threshold = create_arrays(
                N=self.N,
                st=self.st,
                ih=self.ih,
                spike_threshold_default=self.spike_threshold_default,
                resting_membrane=self.resting_potential,
                runtime=T_batch,
                data=data,
            )
            if mode != "train":
                # Training: create batch-sized arrays (don't use huge persistent arrays)
                # The old approach created T_train-sized arrays (e.g., 600k timesteps = 600 MB)
                # This is much more memory efficient and faster
                mp, spikes, I_syn, spike_times, a, spike_threshold = create_arrays(
                    N=self.N,
                    st=self.st,
                    ih=self.ih,
                    spike_threshold_default=self.spike_threshold_default,
                    resting_membrane=self.resting_potential,
                    runtime=T_batch,
                    data=data,
                )
                # Initialize from persistent state if available (carry over from previous batch)
                # CRITICAL: All state variables must carry over for network state continuity
                # Membrane potential: initialize first timestep with final state from previous batch
                mp[0] = self.mp_final_state
                # Synaptic current: 1D array (N - st)
                I_syn[:] = self.I_syn
                # Spike times: 1D array (N) - tracks time since last spike
                spike_times[:] = self.spike_times
                # Adaptation variable: 1D array (N - st)
                a[:] = self.a
                # Spike threshold: 1D array (ih - st) - dynamic threshold per neuron
                spike_threshold[:] = self.spike_threshold

            # Weights: always use trained weights (persist from training to val/test)
            weights_input = self.weights

            # track weights if enabled (separate from plotting)
            # Enable tracking if: track_weights is explicitly True, or if we need to plot trajectories/evolution
            self.track_weights = self.track_weights or (
                mode == "train"
                and (self.plot_weights_trajectory or self.plot_weight_evolution)
            )

            # Pass sleep schedule info to train_network (only for training mode)
            sleep_schedule = (
                getattr(self, "sleep_schedule", set()) if mode == "train" else set()
            )
            sleep_timesteps_total = (
                getattr(self, "sleep_timesteps_total", 0) if mode == "train" else 0
            )
            sleep_timesteps_used_ref = (
                [getattr(self, "sleep_timesteps_used", 0)] if mode == "train" else [0]
            )  # Use list for mutable reference
            # Pass sleep iterations budget (computational time budget)
            sleep_iterations_budget = (
                getattr(self, "sleep_iterations_budget", 0) if mode == "train" else 0
            )
            sleep_iterations_used_ref = (
                [getattr(self, "sleep_iterations_used", 0)] if mode == "train" else [0]
            )  # Use list for mutable reference

            # Call train_network function (imported from .train)
            (
                weights_out,
                spikes_out,
                mp_out,
                spike_threshold_out,
                spike_labels_out,
                sleep_percent,
                I_syn_out,
                spike_times_out,
                a_out,
                weight_tracking_sleep,
            ) = train_network(
                weights=weights_input,
                spike_labels=labels,
                mp=mp,
                T=T_batch,
                spikes=spikes,
                spike_times=spike_times,
                spike_threshold=spike_threshold,
                a=a,
                weight_tracking_sleep=self.all_weight_tracking_sleep,
                I_syn=I_syn,
                train_weights=True if mode == "train" else False,
                track_weights=self.track_weights,
                global_timestep_offset=global_timestep_offset,
                sleep_schedule=sleep_schedule,
                sleep_timesteps_total=sleep_timesteps_total,
                sleep_timesteps_used=sleep_timesteps_used_ref,
                sleep_iterations_budget=sleep_iterations_budget,
                sleep_iterations_used=sleep_iterations_used_ref,
                show_progress=True,  # Suppress per-batch progress bars (use epoch-level instead)
                **self.common_args,
            )

            # Update weights if training
            if mode == "train":
                self.weights = weights_out
                # CRITICAL: Persist final state for next batch to maintain network continuity
                # Membrane potential: store final timestep state (last row of mp_out)
                self.mp_final_state = mp_out[
                    -1
                ].copy()  # Final membrane potential state
                # Other state arrays (1D, current state)
                self.I_syn = I_syn_out.copy() if I_syn_out is not None else self.I_syn
                self.spike_times = (
                    spike_times_out.copy()
                    if spike_times_out is not None
                    else self.spike_times
                )
                self.a = a_out.copy() if a_out is not None else self.a
                self.spike_threshold = (
                    spike_threshold_out.copy()
                    if spike_threshold_out is not None
                    else self.spike_threshold
                )
                # Update sleep timesteps and iterations used counters
                self.sleep_timesteps_used = sleep_timesteps_used_ref[0]
                self.sleep_iterations_used = sleep_iterations_used_ref[0]

                if self.plot_weights_trajectory:
                    weight_trajectories(
                        weight_tracking_epoch=weight_tracking_sleep,
                        epoch=epoch,
                        dataset=self.dataset,
                    )

                if self.plot_spikes_train:
                    spike_plot(spikes_out, spike_labels_out)

                print(spikes_out.shape)
                print(spike_labels_out.shape)

            return {
                "spikes": spikes_out,
                "labels": spike_labels_out,
                "weight_tracking_sleep": weight_tracking_sleep,
            }

        def update_trackers(
            weight_tracking_epoch, all_weight_tracking_sleep, _tracking_time_offset
        ):
            # Accumulate weight tracking data (sleep only)
            if weight_tracking_epoch is not None:
                for key in [
                    "exc_mean",
                    "exc_min",
                    "exc_max",
                    "exc_samples",
                    "inh_mean",
                    "inh_min",
                    "inh_max",
                    "inh_samples",
                ]:
                    if key in weight_tracking_epoch:
                        all_weight_tracking_sleep[key].extend(
                            weight_tracking_epoch[key]
                        )
                _wt_times = weight_tracking_epoch.get("times", [])
                if len(_wt_times) > 0:
                    # Use numpy for memory-efficient processing of large arrays
                    try:
                        _wt_times_array = np.array(_wt_times, dtype=float)
                        _wt_times_offset = _wt_times_array + _tracking_time_offset
                        all_weight_tracking_sleep["times"].extend(
                            _wt_times_offset.tolist()
                        )
                    except MemoryError:
                        # If still too large, process in chunks
                        chunk_size = 100000  # Process 100k at a time
                        for i in range(0, len(_wt_times), chunk_size):
                            chunk = _wt_times[i : i + chunk_size]
                            all_weight_tracking_sleep["times"].extend(
                                [float(t) + _tracking_time_offset for t in chunk]
                            )
                for seg in weight_tracking_epoch.get("sleep_segments", []):
                    try:
                        s, te = (
                            float(seg[0]) + _tracking_time_offset,
                            float(seg[1]) + _tracking_time_offset,
                        )
                        all_weight_tracking_sleep["sleep_segments"].append((s, te))
                    except Exception:
                        pass
                if len(_wt_times) > 0:
                    _tracking_time_offset += float(max(_wt_times)) + 1.0

            return all_weight_tracking_sleep

        def estimate_accuracy(self, mode, accuracy_method, spikes, labels):
            # Support both "MLR" and "pca_lr" as aliases for PCA logistic regression
            if accuracy_method in ["MLR", "pca_lr"]:
                if mode == "train":
                    self.MLR_model = pca_logistic_regression(
                        variance_ratio=self.pca_variance, whiten=True, standardize=True
                    )
                    fit_model(self.MLR_model, spikes, labels)
                return accuracy(self.MLR_model, spikes, labels)
            elif accuracy_method == "QDA":
                if mode == "train":
                    self.QDA_model = pca_quadratic_discriminant(
                        variance_ratio=self.pca_variance, whiten=True, standardize=True
                    )
                    fit_model(self.QDA_model, spikes, labels)
                return accuracy(self.QDA_model, spikes, labels)
            else:
                raise ValueError(
                    f"Invalid accuracy method: {accuracy_method}. Supported: 'MLR', 'pca_lr', 'QDA'"
                )

        def estimate_clustering(self, mode, spikes, labels, num_steps):
            if mode == "train":
                self.Phi_model = Phi().fit(
                    spikes,
                    labels,
                    num_steps,
                    self.N_classes,
                    pca_variance=self.pca_variance,
                    random_state=self.seed,
                )
            return self.Phi_model.score(
                spikes, labels, num_steps=num_steps, require_any=True
            )

        def run_epoch(
            self,
            mode: str,  # "train" | "val" | "test"
            epoch: int,
            batch_size: int,
            collect_for_metric: bool = True,
            max_batches: int | None = None,  # optional for cheap eval
            plot_weights_trajectory: bool = False,
            plot_weights_evolution: bool = False,
        ):
            # make plotting arguments global
            self.plot_weights_trajectory = plot_weights_trajectory

            # Reset partition pointer to start of data for this epoch
            self.data_streamer.reset_partition(mode)

            # Reset persistent arrays at the start of each epoch (even for training)
            # Arrays will persist between batches within the epoch, but reset between epochs
            if mode == "train":
                reset_persistent_arrays(self)
                # Reset batch offset for slicing persistent arrays
                self._batch_offset = 0
                # Reset global flags for per-epoch printing (suppress repeated messages)
                try:
                    from . import train as train_module

                    train_module._train_network_numba_reported = False
                except (ImportError, AttributeError):
                    pass

            all_spikes = []
            all_labels = []
            n_batches = 0

            # Estimate total batches for progress bar
            if mode == "train":
                total_samples = self.num_train
            elif mode == "val":
                total_samples = self.num_val
            else:  # test
                total_samples = self.num_test

            total_batches = (
                total_samples + batch_size - 1
            ) // batch_size  # Ceiling division
            if max_batches is not None:
                total_batches = min(total_batches, max_batches)

            # Create epoch-level progress bar (only for training to avoid clutter)
            if mode == "train":
                # Get latest accuracies (or 0.0 if not yet computed)
                train_acc_display = (
                    self.performance_tracker["train_accuracy"][-1]
                    if len(self.performance_tracker["train_accuracy"]) > 0
                    else 0.0
                )
                val_acc_display = (
                    self.performance_tracker["val_accuracy"][-1]
                    if len(self.performance_tracker["val_accuracy"]) > 0
                    else 0.0
                )
                epoch_pbar = tqdm(
                    total=total_batches,
                    desc=f"{mode} Epoch {self._current_epoch+1}/{self.epochs}",
                    unit="batch",
                    leave=False,  # Don't leave progress bar after completion (cleaner output)
                    ncols=100,  # Fixed width to prevent line wrapping
                    postfix={
                        "train_acc": f"{train_acc_display:.4f}",
                        "val_acc": f"{val_acc_display:.4f}",
                    },
                )
            else:
                test_acc_display = (
                    self.performance_tracker["test_accuracy"][-1]
                    if len(self.performance_tracker["test_accuracy"]) > 0
                    else 0.0
                )
                epoch_pbar = tqdm(
                    total=total_batches,
                    desc=f"{mode} Epoch {self._current_epoch+1}/{self.epochs}",
                    unit="batch",
                    leave=False,  # Don't leave progress bar after completion (cleaner output)
                    ncols=100,  # Fixed width to prevent line wrapping
                    postfix={"test_acc": f"{test_acc_display:.4f}"},
                )

            # Loop through all batches in the partition
            # get_batch() advances an internal pointer and returns (None, None) when exhausted
            while True:
                data, labels = self.data_streamer.get_batch(batch_size, partition=mode)
                if data is None:
                    # No more data available - we've processed all batches
                    break

                # Calculate global timestep offset for this batch (for sleep schedule lookup)
                if mode == "train":
                    batch_start_timestep = self.global_timestep
                else:
                    batch_start_timestep = 0  # Not used for val/test

                out = process_batch(
                    self,
                    data,
                    labels,
                    mode=mode,
                    epoch=epoch,
                    global_timestep_offset=batch_start_timestep,
                )

                # Update global timestep counter after processing batch (only for training)
                if mode == "train":
                    T_batch = data.shape[0]
                    self.global_timestep += T_batch

                # Update epoch-level progress bar
                if epoch_pbar is not None:
                    # Compute mean weights for monitoring (only during training)
                    if (
                        mode == "train"
                        and hasattr(self, "weights")
                        and self.weights is not None
                    ):
                        try:
                            # Compute mean absolute values of excitatory and inhibitory weights
                            mean_exc = float(
                                np.mean(
                                    np.abs(self.weights[: self.ex, self.st : self.ih])
                                )
                            )
                            mean_inh = float(
                                np.mean(
                                    np.abs(
                                        self.weights[
                                            self.ex : self.ih, self.st : self.ex
                                        ]
                                    )
                                )
                            )
                            # Update progress bar with mean weights
                            train_acc_display = (
                                self.performance_tracker["train_accuracy"][-1]
                                if len(self.performance_tracker["train_accuracy"]) > 0
                                else 0.0
                            )
                            val_acc_display = (
                                self.performance_tracker["val_accuracy"][-1]
                                if len(self.performance_tracker["val_accuracy"]) > 0
                                else 0.0
                            )
                            epoch_pbar.set_postfix(
                                {
                                    "train_acc": f"{train_acc_display:.4f}",
                                    "val_acc": f"{val_acc_display:.4f}",
                                    "m_exc": f"{mean_exc:.3f}",
                                    "m_inh": f"{mean_inh:.3f}",
                                }
                            )
                        except Exception:
                            # If computation fails, just update without weights
                            pass
                    epoch_pbar.update(1)

                if self.track_weights:
                    # Limit weight tracking data size to prevent unbounded growth
                    # Keep only last 1M entries per list to prevent slowdown
                    max_tracking_size = 1000000
                    if (
                        len(self.all_weight_tracking_sleep.get("times", []))
                        > max_tracking_size
                    ):
                        # Clear old data, keep only recent entries
                        for key in self.all_weight_tracking_sleep:
                            if isinstance(self.all_weight_tracking_sleep[key], list):
                                self.all_weight_tracking_sleep[key] = (
                                    self.all_weight_tracking_sleep[key][
                                        -max_tracking_size // 2 :
                                    ]
                                )

                    update_trackers(
                        out["weight_tracking_sleep"],
                        self.all_weight_tracking_sleep,
                        self._tracking_time_offset,
                    )

                if collect_for_metric:
                    all_spikes.append(out["spikes"])
                    all_labels.append(out["labels"])

                n_batches += 1

                # Save checkpoint after each batch if enabled (only during training)
                if mode == "train" and self.save_checkpoints:
                    self.save_checkpoint(epoch=self._current_epoch + 1, batch=n_batches)

                if max_batches is not None and n_batches >= max_batches:
                    break

            if not collect_for_metric or len(all_spikes) == 0:
                # Close epoch-level progress bar
                if epoch_pbar is not None:
                    epoch_pbar.close()
                return 0.0, 0.0  # Return default accuracy and clustering when no data

            # Optimize concatenation: pre-allocate array if possible, otherwise use efficient concatenation
            if len(all_spikes) > 0:
                # Get shape from first array to pre-allocate
                first_shape = all_spikes[0].shape
                total_timesteps = sum(arr.shape[0] for arr in all_spikes)
                spikes_cat = np.empty(
                    (total_timesteps, first_shape[1] - self.N_x),
                    dtype=all_spikes[0].dtype,
                )
                labels_cat = np.empty(total_timesteps, dtype=all_labels[0].dtype)

                # Fill pre-allocated arrays
                idx = 0
                for spikes_arr, labels_arr in zip(all_spikes, all_labels):
                    end_idx = idx + spikes_arr.shape[0]
                    spikes_cat[idx:end_idx] = spikes_arr[:, self.N_x :]
                    labels_cat[idx:end_idx] = labels_arr
                    idx = end_idx
            else:
                spikes_cat = np.concatenate(all_spikes, axis=0)
                labels_cat = np.concatenate(all_labels, axis=0)

            accuracy = estimate_accuracy(
                self, mode, self.accuracy_method, spikes_cat, labels_cat
            )
            clustering = estimate_clustering(
                self, mode, spikes_cat, labels_cat, self.num_steps
            )

            # Update progress bar postfix with computed accuracy before closing
            if epoch_pbar is not None:
                if mode == "train":
                    # Update with current epoch's training accuracy (validation will be shown in next epoch)
                    val_acc_display = (
                        self.performance_tracker["val_accuracy"][-1]
                        if len(self.performance_tracker["val_accuracy"]) > 0
                        else 0.0
                    )
                    epoch_pbar.set_postfix(
                        {
                            "train_acc": f"{accuracy:.4f}",
                            "val_acc": f"{val_acc_display:.4f}",
                        }
                    )
                    # Print accuracy so it's visible even when progress bar closes
                    print(
                        f"Epoch {epoch+1}/{self.epochs}: train_acc={accuracy:.4f}, val_acc={val_acc_display:.4f}"
                    )
                elif mode == "test":
                    epoch_pbar.set_postfix({"test_acc": f"{accuracy:.4f}"})
                    print(f"Epoch {epoch+1}/{self.epochs}: test_acc={accuracy:.4f}")
                elif mode == "val":
                    # Validation accuracy is printed from train() method, but print here too for consistency
                    pass
                epoch_pbar.refresh()  # Force refresh to show updated postfix
                epoch_pbar.close()

            # Update weight evolution trackers at end of training epoch
            if mode == "train" and plot_weights_evolution:
                update_epoch_trackers(epoch)
                # Plot weight evolution
                weight_evolution(self.weight_evolution)

            return float(accuracy), float(clustering)

        def train(self, batch_size: int, val_batch_size: int | None = None):
            # Call nested function directly (both are in train_network scope)
            initiate_trackers(self, "train", self.T_train)

            for epoch in range(self.epochs):
                # Store current epoch for batch-level checkpoint saving
                self._current_epoch = epoch
                # ---- TRAIN EPOCH ----
                train_acc, train_clust = run_epoch(
                    self,
                    mode="train",
                    epoch=epoch,
                    batch_size=batch_size,
                    collect_for_metric=True,
                    plot_weights_trajectory=self.plot_weight_trajectory,
                    plot_weights_evolution=self.plot_weight_evolution,
                )

                # ---- VALIDATION ----
                val_acc, val_clust = None, None
                if self.use_validation_data:
                    val_acc, val_clust = run_epoch(
                        self,
                        mode="val",
                        epoch=epoch,
                        batch_size=val_batch_size,
                        collect_for_metric=True,
                        max_batches=None,  # or a small number if you want cheap/fast mid-training val
                    )

                    # Print validation accuracy immediately after it's computed
                    print(
                        f"Epoch {epoch+1}/{self.epochs}: val_acc={val_acc:.4f}, val_clust={val_clust:.4f}"
                    )

                    # early stopping update
                    if self.early_stopping:
                        self.update_early_stopping(val_acc)
                        if self.should_stop:
                            break

                # update performance tracker (validation accuracy is now available for next epoch's progress bar)
                self.performance_tracker["train_accuracy"].append(train_acc)
                self.performance_tracker["train_clustering"].append(train_clust)
                if val_acc is not None:
                    self.performance_tracker["val_accuracy"].append(val_acc)
                    self.performance_tracker["val_clustering"].append(val_clust)

            # ----- TESTING ----
            test_acc, test_clust = run_epoch(
                self,
                mode="test",
                epoch=epoch,
                batch_size=self.test_batch_size,
                collect_for_metric=True,
                max_batches=None,
            )
            # update performance tracker
            self.performance_tracker["test_accuracy"].append(test_acc)
            self.performance_tracker["test_clustering"].append(test_clust)

            # make post-run plots
            post_plot()

        def update_epoch_trackers(epoch):
            # Extract weight matrices
            W_exc = self.weights[: self.ex, self.st : self.ih]
            W_inh = self.weights[self.ex : self.ih, self.st : self.ex]

            # Compute stats on non-zero weights only
            W_exc_nz = W_exc[W_exc != 0]
            if W_exc_nz.size > 0:
                self.weight_evolution["exc_mean"].append(float(np.mean(W_exc_nz)))
                self.weight_evolution["exc_min"].append(float(np.min(W_exc_nz)))
                self.weight_evolution["exc_max"].append(float(np.max(W_exc_nz)))
            else:
                self.weight_evolution["exc_mean"].append(0.0)
                self.weight_evolution["exc_min"].append(0.0)
                self.weight_evolution["exc_max"].append(0.0)

            W_inh_nz = W_inh[W_inh != 0]
            if W_inh_nz.size > 0:
                self.weight_evolution["inh_mean"].append(float(np.mean(W_inh_nz)))
                self.weight_evolution["inh_min"].append(float(np.min(W_inh_nz)))
                self.weight_evolution["inh_max"].append(float(np.max(W_inh_nz)))
            else:
                self.weight_evolution["inh_mean"].append(0.0)
                self.weight_evolution["inh_min"].append(0.0)
                self.weight_evolution["inh_max"].append(0.0)

            # Track epoch number (epoch is 0-indexed, store as 1-indexed for plotting)
            self.weight_evolution["epochs"].append(epoch + 1)

        def post_plot():
            if self.plot_top_response_test:
                top_responders_plotted(
                    spikes=self.spikes_test[50 * self.num_steps :],
                    labels=self.labels_test[50 * self.num_steps :],
                    ih=self.ih,
                    st=self.st,
                    num_classes=self.N_classes,
                    narrow_top=narrow_top,
                    smoothening=smoothening,
                    train=False,
                    wide_top=wide_top,
                )

            if self.plot_top_response_train:
                top_responders_plotted(
                    spikes=self.spikes_train[50 * self.num_steps :],
                    labels=self.labels_train[50 * self.num_steps :],
                    ih=self.ih,
                    st=self.st,
                    num_classes=self.N_classes,
                    narrow_top=narrow_top,
                    smoothening=smoothening,
                    train=True,
                    wide_top=wide_top,
                )

            if self.plot_epoch_performance:
                # performance_tracker columns: [phi_te, acc_te] and optional val: [phi_val, acc_val]
                # Only plot if performance_tracker exists (it's initialized in prepare_early_stopping via initiate_trackers)
                if (
                    hasattr(self, "performance_tracker")
                    and self.performance_tracker is not None
                ):
                    if (
                        hasattr(self, "val_performance_tracker")
                        and self.val_performance_tracker is not None
                    ):
                        plot_epoch_training(
                            self.performance_tracker[:, 1],
                            self.performance_tracker[:, 0],
                            val_acc=self.val_performance_tracker[:, 1],
                            val_phi=self.val_performance_tracker[:, 0],
                        )
                else:
                    plot_epoch_training(
                        self.performance_tracker[:, 1], self.performance_tracker[:, 0]
                    )

        # Expose the nested train function so it can be called after train_network()
        # Bind it to self so it can be called as a method
        import types

        self.train = types.MethodType(train, self)

    def save_model_parameters(self):
        # Save model parameters explicitly (only important training parameters, not plotting flags)
        self.model_parameters = {
            # Training configuration
            "seed": self.seed,
            "early_stopping": self.early_stopping,
            "early_stopping_patience_pct": self.early_stopping_patience_pct,
            "weight_decay_rate_exc": self.weight_decay_rate_exc,
            "weight_decay_rate_inh": self.weight_decay_rate_inh,
            # Learning rates and targets
            "learning_rate_exc": self.learning_rate_exc,
            "learning_rate_inh": self.learning_rate_inh,
            "w_target_exc": self.w_target_exc,
            "w_target_inh": self.w_target_inh,
            # STDP parameters
            "A_plus": self.A_plus,
            "A_minus": self.A_minus,
            "tau_LTP": self.tau_LTP,
            "tau_LTD": self.tau_LTD,
            # Sleep configuration
            "sleep": self.sleep,
            "sleep_ratio": self.sleep_ratio,
            "sleep_synchronized": self.sleep_synchronized,
            "sleep_max_iters": self.sleep_max_iters,
            "on_timeout": self.on_timeout,
            "sleep_tol_frac": self.sleep_tol_frac,
            "sleep_mode": self.sleep_mode,
            "normalize_weights": self.normalize_weights,
            # Neuron dynamics
            "dt": self.dt,
            "tau_m": self.tau_m,
            "tau_syn": self.tau_syn,
            "membrane_resistance": self.membrane_resistance,
            "resting_potential": self.resting_potential,
            "reset_potential": self.reset_potential,
            "min_mp": self.min_mp,
            "max_mp": self.max_mp,
            # Spike parameters
            "spike_threshold_default": self.spike_threshold_default,
            "spike_slope": self.spike_slope,
            "spike_intercept": self.spike_intercept,
            "spike_adaption": self.spike_adaption,
            "delta_adaption": self.delta_adaption,
            "tau_adaption": self.tau_adaption,
            # Noise parameters
            "noisy_potential": self.noisy_potential,
            "noisy_threshold": self.noisy_threshold,
            "noisy_weights": self.noisy_weights,
            "mean_noise": self.mean_noise,
            "var_noise": self.var_noise,
            "weight_mean_noise": self.weight_mean_noise,
            "weight_var_noise": self.weight_var_noise,
            # Weight normalization
            "beta": self.beta,
            # Evaluation
            "pca_variance": self.pca_variance,
            "accuracy_method": self.accuracy_method,
            "use_validation_data": self.use_validation_data,
            "use_QDA": self.use_QDA,
            "use_LR": self.use_LR,
            "narrow_top": self.narrow_top,
            "wide_top": self.wide_top,
            "smoothening": self.smoothening,
            # Data configuration (computed values)
            "all_train": self.T_train,
            "all_test": self.num_test,
            "all_val": self.num_val,
            "batch_train": self.batch_train,
            "batch_test": self.batch_test,
            "batch_val": self.batch_val,
            "epochs": self.epochs,
            # Network architecture
            "w_dense_ee": self.w_dense_ee,
            "w_dense_ei": self.w_dense_ei,
            "w_dense_se": self.w_dense_se,
            "w_dense_ie": self.w_dense_ie,
            "ie_weights": self.ie_weights,
            "ee_weights": self.ee_weights,
            "ei_weights": self.ei_weights,
            "se_weights": self.se_weights,
            "classes": self.which_classes,
            "N_exc": self.N_exc,
            "N_inh": self.N_inh,
            "N_x": self.N_x,
        }

    def prepare_early_stopping(
        self,
        test_only,
        train_weights,
        early_stopping,
        early_stopping_patience_pct,
        patience,
    ):
        # pre-define performance tracking array
        if test_only and not train_weights:
            # single pass; tracker will be (1,2) later
            self.performance_tracker = np.zeros((1, 2))
        else:
            self.performance_tracker = np.zeros((self.epochs, 2))

        # early stopping setup
        if early_stopping:
            self.patience_epochs = (
                max(1, int(early_stopping_patience_pct * self.epochs))
                if patience is None
                else patience
            )
            self.best_val_metric = -np.inf
            self.epochs_without_improvement = 0
            self.best_weights = None
            print(
                f"Early stopping enabled: patience = {self.patience_epochs} epochs ({early_stopping_patience_pct*100:.0f}% of {self.epochs} total)"
            )
        else:
            self.patience_epochs = None
            self.best_val_metric = None
            self.epochs_without_improvement = None
            self.best_weights = None

        # Initialize should_stop flag (always initialize, even if early stopping is disabled)
        self.should_stop = False

    def update_early_stopping(self, val_metric):
        """
        Update early stopping state based on validation metric.

        Args:
            val_metric: Current validation metric (higher is better, e.g., accuracy)

        Returns:
            bool: True if training should stop, False otherwise
        """
        if not self.early_stopping or self.patience_epochs is None:
            self.should_stop = False
            return False

        # Check if current metric is better than best
        if val_metric > self.best_val_metric:
            # Improvement found
            self.best_val_metric = val_metric
            self.epochs_without_improvement = 0
            # Save current weights as best
            self.best_weights = self.weights.copy()
        else:
            # No improvement
            self.epochs_without_improvement += 1

        # Check if we should stop
        self.should_stop = self.epochs_without_improvement >= self.patience_epochs

        if self.should_stop:
            print(
                f"\nEarly stopping triggered after {self.epochs_without_improvement} epochs without improvement"
            )
            print(f"Best validation metric: {self.best_val_metric:.4f}")
            # Restore best weights
            if self.best_weights is not None:
                self.weights = self.best_weights
                print("Restored best weights")

        return self.should_stop

    def prepare_common_args(self):
        # Bundle common training arguments
        # max_sum, max_sum_exc, max_sum_inh are used as thresholds in sleep_func
        # They should be the same as baseline_sum, baseline_sum_exc, baseline_sum_inh
        max_sum = self.baseline_sum
        max_sum_exc = self.baseline_sum_exc
        max_sum_inh = self.baseline_sum_inh

        # initial_sum_exc and initial_sum_inh are used for normalization
        # They are the initial sums before beta scaling (baseline = initial * beta)
        initial_sum_exc = None
        initial_sum_inh = None
        initial_sum_exc = self.baseline_sum_exc / self.beta
        initial_sum_inh = self.baseline_sum_inh / self.beta

        self.common_args = dict(
            tau_syn=self.tau_syn,
            resting_potential=self.resting_potential,
            membrane_resistance=self.membrane_resistance,
            N_exc=self.N_exc,
            N_inh=self.N_inh,
            baseline_sum=self.baseline_sum,
            baseline_sum_exc=self.baseline_sum_exc,
            baseline_sum_inh=self.baseline_sum_inh,
            max_sum=max_sum,
            max_sum_exc=max_sum_exc,
            max_sum_inh=max_sum_inh,
            beta=self.beta,
            sleep_synchronized=self.sleep_synchronized,
            weight_decay_rate_exc=self.weight_decay_rate_exc,
            weight_decay_rate_inh=self.weight_decay_rate_inh,
            learning_rate_exc=self.learning_rate_exc,
            learning_rate_inh=self.learning_rate_inh,
            w_target_exc=self.w_target_exc,
            w_target_inh=self.w_target_inh,
            tau_LTP=self.tau_LTP,
            tau_LTD=self.tau_LTD,
            tau_m=self.tau_m,
            sleep=self.sleep,
            max_mp=self.max_mp,
            min_mp=self.min_mp,
            dt=self.dt,
            N=self.N,
            A_plus=self.A_plus,
            A_minus=self.A_minus,
            spike_adaption=self.spike_adaption,
            delta_adaption=self.delta_adaption,
            tau_adaption=self.tau_adaption,
            spike_threshold_default=self.spike_threshold_default,
            spike_intercept=self.spike_intercept,
            spike_slope=self.spike_slope,
            noisy_threshold=self.noisy_threshold,
            reset_potential=self.reset_potential,
            noisy_potential=self.noisy_potential,
            noisy_weights=self.noisy_weights,
            weight_mean_noise=self.weight_mean_noise,
            weight_var_noise=self.weight_var_noise,
            N_x=self.N_x,
            normalize_weights=self.normalize_weights,
            sleep_max_iters=self.sleep_max_iters,
            on_timeout=self.on_timeout,
            sleep_tol_frac=self.sleep_tol_frac,
            sleep_mode=self.sleep_mode,
            weight_track_samples_exc=self.weight_track_samples,
            weight_track_samples_inh=self.weight_track_samples,
            sleep_snapshot_interval=self.weight_track_sleep_interval,
            train_record_every=(
                self.weight_track_interval if self.weight_track_interval > 0 else 1000
            ),  # Use weight_track_interval if set, otherwise default to 1000
            # Additional required parameters
            mean_noise=self.mean_noise,
            var_noise=self.var_noise,
            sleep_ratio=self.sleep_ratio,
            initial_sum_exc=initial_sum_exc,
            initial_sum_inh=initial_sum_inh,
            sleep_hard_pause=True,  # Default value from train_network signature
            sleep_epsilon=1e-8,  # Default value from train_network signature
        )
