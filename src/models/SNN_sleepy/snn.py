import numpy as np
import os
import gc
import torch
from tqdm import tqdm
import json
from datetime import datetime
import traceback
import hashlib


from .train import train_network
from .dynamics import create_learning_bounds
from .layers import create_weights, create_arrays
from src.datasets.load import (
    DataStreamer,
)
from src.plot.plot import (
    spike_plot,
    mp_plot,
    phi,
    plot_epoch_training,
    get_elite_nodes,
    plot_weight_evolution_during_sleep,
    plot_weight_evolution,
    plot_weight_trajectories_with_sleep_epoch,
    save_weight_distribution_gif,
)
from src.evaluation.classifiers import (
    t_SNE, # might be missing plotting function
    bin_spikes_by_label_no_breaks,
    pca_logistic_regression,
    fit_model,
    accuracy,
    pca_quadratic_discriminant,
    Phi,
    phi_score,
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
        batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S")  # Timestamp for this batch/epoch
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
        return hashlib.sha256(params_str.encode("utf-8")).hexdigest()[:12]  # Use first 12 chars

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
                raise ValueError("model_parameters must be provided when save_model=True")
            
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
                    self._log("No model found with matching parameters. Will train new model from scratch.")
                    return None
            else:
                # No parameters provided - cannot safely load without knowing which model to use
                self._log("No model parameters provided. Cannot load model without parameters.")
                return None

    def _save_model_dir(self, model_dir):
        """Save essentials only: weights as compressed .npz file."""
        try:
            weights = getattr(self, "weights", None)
            if weights is not None:
                np.savez_compressed(
                    os.path.join(model_dir, "weights.npz"),
                    weights=weights
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
        
        Returns:
            Path to saved checkpoint file, or None if save failed
        """
        try:
            if weights is None:
                return None
            
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
        except Exception as e:
            print(f"Warning: Checkpoint save failed ({e})")
            return None

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
        self.image_dataset = dataset
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
        self.noisy_data = noisy_data
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
        self.data_streamer = DataStreamer(
            data_dir=self.data_dir,
            pixel_size=self.pixel_size,
            num_steps=num_steps,
            num_classes=self.N_classes,
            which_classes=self.which_classes,
            gain=gain,
            offset=offset,
            first_spike_time=first_spike_time,
            time_var_input=time_var_input,
            train_ratio=train_split,
            val_ratio=val_split,
            test_ratio=test_split,
            rng=self.rng_numpy,
            noisy_data=noisy_data,
            noise_var=noise_level,
            geom_noise_var=geom_noise_var,
            jitter=geom_jitter,
            jitter_amount=geom_jitter_amount,
            force_recreate=force_recreate,
            num_workers=geom_workers,
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

    def prepare_network(
        self,
        plot_weights=False,
        plot_network=False,
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
            plot_network=plot_network,
        )

        if create_network:
            # create other arrays
            (
                self.mp_train,
                self.mp_test,
                self.spikes_train,
                self.spikes_test,
                self.I_syn,
                self.spike_times,
                self.a,
                self.spike_threshold,
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
                st=self.st,
                ih=self.ih,
                spike_threshold_default=self.spike_threshold_default,
            )
            # return results if retur == True
            if retur:
                return (
                    self.weights,
                    self.spikes_train,
                    self.spikes_test,
                    self.mp_train,
                    self.mp_test,
                    self.resting_potential,
                )
        if retur:
            return (
                self.weights,
            )

    def train_network(
        self,
        plot_spikes_train=False,
        plot_spikes_test=False,
        plot_mp_train=False,
        plot_mp_test=False,
        plot_weights=True,
        plot_threshold=False,
        plot_traces_=False,
        train_weights=True,
        learning_rate_exc=0.0005, # default learning rate for excitatory neurons
        learning_rate_inh=0.0005, # default learning rate for inhibitory neurons
        w_target_exc=0.2,
        w_target_inh=-0.2, # default target weight for inhibitory neurons
        var_noise=2.0, # default noise variance
        spike_threshold_default=-55, 
        min_mp=-100, # default minimum membrane potential
        sleep=False, # default sleep mode is off
        sleep_ratio=0.0,  # Sleep percentage per interval (e.g., 0.1 = 10%)
        normalize_weights=False,  # Alternative to sleep: maintain initial weight sum
        force_train=False, # default force train is off
        save_model=True,
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
        beta=1.0, # default beta for weight normalization
        A_plus=1.0,
        A_minus=1.0,
        tau_LTD=10, # default tau for long-term depression
        tau_LTP=10, # default tau for long-term potentiation
        early_stopping=True,
        early_stopping_patience_pct=0.2,  # Patience as percentage of total epochs (0.1 = 10%)
        dt=1,
        tau_m=30, # default tau for membrane potential
        membrane_resistance=30, # default membrane resistance
        reset_potential=-80, # default reset potential
        spike_slope=-0.1, # default spike slope
        spike_intercept=-4, # default spike intercept
        pca_variance=0.95, # default PCA variance
        start_time_spike_plot=None, # default start time for spike plot
        stop_time_spike_plot=None, # default stop time for spike plot
        start_index_mp=None, # default start index for membrane potential plot
        stop_index_mp=None, # default stop index for membrane potential plot
        time_start_mp=None, # default start time for membrane potential plot
        time_stop_mp=None, # default stop time for membrane potential plot
        mean_noise=0, # default mean noise
        max_mp=40, # default maximum membrane potential
        sleep_synchronized=True, # default sleep synchronization is on
        weight_mean_noise=0.05,  # default mean noise for weights
        weight_var_noise=0.005,  # default variance noise for weights
        plot_epoch_performance=True, # default plot epoch performance is on
        plot_weights_per_epoch=False,  # Plot weights after each epoch (for debugging)
        weight_track_samples=32, # default weight track samples
        weight_track_interval=0, # default weight track interval
        weight_track_sleep_interval=0, # default weight track sleep interval
        narrow_top=0.2,  # Increased from 0.05 to 0.2 (20% of neurons)
        wide_top=0.15,
        tau_syn=30, # default tau for synaptic time constant
        smoothening=350, # default smoothening
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
    ):
        self.dt = dt
        self.pca_variance = pca_variance
        self.use_validation_data = use_validation_data
        self.use_QDA = use_QDA
        self.use_LR = use_LR
        self.sleep_ratio = sleep_ratio
        self.normalize_weights = normalize_weights
        self.learning_rate_exc = learning_rate_exc
        self.learning_rate_inh=learning_rate_inh
        self.w_target_exc=w_target_exc
        self.w_target_inh=w_target_inh
        self.var_noise=var_noise
        self.spike_threshold_default=spike_threshold_default
        self.min_mp=min_mp
        self.sleep=sleep
        self.sleep_ratio=sleep_ratio
        self.weight_decay=weight_decay
        self.weight_decay_rate_exc=weight_decay_rate_exc
        self.weight_decay_rate_inh=weight_decay_rate_inh
        self.noisy_potential=noisy_potential
        self.noisy_threshold=noisy_threshold
        self.noisy_weights=noisy_weights
        self.spike_adaption=spike_adaption
        self.delta_adaption=delta_adaption
        self.tau_adaption=tau_adaption
        self.save_checkpoints=save_checkpoints
        self.checkpoint_frequency=checkpoint_frequency
        self.keep_checkpoints=keep_checkpoints
        self.beta=beta
        self.A_plus=A_plus
        self.A_minus=A_minus
        self.tau_LTD=tau_LTD
        self.tau_LTP=tau_LTP
        self.early_stopping=early_stopping
        self.early_stopping_patience_pct=early_stopping_patience_pct
        self.tau_m=tau_m
        self.membrane_resistance=membrane_resistance
        self.reset_potential=reset_potential
        self.spike_slope=spike_slope
        self.spike_intercept=spike_intercept
        self.pca_variance=pca_variance
        self.start_time_spike_plot=start_time_spike_plot
        self.stop_time_spike_plot=stop_time_spike_plot
        self.start_index_mp=start_index_mp
        self.stop_index_mp=stop_index_mp
        self.time_start_mp=time_start_mp
        self.time_stop_mp=time_stop_mp
        self.mean_noise=mean_noise
        self.max_mp=max_mp
        self.sleep_synchronized=sleep_synchronized
        self.weight_mean_noise=weight_mean_noise
        self.weight_var_noise=weight_var_noise
        self.plot_epoch_performance=plot_epoch_performance
        self.plot_weights_per_epoch=plot_weights_per_epoch
        self.weight_track_samples=weight_track_samples
        self.weight_track_interval=weight_track_interval
        self.weight_track_sleep_interval=weight_track_sleep_interval
        self.narrow_top=narrow_top
        self.wide_top=wide_top
        self.tau_syn=tau_syn
        self.smoothening=smoothening
        self.plot_top_response_train=plot_top_response_train
        self.plot_top_response_test=plot_top_response_test
        self.plot_tsne_during_training=plot_tsne_during_training
        self.tsne_plot_interval=tsne_plot_interval
        self.accuracy_method=accuracy_method        
        self.test_only=test_only
        self.test_batch_size=test_batch_size
        self.patience=patience
        self.sleep_max_iters=sleep_max_iters
        self.on_timeout=on_timeout
        self.sleep_tol_frac=sleep_tol_frac
        self.sleep_mode=sleep_mode

        def initiate_trackers(self, mode, epochs, T):
            if mode == "train":
                # create learning bounds
                self.baseline_sum_exc, self.baseline_sum_inh, self.baseline_sum = create_learning_bounds(self.weights, self.ex, self.st, self.ih, self.beta)
                # prepare early stopping
                self.prepare_early_stopping()

            # prepare common arguments
            self.prepare_common_args()

            # Total timesteps across all epochs
            self.total_timesteps = epochs * T
            # Progress bar updates every 1000 timesteps
            pbar_total = self.total_timesteps // 1000
            pbar_desc = f"{mode} Epoch 0/{epochs}:"

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
            self._tracking_time_offset = 0.0 # what does this actually do?

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

        def reset_arrays(self, T_batch, data):
            mp = np.zeros((T_batch, self.N - self.N_x))
            mp[0] = self.resting_potential
            spikes = np.zeros((T_batch, self.N), dtype=np.int8)
            spikes[:, : self.N_x] = data
            
            # Use instance variables for persistent state arrays (created in prepare_network)
            I_syn = self.I_syn.copy()
            spike_times = self.spike_times.copy()
            a = self.a.copy()
            spike_threshold = self.spike_threshold.copy()
            return mp, spikes, I_syn, spike_times, a, spike_threshold


        def update_trackers(self, weight_tracking_epoch, all_weight_tracking_sleep, _tracking_time_offset):
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
                    all_weight_tracking_sleep[key].extend(
                        weight_tracking_epoch[key]
                    )
                _wt_times = weight_tracking_epoch.get("times", [])
                all_weight_tracking_sleep["times"].extend(
                    [float(t) + _tracking_time_offset for t in _wt_times]
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

        def evaluate_performance(self, mode, accuracy_method, spikes, labels, variance_ratio, num_steps):
            if accuracy_method == "Phi":
                if mode == "train": 
                    self.Phi_model =Phi().fit(spikes, labels, num_steps, self.N_classes, pca_variance=variance_ratio, random_state=self.seed)
                return self.Phi_model.score(spikes, labels, num_steps=num_steps, require_any=True)

            elif accuracy_method == "MLR":
                if mode == "train":
                    self.MLR_model = pca_logistic_regression(variance_ratio=variance_ratio, whiten=True, standardize=True)
                    fit_model(self.MLR_model, spikes, labels)
                return accuracy(self.MLR_model, spikes, labels)
            elif accuracy_method == "QDA":
                if mode == "train":
                    self.QDA_model = pca_quadratic_discriminant(variance_ratio=variance_ratio, whiten=True, standardize=True)
                    fit_model(self.QDA_model, spikes, labels)
                return accuracy(self.QDA_model, spikes, labels)
            else:
                raise ValueError(f"Invalid accuracy method: {accuracy_method}")

        def run_epoch(
            self,
            mode: str,                 # "train" | "val" | "test"
            epochs: int = 1,           # usually 1 here
            batch_size: int,
            train_weights: bool,
            collect_for_metric: bool = True,
            max_batches: int | None = None,   # optional for cheap eval
        ):
            self.data_streamer.reset_partition(mode)

            all_spikes = []
            all_labels = []
            n_batches = 0

            while True:
                data, labels = self.data_streamer.get_batch(batch_size, partition=mode)
                if data is None:
                    break
                out = self.process_batch(data, labels, mode=mode, train_weights=train_weights)

                # trackers (only meaningful when training, but harmless)
                self.update_trackers(out["weight_tracking_sleep"], self.all_weight_tracking_sleep, self._tracking_time_offset)

                if collect_for_metric:
                    all_spikes.append(out["spikes"])
                    all_labels.append(out["labels"])

                n_batches += 1
                
                # Save checkpoint after each batch if enabled (only during training)
                if mode == "train" and self.save_checkpoints:
                    self.save_checkpoint(epoch=self._current_epoch+1, batch=n_batches)
                
                if max_batches is not None and n_batches >= max_batches:
                    break

            if not collect_for_metric or len(all_spikes) == 0:
                return {"n_batches": n_batches}

            spikes_cat = np.concatenate(all_spikes, axis=0)
            labels_cat = np.concatenate(all_labels, axis=0)

            # IMPORTANT: define num_steps for your metric. Often = your stimulus length, not batch size.
            # If each batch is already one stimulus, num_steps = T_batch; otherwise keep your real presentation length.
            num_steps = self.T_train if mode == "train" else (self.T_val if mode == "val" else self.T_test)

            score = self.evaluate_performance(
                "train" if mode == "train" else "eval",  # see note below
                self.accuracy_method,
                spikes_cat,
                labels_cat,
                self.pca_variance,
                num_steps,
            )

            return {
                "score": float(score),
                "n_batches": n_batches,
                "n_samples": int(labels_cat.shape[0]),
            }

        def train(self, *, epochs: int, batch_size: int, val_batch_size: int | None = None, val_every_n_batches: int | None = None):
            if val_batch_size is None:
                val_batch_size = batch_size

            self.initiate_trackers("train", epochs, self.T_train)
            
            # Initialize checkpoint tracking
            batch_counter = 0

            for epoch in range(epochs):
                # Store current epoch for batch-level checkpoint saving
                self._current_epoch = epoch
                # ---- TRAIN EPOCH ----
                # Option A: simplest: just run_epoch() for train
                train_res = self.run_epoch(
                    mode="train",
                    batch_size=batch_size,
                    train_weights=True,
                    collect_for_metric=True,
                )
                train_score = train_res.get("score", None)
                n_batches = train_res.get("n_batches", 0)
                batch_counter += n_batches

                # Save checkpoint if enabled
                if self.save_checkpoints:
                    should_save = True
                    if self.checkpoint_frequency != "epoch" and self.checkpoint_frequency != "batch":
                        should_save = (batch_counter % self.checkpoint_frequency == 0)
                    if should_save:
                        self.save_checkpoint(epoch=epoch+1, batch=None)

                # ---- VALIDATION ----
                if self.use_validation_data:
                    val_res = self.run_epoch(
                        mode="val",
                        batch_size=val_batch_size,
                        train_weights=False,
                        collect_for_metric=True,
                        max_batches=None,   # or a small number if you want cheap/fast mid-training val
                    )
                    val_score = val_res.get("score", None)

                    # early stopping update
                    if self.early_stopping:
                        self.update_early_stopping(val_score)
                        if self.should_stop:
                            break

                # logging/plots here
                self.log_epoch(epoch, train_score, val_score if self.use_validation_data else None)

            # ----- TESTING ----
            test_res = self.run_epoch(
                mode="test",
                batch_size=test_batch_size,
                train_weights=False,
                collect_for_metric=True,
                max_batches=None,
            )
            test_score = test_res.get("score", None)

        def save_model_(weights):
            self.process
            

        def plot():
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

            if plot_traces_:
                plot_traces(
                    N_exc=self.N_exc,
                    N_inh=self.N_inh,
                    pre_traces=self.pre_trace_plot,
                    post_traces=self.post_trace_plot,
                )

            if plot_top_response_test:
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


        if plot_epoch_performance:
            # performance_tracker columns: [phi_te, acc_te] and optional val: [phi_val, acc_val]
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



    def analyze_results(
        self,
        perplexity=8,
        max_iter=1000,
        n_components=2,
        t_sne_train=False,
        t_sne_test=False,
        pca_train=False,
        pca_test=False,
        calculate_phi_=True,
    ):
        if (
            t_sne_train
            and hasattr(self, "spikes_train")
            and self.spikes_train is not None
        ):
            t_SNE(
                spikes=self.spikes_train[:, self.st : self.ex],
                labels_spike=self.labels_train,
                n_components=n_components,
                perplexity=perplexity,
                max_iter=max_iter,
                random_state=self.seed,
                train=True,
                show_plot=False,
            )
        if t_sne_test:
            t_SNE(
                spikes=self.spikes_test[:, self.st : self.ex],
                labels_spike=self.labels_test,
                n_components=n_components,
                perplexity=perplexity,
                max_iter=max_iter,
                random_state=self.seed,
                train=False,
                show_plot=False,
            )
        if (
            pca_train
            and hasattr(self, "spikes_train")
            and self.spikes_train is not None
        ):
            PCA_analysis(
                spikes=self.spikes_train[:, self.N_x :],
                labels_spike=self.labels_train,
                n_components=n_components,
                random_state=self.seed,
            )
        if pca_test:
            PCA_analysis(
                spikes=self.spikes_test[:, self.N_x :],
                labels_spike=self.labels_test,
                n_components=n_components,
                random_state=self.seed,
            )

        test_phi = None
        if calculate_phi_:
            if hasattr(self, "spikes_train") and self.spikes_train is not None:
                phi_tr, phi_te, *_ = calculate_phi(
                    spikes_train=self.spikes_train,
                    spikes_test=self.spikes_test,
                    labels_train=self.labels_train,
                    labels_test=self.labels_test,
                    num_steps=self.num_steps,
                    pca_variance=self.pca_variance,
                    random_state=self.seed,
                    num_classes=self.N_classes,
                )
                test_phi = float(phi_tr) if phi_tr is not None else None
            else:
                print("Skipping phi calculation: no training data (test-only run).")
        # Optional: PCA+LR end-to-end analysis only if training data exists
        test_acc_dict = None
        if (
            hasattr(self, "spikes_train")
            and self.spikes_train is not None
            and hasattr(self, "labels_train")
            and self.labels_train is not None
        ):
            try:
                # Prepare features via binning
                X_tr, y_tr = bin_spikes_by_label_no_breaks(
                    spikes=self.spikes_train[:, self.st : self.ih],
                    labels=self.labels_train,
                )
                X_te, y_te = bin_spikes_by_label_no_breaks(
                    spikes=self.spikes_test[:, self.st : self.ih],
                    labels=self.labels_test,
                )
                if X_tr.size > 0 and X_te.size > 0:
                    # Simple split of training for val
                    rng = np.random.RandomState(42)
                    idx = rng.permutation(X_tr.shape[0])
                    split = max(1, int(0.8 * len(idx)))
                    tr_idx, va_idx = idx[:split], idx[split:]
                    if va_idx.size == 0:
                        va_idx = tr_idx[-1:]
                        tr_idx = tr_idx[:-1]
                    accs, _, _, _ = self._pca_eval(
                        X_train=X_tr[tr_idx],
                        y_train=y_tr[tr_idx],
                        X_val=X_tr[va_idx],
                        y_val=y_tr[va_idx],
                        X_test=X_te,
                        y_test=y_te,
                    )
                    print(f"PCA+LR accuracy: {accs}")
                    test_acc_dict = accs
                    try:
                        if isinstance(test_acc_dict, dict) and "test" in test_acc_dict:
                            self._record_accuracy("test", test_acc_dict.get("test"))
                    except Exception:
                        pass
            except Exception as ex:
                print(f"PCA+LR analysis skipped: {ex}")
        return test_acc_dict, test_phi

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

    def prepare_early_stopping(self, test_only, train_weights, early_stopping, early_stopping_patience_pct, patience):
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

    def prepare_common_args(self):
        # Bundle common training arguments
        self.common_args = dict(
            tau_syn=self.tau_syn,
            resting_potential=self.resting_potential,
            membrane_resistance=self.membrane_resistance,
            N_exc=self.N_exc,
            N_inh=self.N_inh,
            baseline_sum=self.baseline_sum,
            baseline_sum_exc=self.baseline_sum_exc,
            baseline_sum_inh=self.baseline_sum_inh,
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
            train_snapshot_interval=self.weight_track_interval,
            sleep_snapshot_interval=self.weight_track_sleep_interval,
        )

    def _pca_eval(self, X_train, y_train, X_val, y_val, X_test, y_test, reg_param):
        """Run PCA-based classifier with configured options, return (accs, scaler, pca, clf)."""
        try:
            if getattr(self, "use_QDA", False):
                return pca_quadratic_discriminant(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    y_test=y_test,
                    variance_ratio=self.pca_variance,
                    reg_param=reg_param, 
                )
            else:
                return pca_logistic_regression(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    y_test=y_test,
                    variance_ratio=self.pca_variance,
                )
        except Exception as ex:
            raise ValueError(f"PCA classification failed ({ex})")
