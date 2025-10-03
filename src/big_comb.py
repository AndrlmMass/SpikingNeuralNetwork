import numpy as np
import os
import gc
from tqdm import tqdm
import json
from train import train_network
from get_data import (
    create_data,
    load_audio_batch_streaming,
    load_multimodal_batch,
    load_prealigned_multimodal_batch,
    sync_multimodal_datasets,
    load_audio_batch,
    load_image_batch,
)
from plot import (
    spike_plot,
    heat_map,
    mp_plot,
    weights_plot,
    spike_threshold_plot,
    plot_traces,
    top_responders_plotted,
    plot_phi_acc,
    plot_epoch_training,
    plot_audio_spectrograms_and_spikes,
    plot_audio_spectrograms_and_spikes_simple,
    get_elite_nodes,
)
from analysis import (
    t_SNE,
    PCA_analysis,
    calculate_phi,
    pca_logistic_regression,
    bin_spikes_by_label_no_breaks,
)
from create_network import create_weights, create_arrays


class snn_sleepy:
    def __init__(
        self,
        N_exc=200,
        N_inh=50,
        N_x=225,
        classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    ):
        self.N_exc = N_exc
        self.N_inh = N_inh
        self.N_x = N_x
        self.N_classes = len(classes)
        self.classes = classes
        self.st = N_x  # stimulation
        self.ex = self.st + N_exc  # excitatory
        self.ih = self.ex + N_inh  # inhibitory
        self.N = N_exc + N_inh + N_x
        # One-time plotting guard
        self._did_plot_spectrograms = False

    def process(
        self,
        sleep_scores: list = None,
        model_dir_: str = None,
        load_data: bool = False,
        save_data: bool = False,
        load_model: bool = False,
        save_model: bool = False,
        save_phi_model: bool = False,
        load_phi_model: bool = False,
        data_parameters: dict = None,
        model_parameters: dict = None,
    ):

        # Add checks
        if load_data and save_data:
            raise ValueError("load and save data cannot both be True")
        self.data_loaded = False
        self.model_loaded = False
        self.test_data_loaded = False
        self.loaded_phi_model = False

        ########## load or save data ##########
        if save_data:
            # Ensure data/sdata directory exists
            if not os.path.exists("data/sdata"):
                os.makedirs("data/sdata", exist_ok=True)

            # generate random number to create unique folder
            rand_nums = np.random.randint(low=0, high=9, size=5)

            # Check if folder already exists
            while str(rand_nums) in os.listdir("data/sdata"):
                rand_nums = np.random.randint(low=0, high=9, size=5)

            # Create folder to store data
            data_dir = os.path.join("data/sdata", str(rand_nums))
            os.makedirs(data_dir, exist_ok=True)

            # Save training data and labels
            np.save(os.path.join(data_dir, "data_train.npy"), self.data_train)
            np.save(os.path.join(data_dir, "data_test.npy"), self.data_test)
            np.save(os.path.join(data_dir, "labels_train.npy"), self.labels_train)
            np.save(os.path.join(data_dir, "data_test.npy"), self.data_test)
            np.save(os.path.join(data_dir, "labels_test.npy"), self.labels_test)
            filepath = os.path.join(data_dir, "data_parameters.json")

            with open(filepath, "w") as outfile:
                json.dump(data_parameters, outfile)

            print("\rdata saved", end="")
            return

        if load_data:
            # Define folder to load data
            if not os.path.exists("data/sdata"):
                os.makedirs("data/sdata", exist_ok=True)

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

                        print("\rdata loaded", end="")
                        self.data_loaded = True
                        return

        ########## load or save model ##########
        if save_model and load_model:
            raise ValueError("load and save model cannot both be True")

        if save_model:
            if not os.path.exists("model"):
                os.makedirs("model", exist_ok=True)

            # generate random number to create unique folder
            rand_nums = np.random.randint(low=0, high=9, size=5)

            # Check if folder already exists
            while str(rand_nums) in os.listdir("model"):
                rand_nums = np.random.randint(low=0, high=9, size=5)

            # Create folder to store data
            model_dir = os.path.join("model", str(rand_nums))
            os.makedirs(model_dir, exist_ok=True)

            # Save training data and labels
            np.save(os.path.join(model_dir, "weights.npy"), self.weights)
            np.save(os.path.join(model_dir, "spikes_train.npy"), self.spikes_train)
            np.save(os.path.join(model_dir, "mp_train.npy"), self.mp_train)
            np.save(
                os.path.join(model_dir, "weights2plot_exc.npy"),
                self.weights2plot_exc,
            )
            np.save(
                os.path.join(model_dir, "weights2plot_inh.npy"),
                self.weights2plot_inh,
            )
            np.save(
                os.path.join(model_dir, "spike_threshold.npy"), self.spike_threshold
            )
            np.save(
                os.path.join(model_dir, "spikes_test.npy"),
                self.spikes_test,
            )
            np.save(
                os.path.join(model_dir, "max_weight_sum_inh.npy"),
                self.max_weight_sum_inh,
            )
            np.save(
                os.path.join(model_dir, "max_weight_sum_exc.npy"),
                self.max_weight_sum_exc,
            )
            np.save(os.path.join(model_dir, "labels_train.npy"), self.labels_train)
            np.save(os.path.join(model_dir, "labels_test.npy"), self.labels_test)
            np.save(
                os.path.join(model_dir, "performance_tracker.npy"),
                self.performance_tracker,
            )

            filepath = os.path.join(model_dir, "model_parameters.json")

            with open(filepath, "w") as outfile:
                json.dump(model_parameters, outfile)

            print("\rmodel saved", end="")
            return model_dir

        if load_model:
            # Define folder to load data
            if not os.path.exists("model"):
                return
            folders = os.listdir("model")

            # Search for exact parameter match first
            matched_folder = None
            if len(folders) > 0:
                for folder in folders:
                    json_file_path = os.path.join(
                        "model", folder, "model_parameters.json"
                    )
                    if not os.path.exists(json_file_path):
                        continue
                    with open(json_file_path, "r") as j:
                        ex_params = json.loads(j.read())
                    if ex_params == model_parameters:
                        matched_folder = folder
                        break

            # If no exact match, fall back to most recent model folder
            if matched_folder is None and len(folders) > 0:
                try:
                    folders_sorted = sorted(
                        folders,
                        key=lambda f: os.path.getmtime(os.path.join("model", f)),
                        reverse=True,
                    )
                    matched_folder = folders_sorted[0]
                    print("\rNo exact model match; loading latest available.", end="")
                except Exception:
                    matched_folder = None

            if matched_folder is not None:
                folder = matched_folder
                self.weights = np.load(os.path.join("model", folder, "weights.npy"))

                # Optional arrays may be absent depending on save; load if present
                def _maybe_load(name):
                    path = os.path.join("model", folder, name + ".npy")
                    return np.load(path) if os.path.exists(path) else None

                self.spikes_train = _maybe_load("spikes_train")
                self.mp_train = _maybe_load("mp_train")
                self.weights2plot_exc = _maybe_load("weights2plot_exc")
                self.weights2plot_inh = _maybe_load("weights2plot_inh")
                self.spike_threshold = _maybe_load("spike_threshold")
                self.max_weight_sum_inh = _maybe_load("max_weight_sum_inh")
                self.max_weight_sum_exc = _maybe_load("max_weight_sum_exc")
                self.labels_train = _maybe_load("labels_train")
                self.labels_test = _maybe_load("labels_test")
                self.spikes_test = _maybe_load("spikes_test")
                self.performance_tracker = _maybe_load("performance_tracker")

                print("\rmodel loaded", end="")
                self.model_loaded = True
                return os.path.join("model", folder)
            else:
                print(
                    "\rNo model found to load. Will train new model from scratch.",
                    end="",
                )

        if save_phi_model:
            # create sub-folder in already created folder (model dir) for each sleep score
            data_dir = os.path.join(model_dir_, str(sleep_scores))
            os.makedirs(data_dir, exist_ok=True)

            # save phi_all_scores
            np.save(
                os.path.join(data_dir, "phi_all_scores.npy"),
                self.phi_all_scores,
            )

            print("\rphi data has been saved", end="")

        if load_phi_model:
            # get directory
            data_dir = os.path.join(model_dir_, str(sleep_scores))

            # check if path exists
            if os.path.exists(data_dir):
                # load phi_all_scores
                self.phi_all_scores = np.load(
                    os.path.join(data_dir, "phi_all_scores.npy")
                )
                self.loaded_phi_model = True
                print("\rphi data has been loaded", end="")

    # acquire data
    def prepare_data(
        self,
        force_recreate=False,
        plot_comparison=False,
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
        use_validation_data=True,
        val_split=0.2,
        train_split=0.6,
        test_split=0.2,
        audioMNIST=False,
        imageMNIST=False,
        create_data=False,
        plot_spectrograms=False,
        all_audio_train=30000,
        batch_audio_train=500,
        all_audio_test=1000,
        batch_audio_test=200,
        all_audio_val=1000,
        batch_audio_val=100,
        all_images_train=6000,
        batch_image_train=500,
        all_images_test=1000,
        batch_image_test=200,
        all_images_val=1000,
        batch_image_val=100,
    ):
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

        # Update model
        self.data_parameters.update()

        # set parameters
        self.num_steps = num_steps
        self.gain = gain
        self.gain_labels = gain_labels
        self.offset = offset
        self.first_spike_time = first_spike_time
        self.time_var_input = time_var_input
        self.plot_spectrograms = plot_spectrograms
        self.all_images_train = all_images_train
        self.all_images_test = all_images_test
        self.all_images_val = all_images_val
        self.all_audio_train = all_audio_train
        self.all_audio_test = all_audio_test
        self.all_audio_val = all_audio_val
        self.batch_image_train = batch_image_train
        self.batch_image_test = batch_image_test
        self.batch_image_val = batch_image_val
        self.batch_audio_train = batch_audio_train
        self.batch_audio_test = batch_audio_test
        self.batch_audio_val = batch_audio_val
        self.add_breaks = add_breaks
        self.break_lengths = break_lengths
        self.noisy_data = noisy_data
        self.noise_level = noise_level
        self.test_data_ratio = test_data_ratio
        self.val_split = val_split
        self.train_split = train_split
        self.test_split = test_split
        self.all_train = self.all_audio_train if audioMNIST else self.all_images_train
        self.all_test = self.all_audio_test if audioMNIST else self.all_images_test
        self.all_val = self.all_audio_val if audioMNIST else self.all_images_val
        self.batch_train = (
            self.batch_audio_train if audioMNIST else self.batch_image_train
        )
        self.batch_test = self.batch_audio_test if audioMNIST else self.batch_image_test
        self.batch_val = self.batch_audio_val if audioMNIST else self.batch_image_val
        self.epochs = self.all_train // self.batch_train
        self.data_loaded = False
        self.model_loaded = False
        self.test_data_loaded = False
        self.loaded_phi_model = False

        # Initialize streaming data variables
        self.audio_streamer = None
        self.image_streamer = None
        self.current_train_idx = 0
        self.current_test_idx = 0

        # Initialize data attributes
        self.data_train = None
        self.labels_train = None
        self.data_test = None
        self.labels_test = None

        # Set batch sizes and validate total limits
        if audioMNIST and imageMNIST:
            # Multimodal mode - use smaller of the two batch sizes for memory efficiency
            self.batch_train = min(batch_audio_train, batch_image_train)
            self.batch_test = min(batch_audio_test, batch_image_test)

            # Use smaller total for both to ensure synchronization
            self.all_train = min(all_audio_train, all_images_train)
            self.all_test = min(all_audio_test, all_images_test)
            self.epochs = self.all_train // self.batch_train

            # For multimodal, derive total input as 2 * (floor((sqrt(base))/sqrt(2)))^2
            # where base is original single-modality sqrt(self.N_x)
            base_dim = int(np.sqrt(self.N_x))
            shared_dim = int(np.floor(base_dim / np.sqrt(2)))
            self.N_x = int(2 * (shared_dim**2))

            # Update network architecture based on new N_x
            self.st = self.N_x  # stimulation
            self.ex = self.st + self.N_exc  # excitatory
            self.ih = self.ex + self.N_inh  # inhibitory
            self.N = self.N_exc + self.N_inh + self.N_x  # total neurons

            print(f"Multimodal mode total input: {self.N_x} neurons (image + audio)")
            print(
                f"Network: {self.N} total neurons (stim: {self.st}, exc: {self.ex}, inh: {self.ih})"
            )
            print(f"Batch size: {self.batch_train} train, {self.batch_test} test")
            print(f"Total samples: {self.all_train} train, {self.all_test} test")

        elif audioMNIST:
            # Audio only mode
            total_audio = all_audio_train + all_audio_test + all_audio_val
            if total_audio > 30000:
                raise ValueError(
                    f"Total audio samples ({total_audio}) cannot exceed 30000"
                )

            self.batch_train = batch_audio_train
            self.batch_test = batch_audio_test
            self.all_train = all_audio_train
            self.all_test = all_audio_test
            self.epochs = all_audio_train // batch_audio_train

            # Update network architecture (audio uses original N_x)
            self.st = self.N_x  # stimulation
            self.ex = self.st + self.N_exc  # excitatory
            self.ih = self.ex + self.N_inh  # inhibitory
            self.N = self.N_exc + self.N_inh + self.N_x  # total neurons

        elif imageMNIST:
            # Image only mode
            total_images = all_images_train + all_images_test + all_images_val
            if total_images > 30000:
                raise ValueError(
                    f"Total image samples ({total_images}) cannot exceed 30000"
                )

            self.batch_train = batch_image_train
            self.batch_test = batch_image_test
            self.all_train = all_images_train
            self.all_test = all_images_test
            self.epochs = all_images_train // batch_image_train

            # Update network architecture (image uses original N_x)
            self.st = self.N_x  # stimulation
            self.ex = self.st + self.N_exc  # excitatory
            self.ih = self.ex + self.N_inh  # inhibitory
            self.N = self.N_exc + self.N_inh + self.N_x  # total neurons

        # Initialize streamers if needed
        if audioMNIST and not create_data:
            from get_data import AudioDataStreamer

            # Check for existing audio data parameters
            data_dir = "data/mdata"
            audio_data_dir = None
            download_audio = True

            # Ensure data/mdata directory exists
            if not os.path.exists(data_dir):
                os.makedirs(data_dir, exist_ok=True)

            # Search for existing audio data
            folders = os.listdir(data_dir)
            if len(folders) > 0:
                for folder in folders:
                    json_file_path = os.path.join(
                        data_dir, folder, "audio_data_parameters.json"
                    )
                    if os.path.exists(json_file_path):
                        with open(json_file_path, "r") as j:
                            ex_params = json.loads(j.read())

                        # Check if parameters match
                        expected_params = {
                            "batch_size": batch_audio_train,
                            "target_sr": 22050,
                            "duration": 1.0,
                            "N_x": self.N_x,
                            "mode": (
                                "multimodal"
                                if (
                                    hasattr(self, "image_streamer")
                                    and self.image_streamer is not None
                                )
                                else "audio_only"
                            ),
                        }

                        if ex_params == expected_params:
                            audio_data_dir = os.path.join(data_dir, folder)
                            download_audio = False
                            break

            if download_audio:
                # Create new folder for audio data
                rand_nums = np.random.randint(low=0, high=9, size=5)
                while str(rand_nums) in os.listdir(data_dir):
                    rand_nums = np.random.randint(low=0, high=9, size=5)

                audio_data_dir = os.path.join(data_dir, str(rand_nums))
                os.makedirs(audio_data_dir, exist_ok=True)

                # Save audio data parameters
                audio_params = {
                    "batch_size": batch_audio_train,
                    "target_sr": 22050,
                    "duration": 1.0,
                    "N_x": self.N_x,  # Add number of input neurons
                    "mode": (
                        "multimodal"
                        if (
                            hasattr(self, "image_streamer")
                            and self.image_streamer is not None
                        )
                        else "audio_only"
                    ),
                }

                with open(
                    os.path.join(audio_data_dir, "audio_data_parameters.json"), "w"
                ) as f:
                    json.dump(audio_params, f)

            # Use the original audio data path
            data_path = "/home/andreas/Documents/GitHub/AudioMNIST/data"
            self.audio_streamer = AudioDataStreamer(
                data_path,
                batch_size=batch_audio_train,
                train_count=all_audio_train,
                val_count=all_audio_val,
                test_count=all_audio_test,
            )
            print(
                f"Audio streamer initialized with {self.audio_streamer.get_total_samples()} total samples"
            )
            # One-time spectrograms + spikes plot before training if requested
            if (
                getattr(self, "plot_spectrograms", False)
                and not self._did_plot_spectrograms
            ):
                # One-time preview plotting via plot.py helper
                from plot import plot_audio_preview_from_streamer

                training_mode = getattr(
                    self, "get_training_mode", lambda: "audio_only"
                )()
                plot_audio_preview_from_streamer(
                    audio_streamer=self.audio_streamer,
                    num_steps=self.num_steps if hasattr(self, "num_steps") else 100,
                    N_x=self.N_x,
                    training_mode=training_mode,
                    max_batches=50,
                    batch_size=max(100, getattr(self, "batch_train", 500)),
                    sample_rate=22050,
                )
                self._did_plot_spectrograms = True

        if imageMNIST and not create_data:
            from get_data import ImageDataStreamer

            # Check for existing image data parameters
            data_dir = "data/mdata"
            image_data_dir = None
            download_images = True

            # Ensure data/mdata directory exists
            if not os.path.exists(data_dir):
                os.makedirs(data_dir, exist_ok=True)

            # Search for existing image data
            folders = os.listdir(data_dir)
            if len(folders) > 0:
                for folder in folders:
                    json_file_path = os.path.join(
                        data_dir, folder, "image_data_parameters.json"
                    )
                    if os.path.exists(json_file_path):
                        with open(json_file_path, "r") as j:
                            ex_params = json.loads(j.read())

                        # Check if parameters match
                        # Use full N_x for image-only, half for multimodal
                        pixel_size_expected = (
                            int(np.sqrt(self.N_x // 2))
                            if (
                                hasattr(self, "audio_streamer")
                                and self.audio_streamer is not None
                            )
                            else int(np.sqrt(self.N_x))
                        )
                        expected_params = {
                            "pixel_size": pixel_size_expected,
                            "num_steps": num_steps,
                            "gain": gain,
                            "offset": offset,
                            "first_spike_time": first_spike_time,
                            "time_var_input": time_var_input,
                            "N_x": self.N_x,
                            "mode": (
                                "multimodal"
                                if (
                                    hasattr(self, "audio_streamer")
                                    and self.audio_streamer is not None
                                )
                                else "image_only"
                            ),
                        }

                        if ex_params == expected_params:
                            image_data_dir = os.path.join(data_dir, folder)
                            download_images = False
                            break

            if download_images:
                # Create new folder for image data
                rand_nums = np.random.randint(low=0, high=9, size=5)
                while str(rand_nums) in os.listdir(data_dir):
                    rand_nums = np.random.randint(low=0, high=9, size=5)

                image_data_dir = os.path.join(data_dir, str(rand_nums))
                os.makedirs(image_data_dir, exist_ok=True)

                # Save image data parameters
                pixel_size_param = (
                    int(np.sqrt(self.N_x // 2))
                    if (
                        hasattr(self, "audio_streamer")
                        and self.audio_streamer is not None
                    )
                    else int(np.sqrt(self.N_x))
                )
                image_params = {
                    "pixel_size": pixel_size_param,
                    "num_steps": num_steps,
                    "gain": gain,
                    "offset": offset,
                    "first_spike_time": first_spike_time,
                    "time_var_input": time_var_input,
                    "N_x": self.N_x,  # Add number of input neurons
                    "mode": (
                        "multimodal"
                        if (
                            hasattr(self, "audio_streamer")
                            and self.audio_streamer is not None
                        )
                        else "image_only"
                    ),
                }

                with open(
                    os.path.join(image_data_dir, "image_data_parameters.json"), "w"
                ) as f:
                    json.dump(image_params, f)

            # Determine pixel size based on mode
            pixel_size_streamer = (
                int(np.sqrt(self.N_x // 2))
                if (hasattr(self, "audio_streamer") and self.audio_streamer is not None)
                else int(np.sqrt(self.N_x))
            )
            self.image_streamer = ImageDataStreamer(
                "data",  # Use the data directory for MNIST download
                batch_size=batch_image_train,
                pixel_size=pixel_size_streamer,
                num_steps=num_steps,
                gain=gain,
                offset=offset,
                first_spike_time=first_spike_time,
                time_var_input=time_var_input,
                train_count=all_images_train,
                val_count=all_images_val,
                test_count=all_images_test,
            )
            print(
                f"Image streamer initialized with {self.image_streamer.get_total_samples()} total samples"
            )

        # Ensure multimodal pre-alignment if both streamers exist
        if (
            hasattr(self, "audio_streamer")
            and self.audio_streamer is not None
            and hasattr(self, "image_streamer")
            and self.image_streamer is not None
        ):
            try:
                from get_data import create_prealigned_multimodal_datasets

                print(
                    "Pre-aligning multimodal datasets (strict class-matched indices)..."
                )
                # Derive ratios honoring requested validation sample counts
                try:
                    val_count = int(min(all_audio_val, all_images_val))
                except Exception:
                    val_count = 0
                total_count = int(self.all_train + self.all_test + val_count)
                train_ratio = (
                    float(self.all_train) / float(total_count) if total_count else 1.0
                )
                val_ratio = (
                    float(val_count) / float(total_count) if total_count else 0.0
                )
                test_ratio = (
                    float(self.all_test) / float(total_count) if total_count else 0.0
                )

                self.audio_streamer, self.image_streamer = (
                    create_prealigned_multimodal_datasets(
                        audio_streamer=self.audio_streamer,
                        image_streamer=self.image_streamer,
                        max_total_samples=min(total_count, 30000),
                        train_ratio=train_ratio,
                        val_ratio=val_ratio,
                        test_ratio=test_ratio,
                    )
                )
            except Exception as e:
                print(
                    f"Warning: pre-alignment failed ({e}); proceeding with sequential streaming."
                )

        if create_data:
            # create data
            if not force_recreate:
                self.process(load_data=True, data_parameters=self.data_parameters)

            if force_recreate or not self.data_loaded:
                # Define data parameters
                data_parameters = {
                    "pixel_size": int(np.sqrt(self.N_x)),
                    "train_": train_,
                }

                # Ensure data/mdata directory exists
                if not os.path.exists("data/mdata"):
                    os.makedirs("data/mdata")

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

                # Clean up data directory search variables
                del folders

                # get dataset with progress bar
                print("\rDownloading MNIST dataset...", end="")
                if download == True:
                    # Ensure data/mdata directory exists
                    if not os.path.exists("data/mdata"):
                        os.makedirs("data/mdata", exist_ok=True)

                    # generate random number to create unique folder
                    rand_nums = np.random.randint(low=0, high=9, size=5)

                    # Check if folder already exists
                    while str(rand_nums) in os.listdir("data/mdata"):
                        rand_nums = np.random.randint(low=0, high=9, size=5)

                    # Create folder to store data
                    data_dir = os.path.join("data/mdata", str(rand_nums))
                    os.makedirs(data_dir, exist_ok=True)

                    # Save data parameters
                    filepath = os.path.join(data_dir, "data_parameters.json")

                    with open(filepath, "w") as outfile:
                        json.dump(data_parameters, outfile)

                if use_validation_data:
                    (
                        self.data_train,
                        self.labels_train,
                        self.data_test,
                        self.labels_test,
                    ) = create_data(
                        pixel_size=int(np.sqrt(self.N_x)),
                        num_steps=num_steps,
                        plot_comparison=plot_comparison,
                        gain=gain,
                        offset=offset,
                        download=download,
                        data_dir=data_dir,
                        first_spike_time=first_spike_time,
                        time_var_input=time_var_input,
                        num_images_train=self.batch_train,
                        num_images_test=self.batch_test,
                        add_breaks=add_breaks,
                        break_lengths=break_lengths,
                        noisy_data=noisy_data,
                        noise_level=noise_level,
                        use_validation_data=True,
                        val_split=self.val_split,
                        train_split=self.train_split,
                        test_split=self.test_split,
                        audioMNIST=audioMNIST,
                        imageMNIST=imageMNIST,
                    )

                    # Handle streaming audio data
                    if audioMNIST and hasattr(self.data_train, "get_total_samples"):
                        self.audio_streamer = self.data_train
                        # Initialize data_train and labels_train for compatibility
                        self.data_train = None
                        self.labels_train = None
                        print("Audio data set to streaming mode")
                else:
                    (
                        self.data_train,
                        self.labels_train,
                        self.data_test,
                        self.labels_test,
                    ) = create_data(
                        pixel_size=int(np.sqrt(self.N_x)),
                        num_steps=num_steps,
                        plot_comparison=plot_comparison,
                        gain=gain,
                        offset=offset,
                        download=download,
                        data_dir=data_dir,
                        first_spike_time=first_spike_time,
                        time_var_input=time_var_input,
                        num_images_train=self.batch_train,
                        num_images_test=self.batch_test,
                        add_breaks=add_breaks,
                        break_lengths=break_lengths,
                        noisy_data=noisy_data,
                        noise_level=noise_level,
                        idx_train=0,
                        idx_test=0,
                        use_validation_data=False,
                    )
                self.process(save_data=True, data_parameters=self.data_parameters)

            # plot spikes
            if plot_spikes:
                if min_time == None:
                    min_time = 0
                if max_time == None:
                    max_time = self.T_train
                spike_plot(
                    self.data_train[min_time:max_time],
                    self.labels_train[min_time:max_time],
                )

            # plot heatmap of activity
            if plot_heat_map:
                heat_map(self.data_train, pixel_size=10)

            # return data and labels if needed
            if retur:
                if use_validation_data:
                    return (
                        self.data_train,
                        self.labels_train,
                    )
                else:
                    return self.data_train, self.labels_train

    def load_audio_batch_streaming(
        self, batch_size, is_training=True, plot_spectrograms=False, partition=None
    ):
        """
        Load a batch of audio data and convert to spikes on-demand.

        Args:
            batch_size: Number of samples to load
            is_training: If True, load training data; if False, load test data

        Returns:
            (spike_data, labels) or (None, None) if no more data
        """
        spike_data, labels, new_train_idx, new_test_idx = load_audio_batch_streaming(
            audio_streamer=self.audio_streamer,
            batch_size=batch_size,
            current_train_idx=self.current_train_idx,
            current_test_idx=self.current_test_idx,
            all_train=self.all_train,
            all_test=self.all_test,
            num_steps=self.num_steps,
            N_x=self.N_x,
            get_training_mode_func=self.get_training_mode,
            is_training=is_training,
            plot_spectrograms=plot_spectrograms,
            partition=partition,
        )

        # Update indices
        self.current_train_idx = new_train_idx
        self.current_test_idx = new_test_idx

        return spike_data, labels

    def load_multimodal_batch(
        self, batch_size, is_training=True, plot_spectrograms=False, partition=None
    ):
        """
        Load a batch of multimodal data (image + audio) with synchronized labels.

        Args:
            batch_size: Number of samples to load
            is_training: If True, load training data; if False, load test data

        Returns:
            (concatenated_spike_data, labels) or (None, None) if no more data
        """
        multimodal_spikes, labels, new_train_idx, new_test_idx = load_multimodal_batch(
            audio_streamer=self.audio_streamer,
            image_streamer=self.image_streamer,
            batch_size=batch_size,
            current_train_idx=self.current_train_idx,
            current_test_idx=self.current_test_idx,
            all_train=self.all_train,
            all_test=self.all_test,
            num_steps=self.num_steps,
            N_x=self.N_x,
            get_training_mode_func=self.get_training_mode,
            load_prealigned_func=load_prealigned_multimodal_batch,
            is_training=is_training,
            plot_spectrograms=plot_spectrograms,
            partition=partition,
        )

        # Update indices
        self.current_train_idx = new_train_idx
        self.current_test_idx = new_test_idx

        return multimodal_spikes, labels

    def _load_prealigned_multimodal_batch(
        self,
        batch_size,
        is_training=True,
        plot_spectrograms=False,
        partition=None,
        num_image_neurons=None,
        num_audio_neurons=None,
    ):
        """
        Load a batch from multimodal datasets with synchronized shuffling.
        Both datasets are shuffled with the same seed to ensure similar label distributions.
        """
        # Ensure datasets are synchronized before loading
        if not hasattr(self, "multimodal_synced") or not self.multimodal_synced:
            sync_multimodal_datasets()
            self.multimodal_synced = True

        # Call the function from get_data.py
        return load_prealigned_multimodal_batch(
            audio_streamer=self.audio_streamer,
            image_streamer=self.image_streamer,
            batch_size=batch_size,
            current_train_idx=self.current_train_idx,
            current_test_idx=self.current_test_idx,
            all_train=self.all_train,
            all_test=self.all_test,
            num_steps=self.num_steps,
            N_x=self.N_x,
            get_training_mode_func=self.get_training_mode,
            is_training=is_training,
            plot_spectrograms=plot_spectrograms,
            partition=partition,
            num_image_neurons=num_image_neurons,
            num_audio_neurons=num_audio_neurons,
        )

    def _sync_multimodal_datasets(self):
        """
        Synchronize multimodal datasets by ensuring they use the same random seed.
        This ensures that when both datasets are shuffled, they have similar label distributions.
        """
        if not hasattr(self, "multimodal_synced") or not self.multimodal_synced:
            sync_multimodal_datasets()

            # Reset indices to start fresh with synchronized shuffling
            self.current_train_idx = 0
            self.current_test_idx = 0

            self.multimodal_synced = True

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
            "epochs": self.epochs,
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
        plot_network=False,
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
            plot_network=plot_network,
        )

        if create_network:
            # create other arrays
            (
                self.mp_train,
                self.mp_test,
                self.spikes_train,
                self.spikes_test,
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
                    self.spike_times,
                    self.resting_potential,
                    self.max_time,
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
        train_weights=False,
        learning_rate_exc=0.0008,
        learning_rate_inh=0.005,
        w_target_exc=0.01,
        w_target_inh=-0.01,
        var_noise=1,
        min_weight_inh=-25,
        max_weight_inh=-0.01,
        max_weight_exc=25,
        min_weight_exc=0.01,
        spike_threshold_default=-55,
        check_sleep_interval=50000,  # Reduced frequency for better performance
        interval=5000,  # Reduced frequency for better performance
        min_mp=-100,
        sleep=True,
        force_train=False,
        save_model=True,
        weight_decay=False,
        weight_decay_rate_exc=[0.9999],
        weight_decay_rate_inh=[0.9999],
        compare_decay_rates=True,
        noisy_potential=True,
        noisy_threshold=False,
        noisy_weights=False,
        spike_adaption=True,
        delta_adaption=3,
        tau_adaption=100,
        trace_update=False,
        timing_update=True,
        vectorized_trace=False,
        clip_exc_weights=False,
        clip_inh_weights=False,
        alpha=1.1,
        beta=1.0,
        A_plus=0.5,
        A_minus=0.5,
        tau_LTD=10,
        tau_LTP=10,
        early_stopping=False,
        dt=1,
        tau_m=30,
        membrane_resistance=30,
        reset_potential=-80,
        spike_slope=-0.1,
        spike_intercept=-4,
        pca_variance=0.95,
        w_interval=5,
        start_time_spike_plot=None,
        stop_time_spike_plot=None,
        start_index_mp=None,
        stop_index_mp=None,
        time_start_mp=None,
        time_stop_mp=None,
        mean_noise=0,
        max_mp=40,
        sleep_synchronized=True,
        tau_pre_trace_exc=1,
        tau_pre_trace_inh=1,
        tau_post_trace_exc=1,
        tau_post_trace_inh=1,
        weight_mean_noise=0.05,
        weight_var_noise=0.005,
        num_inh=10,
        num_exc=50,
        plot_epoch_performance=True,
        narrow_top=0.2,  # Increased from 0.05 to 0.2 (20% of neurons)
        wide_top=0.15,
        tau_syn=30,
        smoothening=350,
        plot_top_response_train=False,
        plot_top_response_test=False,
        plot_tsne_during_training=True,  # New parameter for t-SNE plotting
        tsne_plot_interval=1,  # Plot t-SNE every N epochs (1 = every epoch)
        plot_spectrograms=False,
        random_state=48,
        samples=10,
        use_validation_data=False,
        accuracy_method="top",
        test_only=False,
        test_batch_size=None,
        patience=None,
    ):
        self.dt = dt
        self.pca_variance = pca_variance
        self.use_validation_data = use_validation_data

        # Save current parameters
        self.model_parameters = {**locals()}
        remove = [
            "self",
            "force_train",
            "save_model",
            "plot_mp_train",
            "plot_mp_test",
            "plot_spikes_train",
            "plot_spikes_test",
            "plot_weights",
            "plot_threshold",
            "plot_traces_",
            "start_time_spike_plot",
            "stop_time_spike_plot",
            "start_index_mp",
            "stop_index_mp",
            "time_start_mp",
            "time_stop_mp",
            "plot_top_response_train",
            "plot_top_response_test",
        ]

        # Remove elements from model_parameters
        for element in remove:
            self.model_parameters.pop(element)

        self.model_parameters["all_train"] = self.all_train
        self.model_parameters["all_test"] = self.all_test
        self.model_parameters["all_val"] = self.all_val
        self.model_parameters["batch_train"] = self.batch_train
        self.model_parameters["batch_test"] = self.batch_test
        self.model_parameters["batch_val"] = self.batch_val
        self.model_parameters["epochs"] = self.epochs
        self.model_parameters["w_dense_ee"] = self.w_dense_ee
        self.model_parameters["w_dense_ei"] = self.w_dense_ei
        self.model_parameters["w_dense_se"] = self.w_dense_se
        self.model_parameters["w_dense_ie"] = self.w_dense_ie

        self.model_parameters["ie_weights"] = self.ie_weights
        self.model_parameters["ee_weights"] = self.ee_weights
        self.model_parameters["ei_weights"] = self.ei_weights
        self.model_parameters["se_weights"] = self.se_weights
        self.model_parameters["classes"] = self.classes

        # Always attempt to load a matching model if not forcing a fresh run,
        # regardless of data_loaded (streaming modes don't set data_loaded).
        # In test-only inference with frozen weights, skip loading any saved model
        if not force_train and not (test_only and not train_weights):
            try:
                model_dir = self.process(
                    load_model=True,
                    model_parameters=self.model_parameters,
                )
            except Exception as e:
                print(f"Warning: model load skipped ({e})")

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
                data_parameters = {"pixel_size": int(np.sqrt(self.N_x)), "train_": True}

                # Ensure data/mdata directory exists
                if not os.path.exists("data/mdata"):
                    os.makedirs("data/mdata", exist_ok=True)

                # Define folder to load data
                folders = os.listdir("data/mdata")

                # Search for existing data
                if len(folders) > 0:
                    for folder in folders:
                        json_file_path = os.path.join(
                            "data", "mdata", folder, "data_parameters.json"
                        )
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
            # define which weights counts towards total sum of weights
            sum_weights_exc = np.sum(np.abs(self.weights[: self.ex, self.st : self.ih]))
            sum_weights_inh = np.sum(
                np.abs(self.weights[self.ex : self.ih, self.st : self.ex])
            )
            sum_weights = np.sum(np.abs(self.weights))

            baseline_sum_exc = sum_weights_exc * beta
            baseline_sum_inh = sum_weights_inh * beta
            baseline_sum = sum_weights * beta
            max_sum_exc = sum_weights_exc * alpha
            max_sum_inh = sum_weights_inh * alpha
            max_sum = sum_weights * alpha

            # Bundle common training arguments
            common_args = dict(
                tau_syn=tau_syn,
                resting_potential=self.resting_potential,
                membrane_resistance=membrane_resistance,
                min_weight_exc=min_weight_exc,
                max_weight_exc=max_weight_exc,
                min_weight_inh=min_weight_inh,
                max_weight_inh=max_weight_inh,
                N_exc=self.N_exc,
                N_inh=self.N_inh,
                max_sum=max_sum,
                max_sum_exc=max_sum_exc,
                max_sum_inh=max_sum_inh,
                baseline_sum=baseline_sum,
                baseline_sum_exc=baseline_sum_exc,
                baseline_sum_inh=baseline_sum_inh,
                beta=beta,
                sleep_synchronized=sleep_synchronized,
                num_exc=num_exc,
                num_inh=num_inh,
                weight_decay=weight_decay,
                weight_decay_rate_exc=weight_decay_rate_exc[0],
                weight_decay_rate_inh=weight_decay_rate_inh[0],
                learning_rate_exc=learning_rate_exc,
                learning_rate_inh=learning_rate_inh,
                w_target_exc=w_target_exc,
                w_target_inh=w_target_inh,
                tau_LTP=tau_LTP,
                tau_LTD=tau_LTD,
                tau_m=tau_m,
                max_mp=max_mp,
                min_mp=min_mp,
                interval=interval,
                dt=self.dt,
                N=self.N,
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
                noisy_potential=noisy_potential,
                noisy_weights=noisy_weights,
                weight_mean_noise=weight_mean_noise,
                weight_var_noise=weight_var_noise,
                vectorized_trace=vectorized_trace,
                N_x=self.N_x,
            )

            # Clean up weight calculation variables after they're used in common_args
            del sum_weights_exc, sum_weights_inh, sum_weights
            del baseline_sum_exc, baseline_sum_inh, baseline_sum
            del max_sum_exc, max_sum_inh, max_sum

            # pre-define performance tracking array
            if test_only and not train_weights:
                # single pass; tracker will be (1,2) later
                self.performance_tracker = np.zeros((1, 2))
            else:
                self.performance_tracker = np.zeros((self.epochs, 2))

            # early stopping disabled
            interval_ES = None

            # define progress bar
            pbar_total = 1 if (test_only and not train_weights) else self.epochs
            pbar = tqdm(
                total=pbar_total,
                desc=(
                    "Test-only"
                    if (test_only and not train_weights)
                    else f"Epoch 0/{self.epochs}:"
                ),
                unit="it",
                ncols=80,
                bar_format="{desc} [{bar}] ETA: {remaining} |{postfix}",
            )
            # Predefine outputs to avoid UnboundLocalError if loop exits early
            spikes_tr_out = None
            labels_tr_out = None
            sleep_tr_out = None
            mp_tr = None
            w4p_exc_tr = None
            w4p_inh_tr = None
            thresh_tr = None
            mx_w_inh_tr = None
            mx_w_exc_tr = None
            spikes_te_out = None
            labels_te_out = None
            mp_te = None

            # create missing arrays
            I_syn = np.zeros(self.N - self.st)
            spike_times = np.zeros(self.N)
            a = np.zeros(self.N - self.st)

            # create spike threshold array
            spike_threshold = np.full(
                shape=(self.ih - self.st),
                fill_value=spike_threshold_default,
                dtype=float,
            )
            # Fast path: test-only mode (no SNN weight training, single pass over test set)
            if test_only and not train_weights:
                try:
                    self.current_test_idx = 0
                    effective_batch = (
                        int(test_batch_size)
                        if (test_batch_size is not None and int(test_batch_size) > 0)
                        else self.batch_test
                    )
                    # Number of test batches (ceil division to include remainder)
                    total_num_tests = max(
                        1, (self.all_test + effective_batch - 1) // effective_batch
                    )

                    # Replace the generic epoch pbar with a dedicated test-only pbar
                    try:
                        pbar.close()
                    except Exception:
                        pass
                    pbar = tqdm(
                        total=total_num_tests,
                        desc=f"Test 0/{total_num_tests}",
                        unit="batch",
                        ncols=80,
                        bar_format="{desc} [{bar}] ETA: {remaining} |{postfix}",
                    )

                    # Pre-allocate accumulation buffers
                    max_test_samples = self.all_test * self.num_steps
                    all_spikes_test = np.zeros(
                        (max_test_samples, self.N), dtype=np.int8
                    )
                    all_labels_test = np.zeros(max_test_samples, dtype=np.int32)
                    all_mp_test = np.zeros(
                        (max_test_samples, self.N - self.N_x), dtype=np.float32
                    )
                    test_sample_count = 0
                    test_acc = 0.0
                    test_phi = 0.0
                    test_acc_top = 0.0  # accumulate only if accuracy_method == "top"

                    for test_batch_idx in range(total_num_tests):
                        # Load test data for test-only evaluation
                        if (
                            self.audio_streamer is not None
                            and self.image_streamer is not None
                        ):
                            # Multimodal streaming mode
                            data_test, labels_test = self.load_multimodal_batch(
                                effective_batch,
                                is_training=False,
                                plot_spectrograms=plot_spectrograms,
                                partition="test",
                            )
                            if data_test is None or labels_test is None:
                                raise ValueError("No test data available")
                        elif self.audio_streamer is not None:
                            data_test, labels_test = self.load_audio_batch_streaming(
                                batch_size=effective_batch,
                                is_training=False,
                                plot_spectrograms=plot_spectrograms,
                                partition="test",
                            )
                            if data_test is None:
                                raise ValueError("No test data available")
                            self.current_test_idx += effective_batch
                        elif self.image_streamer is not None:
                            data_test, labels_test = load_image_batch(
                                self.image_streamer,
                                self.current_test_idx,
                                effective_batch,
                                self.num_steps,
                                int(np.sqrt(self.N_x)) ** 2,
                                partition="test",
                            )
                            if data_test is None:
                                raise ValueError("No test data available")
                            self.current_test_idx += effective_batch
                        else:
                            # Pre-loaded data path - prefer full test data
                            if self.data_test is not None:
                                data_test = self.data_test
                                labels_test = self.labels_test
                            elif (
                                hasattr(self, "data_val") and self.data_val is not None
                            ):
                                data_test = self.data_val
                                labels_test = self.labels_val
                            else:
                                raise ValueError("No test data available")

                        # Adjust width if needed
                        if data_test.shape[1] != self.N_x:
                            actual = data_test.shape[1]
                            if actual < self.N_x:
                                pad = self.N_x - actual
                                data_test = np.concatenate(
                                    [
                                        data_test,
                                        np.zeros(
                                            (data_test.shape[0], pad),
                                            dtype=data_test.dtype,
                                        ),
                                    ],
                                    axis=1,
                                )
                            else:
                                data_test = data_test[:, : self.N_x]

                        T_te = data_test.shape[0]
                        st = self.N_x
                        ex = st + self.N_exc
                        ih = ex + self.N_inh

                        mp_test = np.zeros((T_te, ih - st))
                        mp_test[0] = self.resting_potential
                        spikes_test = np.zeros((T_te, self.N), dtype=np.int8)
                        spikes_test[:, :st] = data_test

                        (
                            _weights_te,
                            spikes_te_out,
                            mp_te,
                            *unused,
                            labels_te_out,
                            _sleep_te_out,
                            _I_syn_te,
                            _spike_times_te,
                            _a_te,
                        ) = train_network(
                            weights=self.weights.copy(),
                            spike_labels=labels_test.copy(),
                            mp=mp_test.copy(),
                            sleep=False,
                            train_weights=False,
                            T=T_te,
                            mean_noise=mean_noise,
                            var_noise=var_noise,
                            spikes=spikes_test.copy(),
                            check_sleep_interval=check_sleep_interval,
                            timing_update=timing_update,
                            spike_times=spike_times.copy(),
                            a=a.copy(),
                            I_syn=I_syn.copy(),
                            spike_threshold=spike_threshold.copy(),
                            **common_args,
                        )

                        # store
                        bs = spikes_te_out.shape[0]
                        all_spikes_test[test_sample_count : test_sample_count + bs] = (
                            spikes_te_out
                        )
                        all_labels_test[test_sample_count : test_sample_count + bs] = (
                            labels_te_out
                        )
                        all_mp_test[test_sample_count : test_sample_count + bs] = mp_te
                        test_sample_count += bs

                        # Compute batch accuracy depending on requested method
                        if (
                            isinstance(accuracy_method, str)
                            and accuracy_method.lower() == "top"
                        ):
                            acc_te_batch = top_responders_plotted(
                                spikes=spikes_te_out[:, self.st : self.ih],
                                labels=labels_te_out,
                                num_classes=self.N_classes,
                                narrow_top=narrow_top,
                                smoothening=self.num_steps,
                                train=False,
                                compute_not_plot=True,
                                n_last_points=10000,
                            )
                            test_acc_top += acc_te_batch

                        # progress bar update for test-only batches
                        pbar.set_description(
                            f"Test {test_batch_idx+1}/{total_num_tests}"
                        )
                        pbar.update(1)

                    # finalize
                    spikes_te_out = all_spikes_test[:test_sample_count]
                    labels_te_out = all_labels_test[:test_sample_count]
                    mp_te = all_mp_test[:test_sample_count]

                    # Compute metrics at the end of test-only pass
                    acc_top = None
                    if (
                        isinstance(accuracy_method, str)
                        and accuracy_method.lower() == "top"
                    ):
                        acc_top = test_acc_top / max(1, total_num_tests)
                    try:
                        # PCA+LR using only test features (split test into train/val/test internally)
                        X_te, y_te = bin_spikes_by_label_no_breaks(
                            spikes=spikes_te_out[:, self.st : self.ih],
                            labels=labels_te_out,
                        )
                        if X_te.size == 0:
                            acc_pca = 0.0
                        else:
                            rng = np.random.RandomState(42)
                            idx = rng.permutation(X_te.shape[0])
                            split_tr = max(1, int(0.6 * len(idx)))
                            split_va = max(split_tr + 1, int(0.8 * len(idx)))
                            tr_idx = idx[:split_tr]
                            va_idx = idx[split_tr:split_va]
                            te_idx = idx[split_va:]
                            if te_idx.size == 0:
                                te_idx = va_idx
                            accs, _, _, _ = pca_logistic_regression(
                                X_train=X_te[tr_idx],
                                y_train=y_te[tr_idx],
                                X_val=X_te[va_idx],
                                y_val=y_te[va_idx],
                                X_test=X_te[te_idx],
                                y_test=y_te[te_idx],
                                variance_ratio=self.pca_variance,
                            )
                            acc_pca = float(accs.get("test", 0.0))
                    except Exception as ex:
                        print(f"Warning: PCA+LR (test-only) failed ({ex}); using 0.0")
                        acc_pca = 0.0

                    # performance tracker for single pass: [top_acc, pca_lr_acc]
                    self.performance_tracker = np.zeros((1, 2))
                    self.performance_tracker[0, 0] = (
                        float(acc_top) if acc_top is not None else 0.0
                    )
                    self.performance_tracker[0, 1] = acc_pca

                    if acc_top is not None:
                        print(
                            f"Test-only summary — Top responders acc: {acc_top:.4f}, PCA+LR acc: {acc_pca:.4f}"
                        )
                    else:
                        print(
                            f"Test-only summary — PCA+LR acc: {acc_pca:.4f} (top metric disabled)"
                        )

                    # Close test-only pbar
                    try:
                        pbar.close()
                    except Exception:
                        pass

                    # expose outputs
                    self.spikes_test = spikes_te_out
                    self.mp_test = mp_te
                    self.labels_test = labels_te_out

                    # Clean up test arrays and variables
                    del all_spikes_test, all_labels_test, all_mp_test
                    del test_acc_top, test_sample_count

                finally:
                    pbar.close()
                    del (
                        I_syn,
                        spike_times,
                        a,
                        spike_threshold,
                    )
                    del (common_args,)
                    gc.collect()
                return

            # loop over self.epochs
            for e in range(self.epochs):

                # Reset test/val indices at the beginning of each epoch
                self.current_test_idx = 0
                # Reset streamer validation/train pointers (streamers ignore start_idx)
                try:
                    if (
                        hasattr(self, "audio_streamer")
                        and self.audio_streamer is not None
                    ):
                        self.audio_streamer.reset_partition("val")
                        self.audio_streamer.reset_partition("train")
                    if (
                        hasattr(self, "image_streamer")
                        and self.image_streamer is not None
                    ):
                        self.image_streamer.reset_partition("val")
                        self.image_streamer.reset_partition("train")
                except Exception:
                    pass

                # Load data for this epoch
                if (
                    self.audio_streamer is not None
                    and hasattr(self, "image_streamer")
                    and self.image_streamer is not None
                ):
                    # Multimodal mode - load both audio and image data
                    data_train, labels_train = self.load_multimodal_batch(
                        self.batch_train,
                        is_training=True,
                        plot_spectrograms=plot_spectrograms,
                    )
                    if data_train is None:
                        print(f"No more multimodal data available at epoch {e}")
                        break
                elif self.audio_streamer is not None:
                    # Audio only mode
                    data_train, labels_train, new_train_idx, new_test_idx = (
                        load_audio_batch_streaming(
                            audio_streamer=self.audio_streamer,
                            batch_size=self.batch_train,
                            current_train_idx=self.current_train_idx,
                            current_test_idx=self.current_test_idx,
                            all_train=self.all_train,
                            all_test=self.all_test,
                            num_steps=self.num_steps,
                            N_x=self.N_x,
                            get_training_mode_func=self.get_training_mode,
                            is_training=True,
                            plot_spectrograms=plot_spectrograms,
                        )
                    )
                    if data_train is None:
                        print(f"No more audio data available at epoch {e}")
                        break
                    # Update training index
                    self.current_train_idx = new_train_idx
                elif self.image_streamer is not None:
                    # Image only streaming mode (missing branch)
                    from get_data import load_image_batch

                    train_start_idx = self.current_train_idx
                    data_train, labels_train = load_image_batch(
                        self.image_streamer,
                        train_start_idx,
                        self.batch_train,
                        self.num_steps,
                        int(np.sqrt(self.N_x)) ** 2,  # Image-only uses full N_x
                    )
                    if data_train is None:
                        print(f"No more image data available at epoch {e}")
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
                        _,
                        spikes_train,
                        _,
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
                else:
                    # Reuse pre-allocated arrays
                    mp_train = np.zeros((self.T_train, self.N - self.N_x))
                    spikes_train = np.zeros((self.T_train, self.N), dtype=np.int8)
                    # Copy data to spikes array
                    spikes_train[:, : self.N_x] = data_train

                # 3a) Train on the training set
                # Ensure data width matches expected N_x
                if data_train is not None and data_train.shape[1] != self.N_x:
                    actual_nx = data_train.shape[1]
                    print(
                        f"Warning: Adjusting training input width from {actual_nx} to {self.N_x}"
                    )
                    if actual_nx < self.N_x:
                        pad_width = self.N_x - actual_nx
                        pad_block = np.zeros(
                            (data_train.shape[0], pad_width), dtype=data_train.dtype
                        )
                        data_train = np.concatenate([data_train, pad_block], axis=1)
                    else:
                        data_train = data_train[:, : self.N_x]
                (
                    self.weights,
                    spikes_tr_out,
                    mp_tr,
                    w4p_exc_tr,
                    w4p_inh_tr,
                    thresh_tr,
                    mx_w_inh_tr,
                    mx_w_exc_tr,
                    labels_tr_out,
                    sleep_tr_out,
                    I_syn,
                    spike_times,
                    a,
                ) = train_network(
                    weights=(self.weights if train_weights else self.weights.copy()),
                    spike_labels=labels_train,
                    mp=mp_train,
                    sleep=sleep,
                    train_weights=train_weights,
                    T=self.T_train,
                    mean_noise=mean_noise,
                    var_noise=var_noise,
                    spikes=spikes_train,
                    check_sleep_interval=check_sleep_interval,
                    timing_update=timing_update,
                    spike_times=spike_times,
                    spike_threshold=spike_threshold,
                    a=a,
                    I_syn=I_syn,
                    **common_args,
                )

                # Calculate training accuracy for current epoch
                if spikes_tr_out is not None and labels_tr_out is not None:
                    print(
                        f"\n--- Epoch {e+1} Training Accuracy ({accuracy_method.upper()}) ---"
                    )

                    if accuracy_method == "top":
                        # Use top-responders method for training accuracy
                        train_acc_batch = top_responders_plotted(
                            spikes=spikes_tr_out[:, self.st : self.ih],
                            labels=labels_tr_out,
                            num_classes=self.N_classes,
                            narrow_top=narrow_top,
                            smoothening=self.num_steps,
                            train=True,
                            compute_not_plot=True,
                            n_last_points=10000,
                        )
                        print(f"Training Accuracy (TOP): {train_acc_batch:.4f}")

                    elif accuracy_method == "pca_lr":
                        # Use PCA+LR method for training accuracy
                        try:
                            X_tr, y_tr = bin_spikes_by_label_no_breaks(
                                spikes=spikes_tr_out[:, self.st : self.ih],
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

                                # Calculate training accuracy using PCA+LR with proper train/test split
                                # Use 80% for training, 20% for testing to avoid data leakage
                                from sklearn.model_selection import train_test_split

                                if len(X_tr) >= 10:  # Need minimum samples for split
                                    X_tr_split, X_te_split, y_tr_split, y_te_split = (
                                        train_test_split(
                                            X_tr,
                                            y_tr,
                                            test_size=0.2,
                                            random_state=42,
                                            stratify=y_tr,
                                        )
                                    )
                                    print(
                                        f"PCA+LR split: {len(X_tr_split)} train, {len(X_te_split)} test samples"
                                    )
                                else:
                                    # Not enough samples for proper split, use original approach but note the limitation
                                    print(
                                        f"⚠️  Only {len(X_tr)} samples - using same data for train/test (limited reliability)"
                                    )
                                    X_tr_split, X_te_split = X_tr, X_tr
                                    y_tr_split, y_te_split = y_tr, y_tr

                                accs, scaler, pca, clf = pca_logistic_regression(
                                    X_train=X_tr_split,
                                    y_train=y_tr_split,
                                    X_val=X_tr_split,  # Use training split for validation
                                    y_val=y_tr_split,
                                    X_test=X_te_split,  # Use test split for final evaluation
                                    y_test=y_te_split,
                                    variance_ratio=self.pca_variance,
                                )
                                train_acc_pca = float(accs.get("test", 0.0))
                                print(
                                    f"Training Accuracy (PCA+LR): {train_acc_pca:.4f}"
                                )

                                # Show training data distribution
                                train_dist = np.bincount(y_tr, minlength=self.N_classes)
                                print(f"\nTraining Data Distribution:")
                                for i in range(self.N_classes):
                                    count = train_dist[i]
                                    percentage = (
                                        (count / len(y_tr)) * 100
                                        if len(y_tr) > 0
                                        else 0
                                    )
                                    print(
                                        f"  Class {i}: {count:3d} samples ({percentage:5.1f}%)"
                                    )
                            else:
                                print("No training features available for PCA+LR")
                        except Exception as ex:
                            print(f"Training PCA+LR failed: {ex}")

                    print(f"{'-'*50}")

                total_num_vals = (
                    self.all_test // self.batch_test
                )  # Use test dataset size for validation batching
                val_acc = 0
                val_phi = 0

                # Initialize arrays to store accumulated validation results
                all_spikes_val = []
                all_labels_val = []
                all_mp_val = []

                # Pre-allocate arrays for better memory management
                max_val_samples = (
                    self.all_test * self.num_steps
                )  # Use test dataset size for pre-allocation
                all_spikes_val = np.zeros((max_val_samples, self.N), dtype=np.int8)
                all_labels_val = np.zeros(max_val_samples, dtype=np.int32)
                all_mp_val = np.zeros(
                    (max_val_samples, self.N - self.N_x), dtype=np.float32
                )
                val_sample_count = 0
                val_batches_counted = 0

                # Predefine variables used in the test loop so cleanup never fails
                data_test = None
                labels_test = None
                mp_test = None
                weights_te = None
                spikes_te_out = None
                labels_te_out = None
                sleep_te_out = None
                I_syn_te = None
                spike_times_te = None
                a_te = None
                T_test_batch = 0
                st = self.N_x
                ex = st + self.N_exc
                ih = ex + self.N_inh

                for val_batch_idx in range(total_num_vals):
                    # Load validation data for training monitoring
                    if (
                        self.audio_streamer is not None
                        and self.image_streamer is not None
                    ):
                        # Multimodal streaming mode
                        data_test, labels_test = self.load_multimodal_batch(
                            self.batch_test,
                            is_training=False,
                            plot_spectrograms=plot_spectrograms,
                            partition="val",  # Use validation data
                        )
                        if data_test is None:
                            print(f"No more validation multimodal data available")
                            break
                    elif self.audio_streamer is not None:
                        # Audio only streaming mode
                        from get_data import load_audio_batch

                        val_start_idx = val_batch_idx * self.batch_test
                        data_test, labels_test = load_audio_batch(
                            self.audio_streamer,
                            val_start_idx,
                            self.batch_test,
                            self.num_steps,
                            int(np.sqrt(self.N_x))
                            ** 2,  # Audio-only mode uses full N_x
                            plot_spectrograms=plot_spectrograms,
                            partition="val",  # Use validation data
                        )
                        if data_test is None:
                            print(f"No more validation audio data available")
                            break
                    elif self.image_streamer is not None:
                        # Image only streaming mode
                        from get_data import load_image_batch

                        val_start_idx = val_batch_idx * self.batch_test
                        data_test, labels_test = load_image_batch(
                            self.image_streamer,
                            val_start_idx,
                            self.batch_test,
                            self.num_steps,
                            int(np.sqrt(self.N_x))
                            ** 2,  # Image-only mode uses full N_x
                            partition="val",  # Use validation data
                        )
                        if data_test is None:
                            print(f"No more validation image data available")
                            break
                    else:
                        # Use pre-loaded data - for validation, use validation data or subset of test data
                        if hasattr(self, "data_val") and self.data_val is not None:
                            # Use dedicated validation data
                            data_test = self.data_val
                            labels_test = self.labels_val
                        elif self.data_test is not None:
                            # Fallback: use a portion of test data for validation
                            val_size = min(
                                len(self.data_test) // 5, 1000
                            )  # Use 20% or max 1000 samples
                            data_test = self.data_test[:val_size]
                            labels_test = self.labels_test[:val_size]
                        else:
                            print("No validation data available")
                            break

                    # Update T_test for this batch
                    # Ensure test data width matches expected N_x
                    if data_test is not None and data_test.shape[1] != self.N_x:
                        actual_nx_te = data_test.shape[1]
                        print(
                            f"Warning: Adjusting test input width from {actual_nx_te} to {self.N_x}"
                        )
                        if actual_nx_te < self.N_x:
                            pad_width_te = self.N_x - actual_nx_te
                            pad_block_te = np.zeros(
                                (data_test.shape[0], pad_width_te),
                                dtype=data_test.dtype,
                            )
                            data_test = np.concatenate(
                                [data_test, pad_block_te], axis=1
                            )
                        else:
                            data_test = data_test[:, : self.N_x]
                    T_test_batch = data_test.shape[0]

                    # Create test arrays directly
                    st = self.N_x  # stimulation
                    ex = st + self.N_exc  # excitatory
                    ih = ex + self.N_inh  # inhibitory

                    mp_test = np.zeros((T_test_batch, ih - st))
                    mp_test[0] = self.resting_potential

                    spikes_test = np.zeros((T_test_batch, self.N), dtype=np.int8)
                    spikes_test[:, :st] = data_test

                    # 3b) Test on the test set
                    # Ensure required state arrays exist even if not set earlier
                    if "spike_times" not in locals():
                        spike_times = np.zeros(self.N)
                    if "a" not in locals():
                        a = np.zeros(self.N)
                    if "I_syn" not in locals():
                        I_syn = np.zeros(self.N)
                    if "spike_threshold" not in locals() or spike_threshold is None:
                        try:
                            spike_threshold = self.spike_threshold.copy()
                        except Exception:
                            spike_threshold = np.zeros(self.N)
                    (
                        weights_te,
                        spikes_te_out,
                        mp_te,
                        *unused,
                        labels_te_out,
                        sleep_te_out,
                        I_syn_te,
                        spike_times_te,
                        a_te,
                    ) = train_network(
                        weights=self.weights.copy(),
                        spike_labels=labels_test.copy(),
                        mp=mp_test.copy(),
                        sleep=False,
                        train_weights=False,
                        T=T_test_batch,
                        mean_noise=mean_noise,
                        var_noise=var_noise,
                        spikes=spikes_test.copy(),
                        check_sleep_interval=check_sleep_interval,
                        timing_update=timing_update,
                        spike_times=spike_times.copy(),
                        a=a.copy(),
                        I_syn=I_syn.copy(),
                        spike_threshold=spike_threshold.copy(),
                        **common_args,
                    )

                    # Store results for accumulation (use pre-allocated arrays)
                    batch_size = spikes_te_out.shape[0]
                    all_spikes_val[val_sample_count : val_sample_count + batch_size] = (
                        spikes_te_out
                    )
                    all_labels_val[val_sample_count : val_sample_count + batch_size] = (
                        labels_te_out
                    )
                    all_mp_val[val_sample_count : val_sample_count + batch_size] = mp_te
                    val_sample_count += batch_size

                    # 4) Compute phi metrics using the trained outputs
                    phi_tr, phi_te, *unused = calculate_phi(
                        spikes_train=spikes_tr_out[:, self.st :],
                        spikes_test=spikes_te_out[:, self.st :],
                        labels_train=labels_tr_out,
                        labels_test=labels_te_out,
                        num_steps=self.num_steps,
                        pca_variance=self.pca_variance,
                        random_state=random_state,
                        num_classes=self.N_classes,
                    )

                    if accuracy_method == "top":
                        # calculate accuracy via top-responders heuristic
                        print(f"Debug - Test data shape: {spikes_te_out.shape}")
                        print(f"Debug - Labels shape: {labels_te_out.shape}")
                        print(
                            f"Debug - Labels range: {np.min(labels_te_out)} to {np.max(labels_te_out)}"
                        )
                        print(f"Debug - Unique labels: {np.unique(labels_te_out)}")
                        print(f"Debug - narrow_top: {narrow_top}")
                        print(f"Debug - num_classes: {self.N_classes}")

                        acc_te_batch = top_responders_plotted(
                            spikes=spikes_te_out[:, self.st : self.ih],
                            labels=labels_te_out,
                            num_classes=self.N_classes,
                            narrow_top=narrow_top,
                            smoothening=self.num_steps,
                            train=False,
                            compute_not_plot=True,
                            n_last_points=10000,
                        )
                        print(f"Debug - Raw validation accuracy: {acc_te_batch}")
                        # accumulate over all validation batches
                        val_acc += acc_te_batch
                        val_batches_counted += 1
                    val_phi += phi_te

                # average over all validation batches
                if accuracy_method == "top":
                    acc_te = val_acc / max(
                        1, val_batches_counted
                    )  # Keep acc_te for compatibility
                else:
                    acc_te = None
                phi_te = val_phi / total_num_vals

                # Enhanced validation distribution analysis (every 5 epochs or on first/last epoch)
                show_distributions = (e % 5 == 0) or (e == 0) or (e == self.epochs - 1)
                if (
                    (accuracy_method == "top" or accuracy_method == "pca_lr")
                    and spikes_te_out is not None
                    and labels_te_out is not None
                    and show_distributions
                ):
                    print(f"\n{'='*60}")
                    print(
                        f"VALIDATION DISTRIBUTIONS - Epoch {e+1}/{self.epochs} ({accuracy_method.upper()})"
                    )
                    print(f"{'='*60}")

                    # Show sample information for PCA+LR
                    if accuracy_method == "pca_lr":
                        min_samples_needed = max(10, self.N_classes * 5)
                        recommended_samples = self.N_classes * 20
                        train_samples = (
                            len(spikes_tr_out) // self.num_steps
                            if spikes_tr_out is not None
                            else 0
                        )
                        val_samples = (
                            len(spikes_te_out) // self.num_steps
                            if spikes_te_out is not None
                            else 0
                        )
                        print(
                            f"Sample Requirements: min {min_samples_needed}, recommended {recommended_samples}"
                        )
                        print(
                            f"Training samples: {train_samples}, Validation samples: {val_samples}"
                        )
                        if train_samples < min_samples_needed:
                            print(
                                f"⚠️  WARNING: Only {train_samples} training samples - PCA+LR training may be unreliable"
                            )
                        elif train_samples < recommended_samples:
                            print(
                                f"ℹ️  NOTE: {train_samples} training samples - more data would improve estimation"
                            )
                        print()

                    # Get predictions and true labels based on accuracy method
                    if accuracy_method == "top":
                        # Calculate prediction distribution using the same logic as top_responders_plotted
                        block_size = self.num_steps
                        num_blocks = spikes_te_out.shape[0] // block_size

                        if num_blocks > 0:
                            means = []
                            labs = []

                            for i in range(num_blocks):
                                block = spikes_te_out[
                                    i * block_size : (i + 1) * block_size
                                ]
                                block_mean = np.mean(block, axis=0)
                                means.append(block_mean)

                                block_lab = labels_te_out[
                                    i * block_size : (i + 1) * block_size
                                ]
                                block_maj = np.argmax(np.bincount(block_lab))
                                labs.append(block_maj)

                            spikes_agg = np.array(means)
                            labels_agg = np.array(labs)

                            # Get elite nodes (same as top_responders_plotted)
                            indices, _, _ = get_elite_nodes(
                                spikes=spikes_te_out,
                                labels=labels_te_out,
                                num_classes=self.N_classes,
                                narrow_top=narrow_top,
                            )

                            # Calculate activations
                            acts = np.zeros((spikes_agg.shape[0], self.N_classes))
                            for c in range(self.N_classes):
                                acts[:, c] = np.sum(
                                    spikes_agg[:, indices[:, c]], axis=1
                                )

                            # Get predictions
                            predictions = np.argmax(acts, axis=1)
                            true_labels = labels_agg
                        else:
                            print(
                                "Warning: Not enough blocks for top-responders analysis"
                            )
                            predictions = np.array([])
                            true_labels = np.array([])

                    elif accuracy_method == "pca_lr":
                        # Use PCA+LR predictions
                        try:
                            # Prepare features from training data for training the classifier
                            X_tr_dist, y_tr_dist = bin_spikes_by_label_no_breaks(
                                spikes=spikes_tr_out[:, self.st : self.ih],
                                labels=labels_tr_out,
                            )

                            # Prepare features from validation data for testing
                            X_te_dist, y_te_dist = bin_spikes_by_label_no_breaks(
                                spikes=spikes_te_out[:, self.st : self.ih],
                                labels=labels_te_out,
                            )

                            if X_tr_dist.size > 0 and X_te_dist.size > 0:
                                # Train PCA+LR on training data, test on validation data
                                accs, scaler, pca, clf = pca_logistic_regression(
                                    X_train=X_tr_dist,
                                    y_train=y_tr_dist,
                                    X_val=X_tr_dist,  # Use training data for validation during training
                                    y_val=y_tr_dist,
                                    X_test=X_te_dist,  # Test on validation data
                                    y_test=y_te_dist,
                                    variance_ratio=self.pca_variance,
                                )

                                # Get predictions from the trained classifier on validation data
                                X_te_p = pca.transform(scaler.transform(X_te_dist))
                                predictions = clf.predict(X_te_p)
                                true_labels = y_te_dist
                            else:
                                print(
                                    "Warning: No features available for PCA+LR prediction analysis"
                                )
                                predictions = np.array([])
                                true_labels = np.array([])

                        except Exception as ex:
                            print(f"Warning: PCA+LR prediction analysis failed ({ex})")
                            predictions = np.array([])
                            true_labels = np.array([])

                    # Display distributions if we have predictions
                    if len(predictions) > 0 and len(true_labels) > 0:
                        pred_dist = np.bincount(predictions, minlength=self.N_classes)
                        true_dist = np.bincount(true_labels, minlength=self.N_classes)

                        # Show what we're analyzing
                        analysis_type = (
                            "PCA+LR Features"
                            if accuracy_method == "pca_lr"
                            else "Spike Patterns"
                        )
                        print(f"Analysis based on: {analysis_type}")
                        print(f"Total samples analyzed: {len(true_labels)}")
                        print()

                        print(f"TRUE LABELS DISTRIBUTION:")
                        for i in range(self.N_classes):
                            count = true_dist[i]
                            percentage = (
                                (count / len(true_labels)) * 100
                                if len(true_labels) > 0
                                else 0
                            )
                            print(
                                f"  Class {i}: {count:4d} samples ({percentage:5.1f}%)"
                            )

                        print(f"\nPREDICTED DISTRIBUTION:")
                        for i in range(self.N_classes):
                            count = pred_dist[i]
                            percentage = (
                                (count / len(predictions)) * 100
                                if len(predictions) > 0
                                else 0
                            )
                            print(
                                f"  Class {i}: {count:4d} samples ({percentage:5.1f}%)"
                            )

                        # Calculate per-class accuracy
                        print(f"\nPER-CLASS ACCURACY:")
                        correct_per_class = np.zeros(self.N_classes)
                        total_per_class = np.zeros(self.N_classes)

                        for i in range(len(predictions)):
                            true_label = true_labels[i]
                            pred_label = predictions[i]
                            total_per_class[true_label] += 1
                            if true_label == pred_label:
                                correct_per_class[true_label] += 1

                        for i in range(self.N_classes):
                            if total_per_class[i] > 0:
                                class_acc = (
                                    correct_per_class[i] / total_per_class[i]
                                ) * 100
                                print(
                                    f"  Class {i}: {class_acc:5.1f}% ({int(correct_per_class[i])}/{int(total_per_class[i])} correct)"
                                )
                            else:
                                print(f"  Class {i}: No samples in validation batch")

                        # Overall accuracy summary
                        total_correct = np.sum(correct_per_class)
                        total_samples = np.sum(total_per_class)
                        overall_acc = (
                            (total_correct / total_samples) * 100
                            if total_samples > 0
                            else 0
                        )
                        print(
                            f"\nOVERALL ACCURACY: {overall_acc:.2f}% ({int(total_correct)}/{int(total_samples)} correct)"
                        )
                        print(f"{'='*60}\n")
                    else:
                        print(
                            "Warning: No predictions available for distribution analysis"
                        )
                        print(f"{'='*60}\n")

                # Use pre-allocated arrays (no concatenation needed)
                spikes_te_out = all_spikes_val[:val_sample_count]
                labels_te_out = all_labels_val[:val_sample_count]
                mp_te = all_mp_val[:val_sample_count]

                # If using PCA+LR, compute accuracy once using aggregated spikes
                if accuracy_method == "pca_lr":
                    try:
                        # Prepare features by binning contiguous label segments
                        X_tr, y_tr = bin_spikes_by_label_no_breaks(
                            spikes=spikes_tr_out[:, self.st : self.ih],
                            labels=labels_tr_out,
                        )
                        X_te, y_te = bin_spikes_by_label_no_breaks(
                            spikes=spikes_te_out[:, self.st : self.ih],
                            labels=labels_te_out,
                        )

                        if X_tr.size == 0 or X_te.size == 0:
                            print(
                                "Warning: empty features for PCA+LR; setting accuracy to 0.0"
                            )
                            acc_te = 0.0
                        else:
                            # Check sample requirements for PCA+LR
                            min_samples_needed = max(
                                10, self.N_classes * 5
                            )  # At least 5 samples per class
                            recommended_samples = (
                                self.N_classes * 20
                            )  # Recommended: 20 samples per class

                            print(
                                f"PCA+LR Validation: {len(X_te)} samples (min: {min_samples_needed}, recommended: {recommended_samples})"
                            )

                            if len(X_te) < min_samples_needed:
                                print(
                                    f"⚠️  WARNING: Only {len(X_te)} validation samples - PCA+LR may be unreliable"
                                )
                            elif len(X_te) < recommended_samples:
                                print(
                                    f"ℹ️  NOTE: {len(X_te)} validation samples - more data would improve estimation"
                                )

                            # Show validation data distribution
                            val_dist = np.bincount(y_te, minlength=self.N_classes)
                            print(f"\nValidation Data Distribution:")
                            for i in range(self.N_classes):
                                count = val_dist[i]
                                percentage = (
                                    (count / len(y_te)) * 100 if len(y_te) > 0 else 0
                                )
                                print(
                                    f"  Class {i}: {count:3d} samples ({percentage:5.1f}%)"
                                )
                            # Simple 80/20 split for validation
                            rng = np.random.RandomState(42)
                            idx = rng.permutation(X_tr.shape[0])
                            split = max(1, int(0.8 * len(idx)))
                            tr_idx, va_idx = idx[:split], idx[split:]
                            if va_idx.size == 0:
                                # ensure non-empty val
                                va_idx = tr_idx[-1:]
                                tr_idx = tr_idx[:-1]
                            accs, _, _, _ = pca_logistic_regression(
                                X_train=X_tr[tr_idx],
                                y_train=y_tr[tr_idx],
                                X_val=X_tr[va_idx],
                                y_val=y_tr[va_idx],
                                X_test=X_te,
                                y_test=y_te,
                                variance_ratio=self.pca_variance,
                            )
                            acc_te = float(accs.get("test", 0.0))
                    except Exception as ex:
                        print(f"Warning: PCA+LR accuracy failed ({ex}); using 0.0")
                        acc_te = 0.0

                # Update performance tracking
                self.performance_tracker[e] = [phi_te, acc_te]

                # Record validation metrics in parallel tracker
                if not hasattr(self, "val_performance_tracker"):
                    self.val_performance_tracker = np.zeros((self.epochs, 2))
                self.val_performance_tracker[e] = [phi_te, acc_te]

                # Early stopping disabled; keep full epoch loop

                # Plot t-SNE clustering after each training batch (if enabled and interval matches)
                if plot_tsne_during_training and (e + 1) % tsne_plot_interval == 0:
                    print(f"\n=== Epoch {e+1}/{self.epochs} - t-SNE Clustering ===")
                    print(f"Validation Accuracy: {acc_te:.3f}, Phi: {phi_te:.3f}")

                    # Plot t-SNE for training data (sample for performance)
                    if spikes_tr_out is not None and labels_tr_out is not None:
                        print("Plotting t-SNE for training data...")
                        # Sample data for faster t-SNE computation
                        sample_size = min(1000, len(spikes_tr_out))
                        sample_indices = np.random.choice(
                            len(spikes_tr_out), sample_size, replace=False
                        )
                        t_SNE(
                            spikes=spikes_tr_out[sample_indices, self.st : self.ih],
                            labels_spike=labels_tr_out[sample_indices],
                            n_components=2,
                            perplexity=min(30, sample_size // 4),  # Adaptive perplexity
                            max_iter=500,  # Reduced iterations for speed
                            random_state=random_state,
                            train=True,
                        )

                    # Plot t-SNE for validation data (sample for performance)
                    if spikes_te_out is not None and labels_te_out is not None:
                        print("Plotting t-SNE for validation data...")
                        # Sample data for faster t-SNE computation
                        sample_size = min(1000, len(spikes_te_out))
                        sample_indices = np.random.choice(
                            len(spikes_te_out), sample_size, replace=False
                        )
                        t_SNE(
                            spikes=spikes_te_out[sample_indices, self.st : self.ih],
                            labels_spike=labels_te_out[sample_indices],
                            n_components=2,
                            perplexity=min(30, sample_size // 4),  # Adaptive perplexity
                            max_iter=500,  # Reduced iterations for speed
                            random_state=random_state,
                            train=False,
                        )
                else:
                    # Just print the performance metrics without plotting
                    print(
                        f"Epoch {e+1}/{self.epochs} - Validation Accuracy: {acc_te:.3f}, Phi: {phi_te:.3f}"
                    )

                # early stopping
                if early_stopping and e > interval_ES:
                    start = max(0, e - interval_ES)

                if plot_spikes_train:
                    if start_time_spike_plot == None:
                        start_time_spike_plot = int(spikes_tr_out.shape[0] * 0.95)
                    if stop_time_spike_plot == None:
                        stop_time_spike_plot = spikes_tr_out.shape[0]

                    spike_plot(
                        spikes_tr_out[start_time_spike_plot:stop_time_spike_plot],
                        labels_tr_out[start_time_spike_plot:stop_time_spike_plot],
                    )

                # Rinse memory
                if e != self.epochs - 1:
                    # Clean up training data
                    del data_train, labels_train
                    # Clean up test data (these exist in the test loop)
                    del (
                        data_test,
                        labels_test,
                        mp_test,
                        weights_te,
                        spikes_te_out,
                        labels_te_out,
                        sleep_te_out,
                        I_syn_te,
                        spike_times_te,
                        a_te,
                    )
                    # Clean up training results
                    del mp_train, spikes_tr_out, labels_tr_out, sleep_tr_out
                    # Clean up accumulated arrays (use validation names since we evaluate on validation data)
                    del all_spikes_val, all_labels_val, all_mp_val
                    # Clean up temporary variables (validation accumulation variables)
                    del val_acc, val_phi
                    del T_test_batch, st, ex, ih
                    gc.collect()

                if plot_weights:
                    self.weights2plot_exc = w4p_exc_tr
                    self.weights2plot_inh = w4p_inh_tr
                    weights_plot(
                        weights_exc=self.weights2plot_exc,
                        weights_inh=self.weights2plot_inh,
                    )

                pbar.set_description(f"Epoch {e+1}/{self.epochs}")
                # Handle None valuTruees safely
                acc_str = f"{acc_te:.3f}" if acc_te is not None else "N/A"
                phi_str = f"{phi_te:.2f}" if phi_te is not None else "N/A"
                pbar.set_postfix(acc=acc_str, phi=phi_str)
                pbar.update(1)
            pbar.close()

            # Clean up main training loop variables
            del (
                I_syn,
                spike_times,
                a,
                spike_threshold,
            )
            del (common_args,)
            del interval_ES, download, data_dir
            gc.collect()

        # Final test pass even if early-stopped (evaluate on test partition)
        if not test_only and (
            self.audio_streamer is not None or self.image_streamer is not None
        ):
            try:
                # Reset test pointers
                if hasattr(self, "audio_streamer") and self.audio_streamer is not None:
                    self.audio_streamer.reset_partition("test")
                if hasattr(self, "image_streamer") and self.image_streamer is not None:
                    self.image_streamer.reset_partition("test")

                # Reuse test-only evaluator logic with partition="test"
                # Minimal: one full pass, compute both accs
                effective_batch = self.batch_test
                total_num_tests = max(1, self.all_test // effective_batch)
                max_test_samples = self.all_test * self.num_steps
                all_spikes_test = np.zeros((max_test_samples, self.N), dtype=np.int8)
                all_labels_test = np.zeros(max_test_samples, dtype=np.int32)
                all_mp_test = np.zeros(
                    (max_test_samples, self.N - self.N_x), dtype=np.float32
                )
                test_sample_count = 0
                acc_top_sum = 0.0

                for test_batch_idx in range(total_num_tests):
                    if (
                        self.audio_streamer is not None
                        and self.image_streamer is not None
                    ):
                        data_test, labels_test = self.load_multimodal_batch(
                            effective_batch,
                            is_training=False,
                            plot_spectrograms=False,
                            partition="test",
                        )
                        if data_test is None:
                            break
                    elif self.audio_streamer is not None:
                        from get_data import load_audio_batch

                        data_test, labels_test = load_audio_batch(
                            self.audio_streamer,
                            0,
                            effective_batch,
                            self.num_steps,
                            int(np.sqrt(self.N_x)) ** 2,
                            plot_spectrograms=False,
                            partition="test",
                        )
                        if data_test is None:
                            break
                    elif self.image_streamer is not None:
                        from get_data import load_image_batch

                        data_test, labels_test = load_image_batch(
                            self.image_streamer,
                            0,
                            effective_batch,
                            self.num_steps,
                            int(np.sqrt(self.N_x)) ** 2,
                            partition="test",
                        )
                        if data_test is None:
                            break
                    else:
                        break

                    if data_test.shape[1] != self.N_x:
                        actual = data_test.shape[1]
                        if actual < self.N_x:
                            pad = self.N_x - actual
                            data_test = np.concatenate(
                                [
                                    data_test,
                                    np.zeros(
                                        (data_test.shape[0], pad), dtype=data_test.dtype
                                    ),
                                ],
                                axis=1,
                            )
                        else:
                            data_test = data_test[:, : self.N_x]

                    T_te = data_test.shape[0]
                    st = self.N_x
                    ex = st + self.N_exc
                    ih = ex + self.N_inh
                    mp_test = np.zeros((T_te, ih - st))
                    mp_test[0] = self.resting_potential
                    spikes_test = np.zeros((T_te, self.N), dtype=np.int8)
                    spikes_test[:, :st] = data_test

                    (
                        _wte,
                        spikes_te_out,
                        mp_te,
                        *unused,
                        labels_te_out,
                        _sleep_te,
                        _Isyn_te,
                        _st_te,
                        _a_te,
                    ) = train_network(
                        weights=self.weights.copy(),
                        spike_labels=labels_test.copy(),
                        mp=mp_test.copy(),
                        sleep=False,
                        train_weights=False,
                        T=T_te,
                        mean_noise=0,
                        var_noise=0,
                        spikes=spikes_test.copy(),
                        check_sleep_interval=1000000,
                        timing_update=False,
                        spike_times=np.zeros_like(spike_times),
                        a=np.zeros_like(a),
                        I_syn=np.zeros_like(I_syn),
                        spike_threshold=spike_threshold.copy(),
                        **common_args,
                    )

                    bs = spikes_te_out.shape[0]
                    all_spikes_test[test_sample_count : test_sample_count + bs] = (
                        spikes_te_out
                    )
                    all_labels_test[test_sample_count : test_sample_count + bs] = (
                        labels_te_out
                    )
                    all_mp_test[test_sample_count : test_sample_count + bs] = mp_te
                    test_sample_count += bs

                    acc_top_sum += top_responders_plotted(
                        spikes=spikes_te_out[:, self.st : self.ih],
                        labels=labels_te_out,
                        num_classes=self.N_classes,
                        narrow_top=narrow_top,
                        smoothening=self.num_steps,
                        train=False,
                        compute_not_plot=True,
                        n_last_points=10000,
                    )

                # Slice outputs and store
                self.spikes_test = all_spikes_test[:test_sample_count]
                self.labels_test = all_labels_test[:test_sample_count]
                self.mp_test = all_mp_test[:test_sample_count]
                final_acc = acc_top_sum / max(1, total_num_tests)
                print(
                    f"Final test (after training/early stop) — Top responders acc: {final_acc:.4f}"
                )

                # Clean up final test arrays and variables
                del all_spikes_test, all_labels_test, all_mp_test
                del test_sample_count, acc_top_sum

            except Exception as e:
                print(f"Warning: final test pass skipped ({e})")

        if save_model and not self.model_loaded:
            if spikes_tr_out is None:
                print(
                    "Warning: Training outputs are unavailable (no data or early exit); skipping save."
                )
            else:
                self.spikes_train = spikes_tr_out
                self.spikes_test = spikes_te_out
                self.mp_train = mp_tr
                self.mp_test = mp_te
                self.weights2plot_exc = w4p_exc_tr
                self.weights2plot_inh = w4p_inh_tr
                self.spike_threshold = thresh_tr
                self.max_weight_sum_inh = mx_w_inh_tr
                self.max_weight_sum_exc = mx_w_exc_tr
            # Persist model to disk for reuse in future runs
            try:
                self.process(
                    save_model=True,
                    model_parameters=self.model_parameters,
                )
            except Exception as e:
                print(f"Warning: model save failed ({e})")
            self.labels_train = labels_tr_out
            self.labels_test = labels_te_out

            # save training results
            model_dir = self.process(
                save_model=True,
                model_parameters=self.model_parameters,
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

        # Always expose latest train outputs for downstream analysis (phi/accuracy)
        try:
            if (
                'spikes_tr_out' in locals()
                and spikes_tr_out is not None
                and 'labels_tr_out' in locals()
                and labels_tr_out is not None
            ):
                self.spikes_train = spikes_tr_out
                self.labels_train = labels_tr_out
                self.mp_train = mp_tr if 'mp_tr' in locals() else getattr(self, 'mp_train', None)
        except Exception:
            pass

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

        if compare_decay_rates:
            # try loading previous run
            self.process(
                sleep_scores=weight_decay_rate_exc,
                model_dir_=model_dir,
                load_phi_model=True,
            )

            # retrain if phi model not loaded
            if not self.loaded_phi_model:
                # Pre-zip the decay rates (must be same length)
                decay_pairs = list(zip(weight_decay_rate_exc, weight_decay_rate_inh))
                # Allocate array: [n_decay_rates, n_samples, 8 metrics]
                self.phi_all_scores = np.zeros((len(decay_pairs), samples, 9))
                # Define data parameters
                data_parameters = {"pixel_size": int(np.sqrt(self.N_x)), "train_": True}

                # Ensure data/mdata directory exists
                if not os.path.exists("data/mdata"):
                    os.makedirs("data/mdata", exist_ok=True)

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

                # Main loop: over samples and decay-rate settings
                with tqdm(
                    total=len(decay_pairs) * samples, desc="Computing φ scores"
                ) as pbar:
                    # Clean up variables that are no longer needed
                    del data_parameters, folders, folder, json_file_path, ex_params
                    gc.collect()
                    for t in range(samples):
                        # 1) Generate fresh data for this sample
                        (
                            data_train,
                            labels_train,
                            data_test,
                            labels_test,
                        ) = create_data(
                            pixel_size=int(np.sqrt(self.N_x)),
                            num_steps=self.num_steps,
                            plot_comparison=False,
                            gain=self.gain,
                            gain_labels=self.gain_labels,
                            train_=True,
                            offset=self.offset,
                            download=False,
                            data_dir=data_dir,
                            true_labels=False,
                            N_classes=self.N_classes,
                            first_spike_time=self.first_spike_time,
                            time_var_input=self.time_var_input,
                            num_images=self.num_images,
                            add_breaks=self.add_breaks,
                            break_lengths=self.break_lengths,
                            noisy_data=self.noisy_data,
                            noise_level=self.noise_level,
                            classes=self.classes,
                            test_data_ratio=self.test_data_ratio,
                        )

                        # 2) Convert raw data into simulation arrays
                        (
                            mp_train,
                            mp_test,
                            spikes_train_init,
                            spikes_test_init,
                        ) = create_arrays(
                            N=self.N,
                            N_exc=self.N_exc,
                            N_inh=self.N_inh,
                            resting_membrane=self.resting_potential,
                            total_time_train=self.T_train,
                            total_time_test=self.T_test,
                            data_train=data_train,
                            data_test=data_test,
                            N_classes=self.N_classes,
                            N_x=self.N_x,
                        )

                        # 3) Loop over each pair of decay rates
                        for r, (decay_exc, decay_inh) in enumerate(decay_pairs):
                            # Bundle common training arguments
                            common_args = dict(
                                N_classes=self.N_classes,
                                supervised=self.supervised,
                                unsupervised=self.unsupervised,
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
                                N_exc=self.N_exc,
                                N_inh=self.N_inh,
                                beta=beta,
                                num_exc=num_exc,
                                num_inh=num_inh,
                                weight_decay=weight_decay,
                                weight_decay_rate_exc=decay_exc,
                                weight_decay_rate_inh=decay_inh,
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
                                noisy_potential=noisy_potential,
                                noisy_weights=noisy_weights,
                                weight_mean_noise=weight_mean_noise,
                                weight_var_noise=weight_var_noise,
                                vectorized_trace=vectorized_trace,
                                N_x=self.N_x,
                            )

                            # 3a) Train on the training set
                            (
                                weights_tr,
                                spikes_tr_out,
                                *unused,
                                labels_tr_out,
                                sleep_tr_out,
                            ) = train_network(
                                weights=self.weights.copy(),
                                spike_labels=labels_train.copy(),
                                mp=mp_train.copy(),
                                sleep=sleep,
                                train_weights=train_weights,
                                T=self.T_train,
                                mean_noise=mean_noise,
                                var_noise=var_noise,
                                spikes=spikes_train_init.copy(),
                                check_sleep_interval=check_sleep_interval,
                                timing_update=timing_update,
                                spike_times=spike_times.copy(),
                                final=False,
                                **common_args,
                            )

                            # Clean up unused variables
                            del unused

                            # 3b) Test on the test set
                            (
                                weights_te,
                                spikes_te_out,
                                *unused,
                                labels_te_out,
                                sleep_te_out,
                            ) = train_network(
                                weights=weights_tr.copy(),
                                spike_labels=labels_test.copy(),
                                mp=mp_test.copy(),
                                sleep=False,
                                train_weights=False,
                                T=self.T_test,
                                mean_noise=mean_noise,
                                var_noise=var_noise,
                                spikes=spikes_test_init.copy(),
                                check_sleep_interval=check_sleep_interval,
                                timing_update=timing_update,
                                spike_times=spike_times.copy(),
                                final=False,
                                **common_args,
                            )

                            # Clean up unused variables
                            del unused

                            # 4) Compute phi metrics using the trained outputs
                            phi_tr, phi_te, wcss_tr, wcss_te, bcss_tr, bcss_te = (
                                calculate_phi(
                                    spikes_train=spikes_tr_out[:, self.st :],
                                    spikes_test=spikes_te_out[:, self.st :],
                                    labels_train=labels_tr_out,
                                    labels_test=labels_te_out,
                                    num_steps=self.num_steps,
                                    pca_variance=self.pca_variance,
                                    random_state=random_state,
                                    num_classes=self.N_classes,
                                )
                            )

                            # calculate accuracy
                            acc_te = top_responders_plotted(
                                spikes=spikes_te_out,
                                labels=labels_te_out,
                                ih=self.ih,
                                st=self.st,
                                num_classes=self.N_classes,
                                narrow_top=narrow_top,
                                smoothening=self.num_steps,
                                train=False,
                                compute_not_plot=True,
                            )

                            # Store acc
                            # 5) Store results and update progress bar
                            self.phi_all_scores[r, t] = [
                                phi_tr,
                                phi_te,
                                decay_exc,
                                sleep_tr_out,
                                wcss_tr,
                                wcss_te,
                                bcss_tr,
                                bcss_te,
                                acc_te,
                            ]

                            pbar.update(1)

                        # 6) Clean up per-sample data to free memory
                        del data_train, labels_train, data_test, labels_test
                        del mp_train, mp_test
                        del spikes_train_init, spikes_test_init
                        # Clean up training and test results from decay rate loop
                        del weights_tr, spikes_tr_out, labels_tr_out, sleep_tr_out
                        del weights_te, spikes_te_out, labels_te_out, sleep_te_out
                        del phi_tr, phi_te, wcss_tr, wcss_te, bcss_tr, bcss_te, acc_te
                        # Clean up common_args for this decay rate iteration
                        del common_args
                        gc.collect()

                # save phi scores, sleep lengths and amounts
                self.process(
                    save_phi_model=True,
                    model_dir_=model_dir,
                    sleep_scores=weight_decay_rate_exc,
                )

                # Clean up compare_decay_rates variables
                del decay_pairs, pbar
                gc.collect()

            # plot phi and sleep amounts with linear regression
            plot_phi_acc(
                all_scores=self.phi_all_scores,
            )

    def analyze_results(
        self,
        perplexity=8,
        max_iter=1000,
        random_state=48,
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
                random_state=random_state,
                train=True,
            )
        if t_sne_test:
            t_SNE(
                spikes=self.spikes_test[:, self.st : self.ex],
                labels_spike=self.labels_test,
                n_components=n_components,
                perplexity=perplexity,
                max_iter=max_iter,
                random_state=random_state,
                train=False,
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
                random_state=random_state,
            )
        if pca_test:
            PCA_analysis(
                spikes=self.spikes_test[:, self.N_x :],
                labels_spike=self.labels_test,
                n_components=n_components,
                random_state=random_state,
            )
        if calculate_phi_:
            if hasattr(self, "spikes_train") and self.spikes_train is not None:
                calculate_phi(
                    spikes_train=self.spikes_train,
                    spikes_test=self.spikes_test,
                    labels_train=self.labels_train,
                    labels_test=self.labels_test,
                    num_steps=self.num_steps,
                    pca_variance=self.pca_variance,
                    random_state=random_state,
                    num_classes=self.N_classes,
                )
            else:
                print("Skipping phi calculation: no training data (test-only run).")
        # Optional: PCA+LR end-to-end analysis only if training data exists
        if (
            hasattr(self, "spikes_train")
            and self.spikes_train is not None
            and hasattr(self, "labels_train")
            and self.labels_train is not None
        ):
            try:
                from pca_linear_classifier import pca_logistic_regression as _pca_lr

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
                    accs, _, _, _ = _pca_lr(
                        X_train=X_tr[tr_idx],
                        y_train=y_tr[tr_idx],
                        X_val=X_tr[va_idx],
                        y_val=y_tr[va_idx],
                        X_test=X_te,
                        y_test=y_te,
                        variance_ratio=self.pca_variance,
                    )
                    print(f"PCA+LR accuracy: {accs}")
            except Exception as ex:
                print(f"PCA+LR analysis skipped: {ex}")

    def sweep_sleep_rates(
        self,
        sleep_rates,
        train_params=None,
        reinit_network_each_rate=True,
    ):
        """Run training for multiple sleep decay rates λ and return accuracy and phi for each.

        Requirements: call prepare_data(...) before this so that data/streamers are ready.

        Args:
            sleep_rates: iterable of λ values to test (used for both exc/inh).
            train_params: optional dict to override training params (e.g., epochs config in prepare_data).
            reinit_network_each_rate: if True, re-create weights before each run.

        Returns:
            List of dicts with keys: {'lambda', 'accuracy_top', 'phi_test'}
        """
        results = []

        # Default training overrides (kept minimal; inherit class defaults otherwise)
        base_train_kwargs = dict(
            train_weights=True,
            save_model=False,
            plot_epoch_performance=False,
            compare_decay_rates=False,
            accuracy_method="top",
        )
        if isinstance(train_params, dict):
            base_train_kwargs.update(train_params)

        for lam in sleep_rates:
            # Reinitialize network weights if requested
            if reinit_network_each_rate:
                try:
                    self.prepare_network(create_network=False)
                except Exception as e:
                    print(f"Warning: network re-init failed ({e}); continuing with current weights")

            # Train with given sleep rate (used for both exc/inh)
            self.train_network(
                weight_decay_rate_exc=[lam],
                weight_decay_rate_inh=[lam],
                **base_train_kwargs,
            )

            # Ensure test outputs exist for accuracy; if not, attempt a lightweight test-only pass
            if not hasattr(self, 'spikes_test') or self.spikes_test is None:
                try:
                    self.train_network(
                        test_only=True,
                        train_weights=False,
                        weight_decay_rate_exc=[lam],
                        weight_decay_rate_inh=[lam],
                        plot_epoch_performance=False,
                        compare_decay_rates=False,
                    )
                except Exception as e:
                    print(f"Warning: test-only evaluation failed ({e})")

            # Compute top-responders accuracy on available test spikes
            acc_top = None
            try:
                if (
                    hasattr(self, 'spikes_test')
                    and self.spikes_test is not None
                    and hasattr(self, 'labels_test')
                    and self.labels_test is not None
                ):
                    acc_top = top_responders_plotted(
                        spikes=self.spikes_test[:, self.st:self.ih],
                        labels=self.labels_test,
                        num_classes=self.N_classes,
                        narrow_top=getattr(self, 'narrow_top', 0.2),
                        smoothening=getattr(self, 'num_steps', 1000),
                        train=False,
                        compute_not_plot=True,
                        n_last_points=10000,
                    )
            except Exception as e:
                print(f"Warning: accuracy computation failed ({e})")

            # Compute phi_test using current train/test spikes
            phi_test = None
            try:
                if (
                    hasattr(self, 'spikes_train')
                    and self.spikes_train is not None
                    and hasattr(self, 'spikes_test')
                    and self.spikes_test is not None
                    and hasattr(self, 'labels_train')
                    and self.labels_train is not None
                    and hasattr(self, 'labels_test')
                    and self.labels_test is not None
                ):
                    _, phi_test, *_ = calculate_phi(
                        spikes_train=self.spikes_train[:, self.st:],
                        spikes_test=self.spikes_test[:, self.st:],
                        labels_train=self.labels_train,
                        labels_test=self.labels_test,
                        num_steps=getattr(self, 'num_steps', 1000),
                        pca_variance=getattr(self, 'pca_variance', 0.95),
                        random_state=48,
                        num_classes=self.N_classes,
                    )
            except Exception as e:
                print(f"Warning: phi computation failed ({e})")

            results.append({
                'lambda': float(lam),
                'accuracy_top': float(acc_top) if acc_top is not None else None,
                'phi_test': float(phi_test) if phi_test is not None else None,
            })

        # Nicely print results
        try:
            print("\nSleep-rate sweep results (λ → accuracy_top, phi_test):")
            for r in results:
                print(f"  λ={r['lambda']}: acc_top={r['accuracy_top']}, phi_test={r['phi_test']}")
        except Exception:
            pass

        return results
