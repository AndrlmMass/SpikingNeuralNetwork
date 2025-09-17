import numpy as np
import os
import gc
from tqdm import tqdm
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
    top_responders_plotted,
    plot_phi_acc,
    plot_epoch_training,
    plot_audio_spectrograms_and_spikes,
    plot_audio_spectrograms_and_spikes_simple,
)
from analysis import t_SNE, PCA_analysis, calculate_phi
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
                        self.mp_train = np.load(
                            os.path.join("model", folder, "mp_train.npy")
                        )
                        self.weights2plot_exc = np.load(
                            os.path.join("model", folder, "weights2plot_exc.npy")
                        )
                        self.weights2plot_inh = np.load(
                            os.path.join("model", folder, "weights2plot_inh.npy")
                        )
                        self.spike_threshold = np.load(
                            os.path.join("model", folder, "spike_threshold.npy")
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
                        self.spikes_test = np.load(
                            os.path.join("model", folder, "spikes_test.npy")
                        )
                        self.performance_tracker = np.load(
                            os.path.join("model", folder, "performance_tracker.npy")
                        )

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
        tot_images_train=10000,
        tot_images_test=10000,
        single_train=1000,
        single_test=166,
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
        # New batch and total parameters
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
        self.tot_images_train = tot_images_train
        self.tot_images_test = tot_images_test
        self.single_train = single_train
        self.single_test = single_test
        self.epochs = tot_images_train // single_train
        self.add_breaks = add_breaks
        self.break_lengths = break_lengths
        self.noisy_data = noisy_data
        self.noise_level = noise_level
        self.test_data_ratio = test_data_ratio
        self.val_split = val_split
        self.train_split = train_split
        self.test_split = test_split
        self.T_train = self.single_train * num_steps
        self.T_test = self.single_test * num_steps
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
            self.single_train = min(batch_audio_train, batch_image_train)
            self.single_test = min(batch_audio_test, batch_image_test)

            # Use smaller total for both to ensure synchronization
            self.tot_images_train = min(all_audio_train, all_images_train)
            self.tot_images_test = min(all_audio_test, all_images_test)
            self.epochs = self.tot_images_train // self.single_train

            # Double input neurons for multimodal
            self.N_x = int(2 * self.N_x)

            # Update network architecture based on new N_x
            self.st = self.N_x  # stimulation
            self.ex = self.st + self.N_exc  # excitatory
            self.ih = self.ex + self.N_inh  # inhibitory
            self.N = self.N_exc + self.N_inh + self.N_x  # total neurons

            print(f"Multimodal mode: {self.N_x} input neurons (image + audio)")
            print(
                f"Network: {self.N} total neurons (stim: {self.st}, exc: {self.ex}, inh: {self.ih})"
            )
            print(f"Batch size: {self.single_train} train, {self.single_test} test")
            print(
                f"Total samples: {self.tot_images_train} train, {self.tot_images_test} test"
            )

        elif audioMNIST:
            # Audio only mode
            total_audio = all_audio_train + all_audio_test + all_audio_val
            if total_audio > 30000:
                raise ValueError(
                    f"Total audio samples ({total_audio}) cannot exceed 30000"
                )

            self.single_train = batch_audio_train
            self.single_test = batch_audio_test
            self.tot_images_train = all_audio_train
            self.tot_images_test = all_audio_test
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

            self.single_train = batch_image_train
            self.single_test = batch_image_test
            self.tot_images_train = all_images_train
            self.tot_images_test = all_images_test
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
                data_path, batch_size=batch_audio_train
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
                    batch_size=max(100, getattr(self, "single_train", 500)),
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
                        expected_params = {
                            "pixel_size": int(np.sqrt(self.N_x // 2)),
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
                image_params = {
                    "pixel_size": int(np.sqrt(self.N_x // 2)),
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

            self.image_streamer = ImageDataStreamer(
                "data",  # Use the data directory for MNIST download
                batch_size=batch_image_train,
                pixel_size=int(np.sqrt(self.N_x // 2)),
                num_steps=num_steps,
                gain=gain,
                offset=offset,
                first_spike_time=first_spike_time,
                time_var_input=time_var_input,
            )
            print(
                f"Image streamer initialized with {self.image_streamer.get_total_samples()} total samples"
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
                        num_images_train=self.single_train,
                        num_images_test=self.single_test,
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
                        num_images_train=self.single_train,
                        num_images_test=self.single_test,
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

    def load_audio_batch(self, batch_size, is_training=True, plot_spectrograms=False):
        """
        Load a batch of audio data and convert to spikes on-demand.

        Args:
            batch_size: Number of samples to load
            is_training: If True, load training data; if False, load test data

        Returns:
            (spike_data, labels) or (None, None) if no more data
        """
        if self.audio_streamer is None:
            return None, None

        # Check if we've exceeded the total limit
        if is_training and self.current_train_idx >= self.tot_images_train:
            return None, None
        if not is_training and self.current_test_idx >= self.tot_images_test:
            return None, None

        # Import the load_audio_batch function
        from get_data import load_audio_batch

        # Determine current index
        if is_training:
            start_idx = self.current_train_idx
        else:
            start_idx = self.current_test_idx

        # Load audio batch - determine correct number of neurons based on mode
        training_mode = self.get_training_mode()
        if training_mode == "multimodal":
            # Multimodal mode - use half of N_x for audio
            num_audio_neurons = int(np.sqrt(self.N_x // 2)) ** 2
        else:
            # Audio-only mode - use full N_x for audio
            num_audio_neurons = int(np.sqrt(self.N_x)) ** 2

        # print(f"Audio neurons: {num_audio_neurons}")  # Commented for performance
        spike_data, labels = load_audio_batch(
            self.audio_streamer,
            start_idx,
            batch_size,
            self.num_steps,
            num_audio_neurons,
            plot_spectrograms=plot_spectrograms,
        )

        if spike_data is not None:
            # Update index
            if is_training:
                self.current_train_idx += batch_size
            else:
                self.current_test_idx += batch_size

        return spike_data, labels

    def load_multimodal_batch(
        self, batch_size, is_training=True, plot_spectrograms=False
    ):
        """
        Load a batch of multimodal data (image + audio) and concatenate.

        Args:
            batch_size: Number of samples to load
            is_training: If True, load training data; if False, load test data

        Returns:
            (concatenated_spike_data, labels) or (None, None) if no more data
        """
        from get_data import load_image_batch

        # Load audio batch
        audio_spikes, audio_labels = self.load_audio_batch(
            batch_size, is_training, plot_spectrograms
        )
        if audio_spikes is None:
            return None, None

        # Load image batch - determine correct number of neurons based on mode
        training_mode = self.get_training_mode()
        if training_mode == "multimodal":
            # Multimodal mode - use half of N_x for image
            num_image_neurons = int(np.sqrt(self.N_x // 2)) ** 2
        else:
            # Image-only mode - use full N_x for image
            num_image_neurons = int(np.sqrt(self.N_x)) ** 2

        # print(f"Image neurons: {num_image_neurons}")  # Commented for performance
        image_spikes, image_labels = load_image_batch(
            self.image_streamer,
            self.current_train_idx if is_training else self.current_test_idx,
            batch_size,
            self.num_steps,
            num_image_neurons,
        )
        if image_spikes is None:
            return None, None

        # Concatenate horizontally: [image_features, audio_features]
        multimodal_spikes = np.concatenate([image_spikes, audio_spikes], axis=1)

        return multimodal_spikes, audio_labels

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
            "single_train": self.single_train,
            "single_test": self.single_test,
            "tot_images_train": self.tot_images_train,
            "tot_images_test": self.tot_images_test,
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

        self.model_parameters["tot_images_train"] = self.tot_images_train
        self.model_parameters["tot_images_test"] = self.tot_images_test
        self.model_parameters["single_train"] = self.single_train
        self.model_parameters["single_test"] = self.single_test
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

        if not force_train and self.data_loaded:
            model_dir = self.process(
                load_model=True,
                model_parameters=self.model_parameters,
            )

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
            self.performance_tracker = np.zeros((self.epochs, 2))

            # early stopping interval
            interval_ES = int(self.epochs * 0.1)  # 10% interval rate

            # define progress bar
            pbar = tqdm(
                total=self.epochs,
                desc=f"Epoch 0/{self.epochs}:",
                unit="it",
                ncols=80,
                bar_format="{desc} [{bar}] ETA: {remaining} |{postfix}",
            )
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

            # Define indices for each dataset
            idx_test = 0
            idx_train = 0
            idx_val = 0

            # loop over self.epochs
            for e in range(self.epochs):

                # Reset test index at the beginning of each epoch
                self.current_test_idx = 0

                # Load data for this epoch
                if (
                    self.audio_streamer is not None
                    and hasattr(self, "image_streamer")
                    and self.image_streamer is not None
                ):
                    # Multimodal mode - load both audio and image data
                    data_train, labels_train = self.load_multimodal_batch(
                        self.single_train,
                        is_training=True,
                        plot_spectrograms=plot_spectrograms,
                    )
                    if data_train is None:
                        print(f"No more multimodal data available at epoch {e}")
                        break
                elif self.audio_streamer is not None:
                    # Audio only mode
                    data_train, labels_train = self.load_audio_batch(
                        self.single_train,
                        is_training=True,
                        plot_spectrograms=plot_spectrograms,
                    )
                    if data_train is None:
                        print(f"No more audio data available at epoch {e}")
                        break
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
                    weights=self.weights,
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
                total_num_tests = self.tot_images_test // self.single_test
                test_acc = 0
                test_phi = 0

                # Initialize arrays to store accumulated test results
                all_spikes_test = []
                all_labels_test = []
                all_mp_test = []

                # Pre-allocate arrays for better memory management
                max_test_samples = self.tot_images_test * self.num_steps
                all_spikes_test = np.zeros((max_test_samples, self.N), dtype=np.int8)
                all_labels_test = np.zeros(max_test_samples, dtype=np.int32)
                all_mp_test = np.zeros(
                    (max_test_samples, self.N - self.N_x), dtype=np.float32
                )
                test_sample_count = 0

                for test_batch_idx in range(total_num_tests):
                    # Load test data
                    if (
                        self.audio_streamer is not None
                        and self.image_streamer is not None
                    ):
                        # Multimodal streaming mode
                        data_test, labels_test = self.load_multimodal_batch(
                            self.single_test,
                            is_training=False,
                            plot_spectrograms=plot_spectrograms,
                        )
                        if data_test is None:
                            print(f"No more test multimodal data available")
                            break
                    elif self.audio_streamer is not None:
                        # Audio only streaming mode
                        from get_data import load_audio_batch

                        test_start_idx = test_batch_idx * self.single_test
                        data_test, labels_test = load_audio_batch(
                            self.audio_streamer,
                            test_start_idx,
                            self.single_test,
                            self.num_steps,
                            int(np.sqrt(self.N_x))
                            ** 2,  # Audio-only mode uses full N_x
                            plot_spectrograms=plot_spectrograms,
                        )
                        if data_test is None:
                            print(f"No more test audio data available")
                            break
                    elif self.image_streamer is not None:
                        # Image only streaming mode
                        from get_data import load_image_batch

                        test_start_idx = test_batch_idx * self.single_test
                        data_test, labels_test = load_image_batch(
                            self.image_streamer,
                            test_start_idx,
                            self.single_test,
                            self.num_steps,
                            int(np.sqrt(self.N_x))
                            ** 2,  # Image-only mode uses full N_x
                        )
                        if data_test is None:
                            print(f"No more test image data available")
                            break
                    else:
                        # Use pre-loaded data (for image data)
                        if self.data_test is not None:
                            data_test = self.data_test
                            labels_test = self.labels_test
                        else:
                            print("No test data available")
                            break

                    # Update T_test for this batch
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
                    all_spikes_test[
                        test_sample_count : test_sample_count + batch_size
                    ] = spikes_te_out
                    all_labels_test[
                        test_sample_count : test_sample_count + batch_size
                    ] = labels_te_out
                    all_mp_test[test_sample_count : test_sample_count + batch_size] = (
                        mp_te
                    )
                    test_sample_count += batch_size

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

                    # calculate accuracy
                    print(f"Debug - Test data shape: {spikes_te_out.shape}")
                    print(f"Debug - Labels shape: {labels_te_out.shape}")
                    print(
                        f"Debug - Labels range: {np.min(labels_te_out)} to {np.max(labels_te_out)}"
                    )
                    print(f"Debug - Unique labels: {np.unique(labels_te_out)}")
                    print(f"Debug - narrow_top: {narrow_top}")
                    print(f"Debug - num_classes: {self.N_classes}")

                    acc_te = top_responders_plotted(
                        spikes=spikes_te_out[:, self.st : self.ih],
                        labels=labels_te_out,
                        num_classes=self.N_classes,
                        narrow_top=narrow_top,
                        smoothening=self.num_steps,
                        train=False,
                        compute_not_plot=True,
                        n_last_points=10000,
                    )
                    print(f"Debug - Raw accuracy: {acc_te}")
                    # accumulate over all tests
                    test_acc += acc_te
                    test_phi += phi_te

                    idx_test += self.single_test

                # average over all tests
                acc_te = test_acc / total_num_tests
                phi_te = test_phi / total_num_tests

                # Use pre-allocated arrays (no concatenation needed)
                spikes_te_out = all_spikes_test[:test_sample_count]
                labels_te_out = all_labels_test[:test_sample_count]
                mp_te = all_mp_test[:test_sample_count]

                # Update performance tracking
                self.performance_tracker[e] = [
                    phi_te,
                    acc_te,
                ]

                # Plot t-SNE clustering after each training batch (if enabled and interval matches)
                if plot_tsne_during_training and (e + 1) % tsne_plot_interval == 0:
                    print(f"\n=== Epoch {e+1}/{self.epochs} - t-SNE Clustering ===")
                    print(f"Accuracy: {acc_te:.3f}, Phi: {phi_te:.3f}")

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

                    # Plot t-SNE for test data (sample for performance)
                    if spikes_te_out is not None and labels_te_out is not None:
                        print("Plotting t-SNE for test data...")
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
                        f"Epoch {e+1}/{self.epochs} - Accuracy: {acc_te:.3f}, Phi: {phi_te:.3f}"
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
                    # Clean up accumulated arrays
                    del all_spikes_test, all_labels_test, all_mp_test
                    # Clean up temporary variables
                    del test_acc, test_phi
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
            del I_syn, spike_times, a, spike_threshold, acc_te, phi_te, phi_tr
            del common_args, total_num_tests, idx_train, idx_test
            del interval_ES, download, data_dir
            gc.collect()

        if save_model and not self.model_loaded:
            self.spikes_train = spikes_tr_out
            self.spikes_test = spikes_te_out
            self.mp_train = mp_tr
            self.mp_test = mp_te
            self.weights2plot_exc = w4p_exc_tr
            self.weights2plot_inh = w4p_inh_tr
            self.spike_threshold = thresh_tr
            self.max_weight_sum_inh = mx_w_inh_tr
            self.max_weight_sum_exc = mx_w_exc_tr
            self.labels_train = labels_tr_out
            self.labels_test = labels_te_out

            # save training results
            model_dir = self.process(
                save_model=True,
                model_parameters=self.model_parameters,
            )

        if plot_epoch_performance:
            plot_epoch_training(
                self.performance_tracker[:, 1], self.performance_tracker[:, 0]
            )

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
        if t_sne_train:
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
        if calculate_phi_:
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
