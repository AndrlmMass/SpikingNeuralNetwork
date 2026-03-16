import numpy as np
import os
import gc
from tqdm import tqdm
import cProfile
import json
import pstats
import time
from datetime import datetime
from train import train_network
from get_data import (
    create_data,
    GeomfigDataStreamer,
)
from plot import (
    spike_plot,
    heat_map,
    WTA_accuracy,
    gif_spike_rate_by_label,
)
from analysis import (
    calculate_phi,
    bin_spikes_by_label_no_breaks,
)
from create_network import create_weights, create_arrays
import matplotlib.pyplot as plt


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
        self.random_state = random_state
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

    def _ensure_acc_logger(self):
        if self._acc_log_file is None:
            from datetime import datetime

            self.image_dataset = getattr(self, "image_dataset", "unknown")
            date = datetime.now().strftime("%Y.%m.%d")
            try:
                self._acc_log_dir = os.path.join(
                    "results",
                    "acc_history",
                    f"{self.image_dataset}",
                    f"{date}",
                    f"acc_{self.ts_spec}",
                )
                os.makedirs(self._acc_log_dir, exist_ok=True)
            except Exception:
                pass
            self._acc_log_file = os.path.join(
                self._acc_log_dir, f"acc_{self.ts_spec}.jsonl"
            )

    def _record_accuracy(
        self, split: str, value, epoch: int | None = None, method: str | None = None
    ):
        try:
            acc_val = float(value) if value is not None else None
        except Exception:
            acc_val = None
        # Append to in-memory history if numeric
        try:
            if acc_val is not None:
                # Use method as key, or "default" if no method specified (for backward compatibility)
                method_key = method if method is not None else "default"
                if split not in self.acc_history:
                    self.acc_history[split] = {}
                if method_key not in self.acc_history[split]:
                    self.acc_history[split][method_key] = []
                self.acc_history[split][method_key].append(acc_val)
        except Exception:
            pass
        # Persist incrementally
        try:
            self._ensure_acc_logger()
            rec = {
                "timestamp": datetime.now().isoformat(),
                "split": str(split),
                "epoch": int(epoch) if epoch is not None else None,
                "accuracy": acc_val,
            }
            if method is not None:
                rec["method"] = method
            with open(self._acc_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
        except Exception as e:
            # Non-fatal: continue training even if logging fails
            print(f"Warning: failed to persist accuracy record ({e})")

    def _record_phi(self, split: str, phi_value, epoch: int | None = None):
        try:
            phi_val = float(phi_value) if phi_value is not None else None
        except Exception:
            phi_val = None
        # Persist to JSONL file
        try:
            self._ensure_acc_logger()
            rec = {
                "timestamp": datetime.now().isoformat(),
                "split": str(split),
                "epoch": int(epoch) if epoch is not None else None,
                "phi": phi_val,
            }
            with open(self._acc_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
        except Exception as e:
            # Non-fatal: continue even if logging fails
            print(f"Warning: failed to persist phi record ({e})")

    def _record_mcc(self, split: str, mcc_value, epoch: int | None = None):
        try:
            mcc_val = float(mcc_value) if mcc_value is not None else None
        except Exception:
            mcc_val = None
        # Persist to JSONL file
        try:
            self._ensure_acc_logger()
            rec = {
                "timestamp": datetime.now().isoformat(),
                "split": str(split),
                "epoch": int(epoch) if epoch is not None else None,
                "mcc": mcc_val,
            }
            with open(self._acc_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
        except Exception as e:
            # Non-fatal: continue even if logging fails
            print(f"Warning: failed to persist mcc record ({e})")

    def _read_jsonl(self, path):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def pad_to_match(self, a, b, pad_value=0):
        len_a, len_b = len(a), len(b)

        if len_a < len_b:
            a = np.pad(a, (0, len_b - len_a), constant_values=pad_value)
        elif len_b < len_a:
            b = np.pad(b, (0, len_a - len_b), constant_values=pad_value)

        return a, b

    def _plot_accuracy(self, wta, mcc, phi, pca):
        """Plot train/val accuracy (left axis) and val phi (right axis) from JSONL log."""
        if not getattr(self, "_acc_log_file", None) or not os.path.exists(
            self._acc_log_file
        ):
            return

        records = self._read_jsonl(self._acc_log_file)

        # Collect series keyed by epoch and method (JSONL may have multiple lines per epoch)
        if pca:
            train_acc_pca = {}
            val_acc_pca = {}
        if wta:
            train_acc_top = {}
            val_acc_top = {}
        if phi:
            val_phi = {}
        if mcc:
            train_mcc = {}
            val_mcc = {}

        for r in records:
            epoch = r.get("epoch")
            if epoch is None:
                continue

            split = r.get("split")
            method = r.get("method")

            if "accuracy" in r and r["accuracy"] is not None:
                acc_val = float(r["accuracy"])
                if split == "train":
                    if method == "pca_lr" and pca:
                        train_acc_pca[int(epoch)] = acc_val
                    elif method == "top" and wta:
                        train_acc_top[int(epoch)] = acc_val
                elif split == "val":
                    if method == "pca_lr" and pca:
                        val_acc_pca[int(epoch)] = acc_val
                    elif method == "top" and wta:
                        val_acc_top[int(epoch)] = acc_val

            if "phi" in r and r["phi"] is not None:
                # Based on your log format: phi is recorded under split="val"
                if split == "val" and phi:
                    val_phi[int(epoch)] = float(r["phi"])

            if "mcc" in r and r["mcc"] is not None:
                # Based on your log format: mcc is recorded under split="val"
                if split == "val" and mcc:
                    val_mcc[int(epoch)] = float(r["mcc"])
                if split == "train" and mcc:
                    train_mcc[int(epoch)] = float(r["mcc"])

        if wta:
            fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)
        else:
            fig, ax = plt.subplots()

        handles = []
        labels = []
        if wta:
            labels2 = []
            handles2 = []

        # if train_acc_pca:
        #     xs = sorted(train_acc_pca)
        #     train_acc = np.asarray([train_acc_pca[e] for e in xs])
        #     line = ax.plot(
        #         xs,
        #         train_acc,
        #         label="train acc (pca_lr)",
        #         linestyle="-",
        #         marker="o",
        #         markersize=3,
        #         color="gray",
        #     )
        #     handles.append(line[0])
        #     labels.append("Train accuracy")
        if pca:
            xs = sorted(val_acc_pca)
            val_acc = np.asarray([val_acc_pca[e] for e in xs])
            line = ax.plot(
                xs,
                val_acc,
                label="val acc (pca_lr)",
                linestyle="dotted",
                marker="o",
                markersize=3,
                color="red",
                linewidth=1.5,
            )
            handles.append(line[0])
            labels.append("Val accuracy")
        if wta:
            xs = sorted(train_acc_top)
            line = ax2.plot(
                xs,
                [train_acc_top[e] for e in xs],
                label="train acc (top)",
                linestyle="-",
                marker="s",
                markersize=3,
                color="red",
            )
            handles2.append(line[0])
            labels2.append("Train accuracy")
            xs = sorted(val_acc_top)
            line = ax2.plot(
                xs,
                [val_acc_top[e] for e in xs],
                label="val acc (top)",
                linestyle="-",
                marker="s",
                markersize=3,
                color="orange",
            )
            handles2.append(line[0])
            labels2.append("Val accuracy")

        # mcc (right axis)
        if mcc:
            xs = sorted(val_mcc)
            mcc_line = ax.plot(
                xs,
                [val_mcc[e] for e in xs],
                linestyle="solid",
                label="MCC",
                color="blue",
                marker="s",
                markersize=3,
                linewidth=1.5,
            )
            handles.append(mcc_line[0])
            labels.append("MCC")
            # xs = sorted(train_mcc)
            # mcc_line = ax.plot(
            #     xs,
            #     [train_mcc[e] for e in xs],
            #     linestyle="dashed",
            #     label="train mcc",
            #     color="blue",
            #     marker="s",
            #     markersize=3,
            #     linewidth=1.5,
            # )
            # handles.append(mcc_line[0])
            # labels.append("train mcc")

        # add mean line from the first position
        if pca:
            y_line = val_acc[0]
            ax.axhline(y=y_line, linestyle="dashed", linewidth=1, color="grey")

        if wta:
            fig.supylabel("Accuracy")
            fig.supxlabel("Batch")
            ax.set_ylabel("PCA+LR")
        else:
            ax.set_ylabel("Accuracy")
            ax.set_xlabel("Batches")

        # Phi (right axis)
        if phi:
            ax_tw = ax.twinx()
            xs = sorted(val_phi)
            phi_line = ax_tw.plot(
                xs,
                [val_phi[e] for e in xs],
                linestyle="--",
                label="val phi",
                color="green",
                marker="p",
                markersize=3,
                linewidth=1.5,
            )
            ax_tw.set_ylabel("Phi")
            handles.append(phi_line[0])
            labels.append("Phi")

        # Create legend with white background, box, and smaller font
        if handles:
            ax.legend(handles, labels, loc="lower left", framealpha=1.0, fontsize=5)
        else:
            ax.legend(loc="lower left", framealpha=1.0, fontsize=5)

        if wta:
            if handles2:
                ax2.legend(
                    handles2, labels2, loc="lower left", framealpha=1.0, fontsize=5
                )
            else:
                ax2.legend(loc="lower left", framealpha=1.0, fontsize=5)

        out_path = self._acc_log_file.replace(".jsonl", ".png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def preview_loaded_data(
        self, num_image_samples: int = 9, save_path: str | None = None
    ):
        """
        Plot a small grid of images from the loaded dataset once so the user can
        verify that the expected dataset is being used.
        """
        if getattr(self, "_image_preview_done", False):
            return
        # Special handling for geomfig: show N examples per class (0..3)
        if getattr(self, "image_dataset", "").lower() == "geomfig":
            try:
                import matplotlib.pyplot as plt
                from get_data import _geomfig_generate_one  # type: ignore
            except Exception as exc:
                print(f"Dataset preview skipped ({exc})")
                self._image_preview_done = True
                return
            try:
                classes = [0, 1, 2, 3]
                per_class = max(1, int(num_image_samples))
                # Special case: if 1 per class and 4 classes, arrange as 2x2 instead of 4x1
                if per_class == 1 and len(classes) == 4:
                    rows, cols = 2, 2
                    fig, axes = plt.subplots(rows, cols, figsize=(4.0, 4.0))
                    axes = axes.flatten()
                    titles = ["Triangle (0)", "Circle (1)", "Square (2)", "X (3)"]
                    for idx, cls in enumerate(classes):
                        img = _geomfig_generate_one(
                            cls_id=cls,
                            pixel_size=self.pixel_size,
                            noise_var=getattr(self, "geom_noise_var", 0.02),
                            jitter=getattr(self, "geom_jitter", False),
                            jitter_amount=getattr(self, "geom_jitter_amount", 0.05),
                        )
                        ax = axes[idx]
                        ax.imshow(img, cmap="gray")
                        ax.set_title(titles[idx], fontsize=10)
                        ax.axis("off")
                else:
                    fig, axes = plt.subplots(
                        len(classes),
                        per_class,
                        figsize=(2.0 * per_class, 2.0 * len(classes)),
                    )
                    if per_class == 1:
                        axes = np.atleast_2d(axes).reshape(len(classes), 1)
                    titles = ["Triangle (0)", "Circle (1)", "Square (2)", "X (3)"]
                    for r, cls in enumerate(classes):
                        for c in range(per_class):
                            img = _geomfig_generate_one(
                                cls_id=cls,
                                pixel_size=self.pixel_size,
                                noise_var=getattr(self, "geom_noise_var", 0.02),
                                jitter=getattr(self, "geom_jitter", False),
                                jitter_amount=getattr(self, "geom_jitter_amount", 0.05),
                            )
                            ax = axes[r, c]
                            ax.imshow(img, cmap="gray")
                            if c == 0:
                                ax.set_title(titles[r], fontsize=10)
                            ax.axis("off")
                plt.tight_layout()
                try:
                    if save_path is None:
                        os.makedirs("plots", exist_ok=True)
                        save_path = os.path.join("plots", "geomfig_preview.png")
                    fig.savefig(save_path)
                    print(f"Dataset preview saved to {save_path}")
                except Exception as exc:
                    print(f"Failed to save dataset preview ({exc})")
                plt.show()
                plt.close(fig)
            except Exception as exc:
                print(f"Dataset preview skipped ({exc})")
            finally:
                self._image_preview_done = True
            return
        # Default: use image streamer preview if available
        if not hasattr(self, "image_streamer") or self.image_streamer is None:
            return
        try:
            self.image_streamer.show_preview(
                num_samples=num_image_samples, save_path=save_path
            )
        except Exception as exc:  # pragma: no cover - visualization fallback
            print(f"Dataset preview skipped ({exc})")
        finally:
            self._image_preview_done = True

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
            self._save_model_dir(model_dir)

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
                    self._log("No exact model match; loading latest available.")
                except Exception:
                    matched_folder = None

            if matched_folder is not None:
                folder = matched_folder
                self._load_model_dir(folder)
                print("\rmodel loaded", end="")
                self.model_loaded = True
                return os.path.join("model", folder)
            else:
                self._log("No model found to load. Will train new model from scratch.")

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
        num_steps=100,
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
        max_rate_hz=67.0,
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
        image_dataset="mnist",
        geom_noise_var=0.02,
        geom_noise_mean=0.0,
        geom_jitter=False,
        geom_jitter_amount=0.05,
        geom_workers=None,
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
        self.image_dataset = (image_dataset or "mnist").lower()
        self.plot_spectrograms = plot_spectrograms
        self.all_images_train = all_images_train
        self.all_images_test = all_images_test
        self.all_images_val = all_images_val
        # Geomfig-specific knobs (optional)
        self.geom_jitter = locals().get("geom_jitter", False)
        self.geom_jitter_amount = locals().get("geom_jitter_amount", 0.05)
        self.geom_noise_var = locals().get("geom_noise_var", 0.02)
        self.geom_noise_mean = locals().get("geom_noise_mean", 0.0)
        self.geom_workers = geom_workers
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
        self.data_val = None
        self.labels_val = None

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

        # Geomfig dataset — load from cache or generate; wrap in streamer for batched access
        if self.image_dataset == "geomfig":
            from get_data import load_or_create_geomfig_data

            # Use configured counts
            train_count = int(self.all_images_train)
            val_count = int(self.all_images_val)
            test_count = int(self.all_images_test)
            noise_var = locals().get("geom_noise_var", 0.02)
            noise_mean = locals().get("geom_noise_mean", 0.0)
            data_tr, lab_tr, data_va, lab_va, data_te, lab_te = (
                load_or_create_geomfig_data(
                    pixel_size=self.pixel_size,
                    num_steps=num_steps,
                    gain=gain,
                    train_count=train_count,
                    val_count=val_count,
                    test_count=test_count,
                    noise_var=noise_var,
                    noise_mean=noise_mean,
                    jitter=getattr(self, "geom_jitter", False),
                    jitter_amount=getattr(self, "geom_jitter_amount", 0.05),
                    seed=42,
                    force_recreate=bool(force_recreate),
                    num_workers=self.geom_workers,
                )
            )
            # Wrap in streamer for consistent batched access
            self.image_streamer = GeomfigDataStreamer(
                data_train=data_tr,
                labels_train=lab_tr,
                data_val=data_va,
                labels_val=lab_va,
                data_test=data_te,
                labels_test=lab_te,
                batch_size=batch_image_train,
                num_steps=num_steps,
            )
            # Also keep direct references for backward compatibility (if needed)
            self.data_train = (
                data_tr if data_tr is not None and len(data_tr) > 0 else None
            )
            self.labels_train = (
                lab_tr if lab_tr is not None and len(lab_tr) > 0 else None
            )
            self.data_val = (
                data_va if data_va is not None and len(data_va) > 0 else None
            )
            self.labels_val = lab_va if lab_va is not None and len(lab_va) > 0 else None
            self.data_test = (
                data_te if data_te is not None and len(data_te) > 0 else None
            )
            self.labels_test = (
                lab_te if lab_te is not None and len(lab_te) > 0 else None
            )
            self.N_classes = 4

            # Short summary
            def _s(x):
                return 0 if x is None else len(x)

            print(
                f"Geomfig data prepared — train:{_s(self.data_train)} val:{_s(self.data_val)} test:{_s(self.data_test)} timesteps (using batched streamer)"
            )

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

            # Ensure data/mdata directory exists|
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
                        expected_params = {
                            "pixel_size": self.pixel_size,
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
                            "dataset": self.image_dataset,
                            "train_count": all_images_train,
                            "val_count": all_images_val,
                            "test_count": all_images_test,
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
                    "pixel_size": self.pixel_size,
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
                    "dataset": self.image_dataset,
                    "train_count": all_images_train,
                    "val_count": all_images_val,
                    "test_count": all_images_test,
                }

                with open(
                    os.path.join(image_data_dir, "image_data_parameters.json"), "w"
                ) as f:
                    json.dump(image_params, f)

            # Determine pixel size based on mode
            self.image_streamer = ImageDataStreamer(
                "data",  # Use the data directory for MNIST download
                batch_size=batch_image_train,
                pixel_size=self.pixel_size,
                num_steps=num_steps,
                gain=gain,
                offset=offset,
                max_rate_hz=max_rate_hz,
                first_spike_time=first_spike_time,
                time_var_input=time_var_input,
                train_count=all_images_train,
                val_count=all_images_val,
                test_count=all_images_test,
                dataset=self.image_dataset,
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
                    "pixel_size": self.pixel_size,
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
                        pixel_size=self.pixel_size,
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
                        pixel_size=self.pixel_size,
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
                heat_map(self.data_train, pixel_size=self.pixel_size)

            # return data and labels if needed
            if retur:
                if use_validation_data:
                    return (
                        self.data_train,
                        self.labels_train,
                    )
                else:
                    return self.data_train, self.labels_train

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

    # --- Helpers: logging, model IO, and PCA ---
    def _log(self, message):
        try:
            if getattr(self, "verbose", False):
                print(message)
        except Exception:
            pass

    def _save_model_dir(self, model_dir):
        """Save essentials only: weights and parameters (JSON saved by caller)."""
        try:
            np.save(
                os.path.join(model_dir, "weights.npy"), getattr(self, "weights", None)
            )
        except Exception as e:
            print(f"Warning: model save failed ({e})")

    def _load_model_dir(self, folder):
        """Load essentials only from model/<folder>."""
        self.weights = np.load(os.path.join("model", folder, "weights.npy"))
        # Other large artifacts are not persisted to save disk space.

    def _pca_eval(self, X_train, y_train, X_test, y_test, split):
        """Run PCA-based classifier with configured options, return (accs, scaler, pca, clf)."""
        if self.use_QDA:
            from pca_linear_classifier import pca_quadratic_discriminant

            return pca_quadratic_discriminant(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                variance_ratio=self.pca_variance,
                reg_param=0.5,
            )
        else:
            from pca_linear_classifier import pca_logistic_regression

            if split == "train":
                self.clf = None
                self.pca = None
                self.scaler = None

            return pca_logistic_regression(
                X_train=X_train,
                y_train=y_train,
                split=split,
                X_test=X_test,
                y_test=y_test,
                variance_ratio=self.pca_variance,
                clf=self.clf,
                scaler=self.scaler,
                pca=self.pca,
            )

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

    def plot_spikes(self):
        label_for_plotting = input("Which label should we plot? ")
        while label_for_plotting != "stop":
            if label_for_plotting == "all":
                frame_folder = f"plots\\spikes\\{self.image_dataset}\\all\\{self.ts}\\{self.ts_spec}"
                output_filename = f"plots\\spikes\\{self.image_dataset}\\all\\{self.ts}\\{self.ts_spec}\\evolution.gif"
            else:
                frame_folder = f"plots\\spikes\\{self.image_dataset}\\{label_for_plotting}\\{self.ts}\\{self.ts_spec}"
                output_filename = f"plots\\spikes\\{self.image_dataset}\\{label_for_plotting}\\{self.ts}\\{self.ts_spec}\\evolution.gif"
            gif_spike_rate_by_label(
                frame_folder=frame_folder,
                output_filename=output_filename,
                duration=100,
                loop=0,
            )
            label_for_plotting = input("Which label should we plot? ")

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
        tau_trace=25,
        learning_rate_exc=0.0008,
        learning_rate_inh=0.005,
        w_target_exc=0.01,
        w_target_inh=-0.01,
        var_noise=1,
        min_weight_inh=-25,
        x_tar=0.1,
        track_weights=False,
        max_weight_inh=-0.01,
        max_weight_exc=25,
        min_weight_exc=0.01,
        spike_threshold_default=-55,
        check_sleep_interval=50000,  # Reduced frequency for better performance
        interval=5000,  # Reduced frequency for better performance
        min_mp=-100,
        sleep=False,
        sleep_ratio=0.0,  # Sleep percentage per interval (e.g., 0.1 = 10%)
        normalize_weights=False,  # Alternative to sleep: maintain initial weight sum
        normalize_per_column=True,  # Per-post-neuron normalization to initial column sums
        normalize_per_column_interval=1000,  # Normalize every N timesteps (default: 1000)
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
        delta_adaption=1,
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
        w_max=10,
        early_stopping=False,
        early_stopping_patience_pct=0.1,  # Patience as percentage of total epochs (0.1 = 10%)
        dt=1,
        membrane_resistance_exc=30,
        membrane_resistance_inh=30,
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
        heatmap_plot=False,
        get_giffed=False,
        sleep_synchronized=True,
        tau_pre_trace_exc=1,
        tau_pre_trace_inh=1,
        tau_post_trace_exc=1,
        tau_post_trace_inh=1,
        weight_mean_noise=0.05,
        weight_var_noise=0.005,
        num_inh=10,
        num_exc=50,
        mu_weight=0.6,
        plot_epoch_performance=True,
        plot_weights_per_epoch=False,  # Plot weights after each epoch (for debugging)
        plot_spikes_per_epoch=False,  # Plot spikes after each epoch (for debugging)
        narrow_top=0.2,  # Increased from 0.05 to 0.2 (20% of neurons)
        wide_top=0.15,
        tau_syn_exc=30,
        tau_syn_inh=30,
        tau_m_exc=30,
        tau_m_inh=30,
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
        patience=None,  # Can override percentage with explicit epoch count
        use_QDA=False,
        use_LR=True,
        # Hard-pause sleep knobs
        sleep_max_iters=5000,
        on_timeout="scale_to_target",
        sleep_tol_frac=1e-3,
        sleep_mode="static",
        track_stats=False,
    ):
        self.dt = dt
        self.pca_variance = pca_variance
        self.use_validation_data = use_validation_data
        self.use_QDA = use_QDA
        self.use_LR = use_LR
        self.sleep_ratio = sleep_ratio
        self.normalize_weights = normalize_weights

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
            # define which weights counts towards total sum of weights
            sum_weights_exc = np.sum(np.abs(self.weights[: self.ex, self.st : self.ih]))
            sum_weights_inh = np.sum(
                np.abs(self.weights[self.ex : self.ih, self.st : self.ex])
            )
            sum_weights = np.sum(np.abs(self.weights))

            # Store initial sums for normalization (if enabled)
            initial_sum_exc = sum_weights_exc
            initial_sum_inh = sum_weights_inh
            initial_sum_total = sum_weights

            # Per-column initial sums (one per post-neuron)
            initial_sum_st_ex = np.sum(
                np.abs(self.weights[: self.st, self.st : self.ex]), axis=0
            )  # (N_exc,)
            initial_sum_ex_ex = np.sum(
                np.abs(self.weights[self.st : self.ex, self.st : self.ex]), axis=0
            )  # (N_exc,)
            initial_sum_ex_ih = np.sum(
                np.abs(self.weights[self.st : self.ex, self.ex : self.ih]), axis=0
            )  # (N_inh,)
            initial_sum_ih_ex = np.sum(
                np.abs(self.weights[self.ex : self.ih, self.st : self.ex]), axis=0
            )  # (N_exc,)

            baseline_sum_exc = sum_weights_exc * beta
            baseline_sum_inh = sum_weights_inh * beta
            baseline_sum = sum_weights * beta
            max_sum_exc = sum_weights_exc * alpha
            max_sum_inh = sum_weights_inh * alpha
            max_sum = sum_weights * alpha

            # Bundle common training arguments
            common_args = dict(
                tau_syn_exc=tau_syn_exc,
                tau_syn_inh=tau_syn_inh,
                tau_m_exc=tau_m_exc,
                tau_m_inh=tau_m_inh,
                w_max=w_max,
                mu_weight=mu_weight,
                resting_potential=self.resting_potential,
                membrane_resistance_exc=membrane_resistance_exc,
                membrane_resistance_inh=membrane_resistance_inh,
                min_weight_exc=min_weight_exc,
                max_weight_exc=max_weight_exc,
                min_weight_inh=min_weight_inh,
                max_weight_inh=max_weight_inh,
                N_exc=self.N_exc,
                N_inh=self.N_inh,
                track_stats=track_stats,
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
                normalize_weights=normalize_weights,
                normalize_per_column=normalize_per_column,
                normalize_per_column_interval=normalize_per_column_interval,
                initial_sum_exc=initial_sum_exc,
                initial_sum_inh=initial_sum_inh,
                initial_sum_total=initial_sum_total,
                initial_sum_st_ex=initial_sum_st_ex,
                initial_sum_ex_ex=initial_sum_ex_ex,
                initial_sum_ex_ih=initial_sum_ex_ih,
                initial_sum_ih_ex=initial_sum_ih_ex,
                # pass hard-pause knobs
                sleep_max_iters=sleep_max_iters,
                on_timeout=on_timeout,
                sleep_tol_frac=sleep_tol_frac,
                sleep_mode=sleep_mode,
            )

            # Keep weight calculation variables available for final test pass

            # pre-define performance tracking array
            if test_only and not train_weights:
                # single pass; tracker will be (1,2) later
                self.performance_tracker = np.zeros((1, 2))
            else:
                self.performance_tracker = np.zeros((self.epochs, 2))

            # early stopping setup
            patience_epochs = None

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
            sleep_percent_sum = 0.0
            sleep_percent_count = 0

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
                except Exception:
                    pass

                if self.image_streamer is not None:
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
                        _,
                        spikes_train,
                        _,
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
                    if mp_train is None:
                        # Reuse pre-allocated arrays
                        mp_train = np.zeros(self.N - self.N_x)
                        mp_train[:] = self.resting_potential
                    # add input data to array
                    spikes_train = np.zeros((self.T_train, self.N), dtype=np.int8)
                    spikes_train[:, : self.N_x] = data_train
                    if spike_trace is None:
                        spike_trace = np.zeros(self.N - self.N_inh)
                d = 5

                (
                    self.weights,
                    spikes_tr_out,
                    mp_train,
                    __,
                    __,
                    thresh_tr,
                    __,
                    __,
                    labels_tr_out,
                    sleep_tr_out,
                    I_syn_exc,
                    I_syn_inh,
                    a,
                    __,
                    spike_trace,
                    x_tar,
                ) = train_network(
                    weights=(self.weights if train_weights else self.weights.copy()),
                    spike_labels=labels_train,
                    mp=mp_train,
                    sleep=sleep,
                    train_weights=train_weights,
                    tau_trace=tau_trace,
                    T=self.T_train,
                    run=self.ts_spec,
                    save_plots=False,
                    dataset=self.image_dataset,
                    mean_noise=mean_noise,
                    spike_trace=spike_trace.copy(),
                    var_noise=var_noise,
                    spikes=spikes_train,
                    check_sleep_interval=check_sleep_interval,
                    timing_update=timing_update,
                    spike_threshold=spike_threshold,
                    a=a,
                    track_weights=track_weights,
                    x_tar=x_tar,
                    I_syn_exc=I_syn_exc,
                    I_syn_inh=I_syn_inh,
                    sleep_ratio=getattr(self, "sleep_ratio", 0.0),
                    **common_args,
                )
                spike_threshold = thresh_tr
                # plot gif
                if get_giffed:
                    self.plot_spikes()

                # accumulate sleep percent if available
                try:
                    if sleep and sleep_tr_out is not None:
                        sleep_percent_sum += float(sleep_tr_out)
                        sleep_percent_count += 1
                except Exception:
                    pass

                # Calculate training accuracy for current epoch
                if spikes_tr_out is not None and labels_tr_out is not None:
                    print(
                        f"\n--- Epoch {e+1} Training Accuracy ({accuracy_method.upper()}) ---"
                    )

                    if accuracy_method == "top":
                        # Use top-responders method for training accuracy
                        train_acc_batch, dbg_train, state = WTA_accuracy(
                            spikes=spikes_tr_out[:, self.st : self.ex],
                            labels=labels_tr_out,
                            num_classes=self.N_classes,
                            smoothening=self.num_steps,
                            split="train",
                        )
                        print(f"Training Accuracy (TOP): {train_acc_batch:.4f}")
                        try:
                            self._record_accuracy(
                                "train", train_acc_batch, epoch=e + 1, method="top"
                            )
                        except Exception:
                            pass

                    elif accuracy_method == "pca_lr":
                        # Use PCA+LR method for training accuracy
                        try:
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
                            nz_ex_ws = self.weights[
                                self.st : self.ex, self.st : self.ih
                            ][self.weights[self.st : self.ex, self.st : self.ih] != 0]
                            nz_ih_ws = self.weights[
                                self.ex : self.ih, self.st : self.ex
                            ][self.weights[self.ex : self.ih, self.st : self.ex] != 0]
                            print(f"Mean input weights: {np.mean(nz_st_ws)}")
                            print(f"Mean excitatory weights: {np.mean(nz_ex_ws)}")
                            print(f"Mean inhibitory weights: {np.mean(nz_ih_ws)}")

                            X_tr, y_tr = bin_spikes_by_label_no_breaks(
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
                                    raise ValueError(
                                        "not enough samples to split and perform accuracy estimation"
                                    )

                                acc, mcc, self.scaler, self.pca, self.clf = (
                                    self._pca_eval(  # TRAIN STOPS HERE
                                        X_train=X_tr_split,
                                        y_train=y_tr_split,
                                        X_test=X_te_split,
                                        y_test=y_te_split,
                                        split="train",
                                    )
                                )
                                print(f"Training Accuracy (PCA+LR): {acc:.4f}")
                                print(f"Training MCC (PCA+LR): {mcc:.4f}")

                                # Also compute top_method accuracy for comparison
                                try:
                                    train_acc_top, __, state = WTA_accuracy(
                                        spikes=spikes_tr_out[:, self.st : self.ex],
                                        labels=labels_tr_out,
                                        num_classes=self.N_classes,
                                        smoothening=self.num_steps,
                                        split="train",
                                    )
                                    print(
                                        f"Training Accuracy (TOP): {train_acc_top:.4f}"
                                    )
                                except Exception as ex:
                                    print(
                                        f"Warning: Top method accuracy computation failed ({ex})"
                                    )
                                    train_acc_top = None
                                self._record_accuracy(
                                    "train",
                                    acc,
                                    epoch=e + 1,
                                    method="pca_lr",
                                )
                                self._record_accuracy(
                                    "train",
                                    train_acc_top,
                                    epoch=e + 1,
                                    method="top",
                                )
                                self._record_mcc(
                                    "train",
                                    mcc,
                                    epoch=e + 1,
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
                    self.all_val // self.batch_val
                )  # Use test dataset size for validation batching
                acc_LR_sum = 0.0
                acc_top_sum = 0.0
                mcc_sum = 0.0
                phi_sum = 0.0
                val_sample_count = 0

                # Predefine variables used in the test loop so cleanup never fails
                data_test = None
                labels_test = None
                mp_test = None
                weights_te = None
                spikes_te_out = None
                labels_te_out = None
                sleep_te_out = None
                I_syn_te = None
                a_te = None
                T_test_batch = 0

                for val_batch_idx in range(total_num_vals):
                    # Image only streaming mode
                    from get_data import load_image_batch

                    val_start_idx = val_batch_idx * self.batch_val
                    data_test, labels_test = load_image_batch(
                        self.image_streamer,
                        val_start_idx,
                        self.batch_val,
                        self.num_steps,
                        int(np.sqrt(self.N_x)) ** 2,  # Image-only mode uses full N_x
                        partition="val",  # Use validation data
                    )
                    if data_test is None:
                        print(f"No more validation image data available")
                        break

                    # Update T_test for this batch
                    # Ensure test data width matches expected N_x
                    if data_test is not None:
                        data_test = data_test[:, : self.st]
                    T_val_batch = data_test.shape[0]

                    mp_test = np.zeros((self.N - self.st))
                    mp_test[:] = self.resting_potential

                    spikes_test = np.zeros((T_val_batch, self.N), dtype=np.int8)
                    spikes_test[:, : self.st] = data_test
                    spike_trace_val = np.zeros(self.N - self.N_inh)
                    a_test = np.zeros(self.N_exc)
                    I_syn_exc_test = np.zeros(self.N_exc)
                    I_syn_inh_test = np.zeros(self.N_inh)
                    spike_threshold_test = np.full(
                        self.N_exc + self.N_inh, fill_value=spike_threshold_default
                    )

                    (
                        weights_te,
                        spikes_te_out,
                        mp_te,
                        *unused,
                        labels_te_out,
                        sleep_te_out,
                        I_syn_exc_te,
                        I_syn_inh_te,
                        a_te,
                        weight_tracking_te,
                        _spike_trace_te,
                        _,
                    ) = train_network(  # VAL STOPS HERE
                        weights=self.weights.copy(),
                        spike_labels=labels_test.copy(),
                        mp=mp_test.copy(),
                        sleep=False,
                        dataset=self.image_dataset,
                        train_weights=False,
                        T=T_val_batch,
                        mean_noise=mean_noise,
                        run=self.ts_spec,
                        save_plots=heatmap_plot,
                        tau_trace=tau_trace,
                        track_weights=False,
                        x_tar=x_tar,
                        spike_trace=spike_trace_val,
                        var_noise=var_noise,
                        spikes=spikes_test.copy(),
                        check_sleep_interval=check_sleep_interval,
                        timing_update=timing_update,
                        a=a_test,
                        I_syn_exc=I_syn_exc_test,
                        I_syn_inh=I_syn_inh_test,
                        spike_threshold=spike_threshold_test,
                        sleep_ratio=self.sleep_ratio,
                        **common_args,
                    )

                    # Store results for accumulation (use pre-allocated arrays)
                    batch_size = spikes_te_out.shape[0]
                    val_sample_count += batch_size

                    # Prepare features from validation data for testing
                    X_val_dist, y_val_dist = bin_spikes_by_label_no_breaks(
                        spikes=spikes_te_out[:, self.st : self.ex],
                        labels=labels_te_out,
                    )

                    # PCA+LR accuracy estimation
                    acc_LR, mcc = self._pca_eval(
                        X_train=None,
                        y_train=None,
                        X_test=X_val_dist,
                        y_test=y_val_dist,
                        split="val",
                    )
                    acc_LR_sum += acc_LR
                    mcc_sum += mcc

                    # WTA accuracy estimation
                    acc_top, _, _ = WTA_accuracy(
                        spikes=spikes_te_out[:, self.st : self.ex],
                        labels=labels_te_out,
                        num_classes=self.N_classes,
                        smoothening=self.num_steps,
                        split="val",
                        state=state,
                    )
                    acc_top_sum += acc_top

                    # Phi estimation
                    phi_train, phi_val, *unused = calculate_phi(
                        spikes_train=spikes_tr_out[:, self.st : self.ex],
                        spikes_test=spikes_te_out[:, self.st : self.ex],
                        labels_train=labels_tr_out,
                        labels_test=labels_te_out,
                        num_steps=self.num_steps,
                        pca_variance=self.pca_variance,
                        random_state=random_state,
                        num_classes=self.N_classes,
                    )
                    phi_sum += phi_val
                    print("phi train: ", phi_train)

                # estimate the mean accuracy and phi
                final_acc_top = acc_top_sum / max(1, total_num_vals)
                final_acc_LR = acc_LR_sum / max(1, total_num_vals)
                final_phi = phi_sum / max(1, total_num_vals)
                final_mcc = mcc_sum / max(1, total_num_vals)

                # print results
                print(f"Validation accuracy (PCA+LR): {final_acc_LR}")
                print(f"Validation accuracy (WTA): {final_acc_top}")
                print(f"Validation phi: {final_phi}")
                print(f"Validation MCC: {final_mcc}")

                # record the test accuracy
                if final_acc_top is not None:
                    self._record_accuracy(
                        "val",
                        final_acc_top,
                        epoch=e + 1,
                        method="top",
                    )
                if final_acc_LR is not None:
                    self._record_accuracy(
                        "val",
                        final_acc_LR,
                        epoch=e + 1,
                        method="pca_lr",
                    )
                if final_phi is not None:
                    self._record_phi("val", final_phi, epoch=e + 1)

                if final_mcc is not None:
                    self._record_mcc("val", final_mcc, epoch=e + 1)

                # plot validation and training accuracy progress
                self._plot_accuracy(mcc=True, pca=True, wta=False, phi=True)

                # Rinse memory
                if e != self.epochs - 1:
                    # Clean up training data
                    del data_train, labels_train
                    # Clean up test data (these exist in the test loop)
                    del (
                        data_test,
                        labels_test,
                        weights_te,
                        spikes_te_out,
                        labels_te_out,
                        sleep_te_out,
                        I_syn_te,
                        a_te,
                    )
                    # Clean up training results
                    del spikes_tr_out, labels_tr_out, sleep_tr_out
                    # Clean up temporary variables (validation accumulation variables)
                    try:
                        del (
                            T_test_batch,
                            ex,
                        )
                    except:
                        pass
                    gc.collect()

                pbar.set_description(f"Epoch {e+1}/{self.epochs}")
                # Handle None valuTruees safely
                acc_str = f"{final_acc_LR:.3f}" if final_acc_LR is not None else "N/A"
                phi_str = f"{final_phi:.2f}" if final_phi is not None else "N/A"
                pbar.set_postfix(acc=acc_str, phi=phi_str)
                pbar.update(1)
            pbar.close()

            # Clean up main training loop variables
            del (
                I_syn_exc,
                I_syn_inh,
                a,
                spike_threshold,
            )
            del (common_args,)
            try:
                del download, data_dir
            except:
                pass
            gc.collect()

        # Final test pass even if early-stopped (evaluate on test partition)
        from get_data import load_image_batch

        ################## THIS IS WHERE WE TEST THE MODEL ######################
        # Reset test pointers:
        self.image_streamer.reset_partition("test")

        # Reuse test-only evaluator logic with partition="test"
        total_num_tests = max(1, self.all_images_test // self.batch_image_test)
        test_sample_count = 0
        acc_top_sum = 0.0
        acc_LR_sum = 0.0
        mcc_sum = 0.0
        phi_sum = 0.0

        for __ in range(total_num_tests):
            bs = min(
                self.batch_image_test,
                self.all_images_test - self.batch_image_test + test_sample_count,
            )
            bs_spikes = int(bs * self.num_steps)
            # make sure we are actually generating content from the test-set of MNIST
            data_test, labels_test = load_image_batch(
                self.image_streamer,
                0,
                bs,
                self.num_steps,
                self.pixel_size,
                partition="test",
            )

            if data_test is None:
                raise ValueError("data_test is None")

            mp_test = np.zeros((self.ih - self.st))
            mp_test[:] = self.resting_potential
            spikes_test = np.zeros((bs_spikes, self.N), dtype=np.int8)
            spikes_test[:, : self.st] = data_test
            a = np.zeros(self.N_exc)
            I_syn_exc = np.zeros(self.N_exc)
            I_syn_inh = np.zeros(self.N_inh)
            spike_threshold = np.full(
                self.N_exc, fill_value=spike_threshold_default, dtype=float
            )
            spike_trace_te = np.zeros(self.N - self.N_inh)

            # Build common args locally in case outer scope wasn't initialized in this path
            try:
                _ca = common_args
            except Exception:
                _ca = dict(
                    tau_syn_exc=tau_syn_exc,
                    tau_syn_inh=tau_syn_inh,
                    tau_m_exc=tau_m_exc,
                    tau_m_inh=tau_m_inh,
                    w_max=w_max,
                    mu_weight=mu_weight,
                    resting_potential=self.resting_potential,
                    membrane_resistance_exc=membrane_resistance_exc,
                    membrane_resistance_inh=membrane_resistance_inh,
                    min_weight_exc=min_weight_exc,
                    max_weight_exc=max_weight_exc,
                    min_weight_inh=min_weight_inh,
                    max_weight_inh=max_weight_inh,
                    N_exc=self.N_exc,
                    N_inh=self.N_inh,
                    track_stats=track_stats,
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
                    normalize_weights=normalize_weights,
                    normalize_per_column=normalize_per_column,
                    normalize_per_column_interval=normalize_per_column_interval,
                    initial_sum_exc=initial_sum_exc,
                    initial_sum_inh=initial_sum_inh,
                    initial_sum_total=initial_sum_total,
                    initial_sum_st_ex=initial_sum_st_ex,
                    initial_sum_ex_ex=initial_sum_ex_ex,
                    initial_sum_ex_ih=initial_sum_ex_ih,
                    initial_sum_ih_ex=initial_sum_ih_ex,
                    # pass hard-pause knobs
                    sleep_max_iters=sleep_max_iters,
                    on_timeout=on_timeout,
                    sleep_tol_frac=sleep_tol_frac,
                    sleep_mode=sleep_mode,
                )

            (
                weights_te,
                spikes_te_out,
                mp_te,
                *unused,
                labels_te_out,
                sleep_te_out,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = train_network(
                weights=self.weights.copy(),
                spike_labels=labels_test.copy(),
                mp=mp_test.copy(),
                sleep=False,
                dataset=self.image_dataset,
                train_weights=False,
                T=T_test_batch,
                mean_noise=mean_noise,
                run=self.ts_spec,
                save_plots=False,
                tau_trace=tau_trace,
                track_weights=track_weights,
                x_tar=x_tar,
                spike_trace=spike_trace_te,
                var_noise=var_noise,
                spikes=spikes_test.copy(),
                check_sleep_interval=check_sleep_interval,
                timing_update=timing_update,
                a=a.copy(),
                I_syn_exc=I_syn_exc.copy(),
                I_syn_inh=I_syn_inh.copy(),
                spike_threshold=spike_threshold.copy(),
                sleep_ratio=self.sleep_ratio,
                **_ca,
            )

            # Increase sample index
            test_sample_count += bs

            # PCA+LR accuracy estimation
            acc_LR, mcc = self._pca_eval(
                X_train=None,
                y_train=None,
                X_test=spikes_te_out[:, self.st : self.ex],
                y_test=labels_te_out,
                split="test",
            )
            acc_LR_sum += acc_LR
            mcc_sum += mcc

            # WTA accuracy estimation
            acc_top, dbg_test, _ = WTA_accuracy(
                spikes=spikes_te_out[:, self.st : self.ex],
                labels=labels_te_out,
                num_classes=self.N_classes,
                smoothening=self.num_steps,
                split="test",
                state=state,
            )
            acc_top_sum += acc_top

            # Phi estimation
            phi_tr, phi_te, *_ = calculate_phi(
                spikes_train=spikes_tr_out[:, self.st : self.ex],
                spikes_test=spikes_te_out[:, self.st : self.ex],
                labels_train=labels_tr_out,
                labels_test=labels_te_out,
                num_steps=self.num_steps,
                pca_variance=self.pca_variance,
                random_state=random_state,
                num_classes=self.N_classes,
            )
            phi_sum += phi_te
            print("phi train: ", phi_tr)

        # estimate the mean accuracy and phi
        final_acc_top = acc_top_sum / max(1, total_num_tests)
        final_acc_LR = acc_LR_sum / max(1, total_num_tests)
        final_phi = phi_sum / max(1, total_num_tests)
        final_mcc = mcc_sum / max(1, total_num_tests)

        # print results
        print(f"Testing accuracy (PCA+LR): {final_acc_LR}")
        print(f"Testing accuracy (WTA): {final_acc_top}")
        print(f"Testing phi: {final_phi}")
        print(f"Testing MCC: {final_mcc}")

        # record the test accuracy
        if final_acc_top is not None:
            self._record_accuracy(
                "test",
                final_acc_top,
                epoch=e + 1,
                method="top",
            )
        if final_acc_LR is not None:
            self._record_accuracy(
                "test",
                final_acc_LR,
                epoch=e + 1,
                method="pca_lr",
            )
        if final_phi is not None:
            self._record_phi("test", final_phi, epoch=e + 1)

        # Clean up final test arrays and variables
        del test_sample_count, acc_top_sum
