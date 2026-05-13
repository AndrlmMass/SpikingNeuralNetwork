import os
import json
import numpy as np


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
                json_file_path = os.path.join("model", folder, "model_parameters.json")
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
            self.phi_all_scores = np.load(os.path.join(data_dir, "phi_all_scores.npy"))
            self.loaded_phi_model = True
            print("\rphi data has been loaded", end="")


def _log(self, message):
    try:
        if getattr(self, "verbose", False):
            print(message)
    except Exception:
        pass


def _save_model_dir(self, model_dir):
    """Save essentials only: weights and parameters (JSON saved by caller)."""
    try:
        np.save(os.path.join(model_dir, "weights.npy"), getattr(self, "weights", None))
    except Exception as e:
        print(f"Warning: model save failed ({e})")


def _load_model_dir(self, folder):
    """Load essentials only from model/<folder>."""
    self.weights = np.load(os.path.join("model", folder, "weights.npy"))
