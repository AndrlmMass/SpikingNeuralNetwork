import json
import os

import numpy as np


class CheckpointManager:
    def save_model(self, weights, parameters, base_dir="model") -> str:
        os.makedirs(base_dir, exist_ok=True)
        rng = np.random.default_rng()
        tag = str(rng.integers(10000, 99999))
        while tag in os.listdir(base_dir):
            tag = str(rng.integers(10000, 99999))
        model_dir = os.path.join(base_dir, tag)
        os.makedirs(model_dir, exist_ok=True)
        np.save(os.path.join(model_dir, "weights.npy"), weights)
        with open(os.path.join(model_dir, "model_parameters.json"), "w") as f:
            json.dump(parameters, f)
        print(f"\rmodel saved → {model_dir}", end="")
        return model_dir

    def load_model(self, parameters, base_dir="model"):
        if not os.path.exists(base_dir):
            return None, None
        folders = [
            f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))
        ]
        # exact parameter match first
        for folder in folders:
            param_path = os.path.join(base_dir, folder, "model_parameters.json")
            if not os.path.exists(param_path):
                continue
            with open(param_path) as f:
                stored = json.load(f)
            if stored == parameters:
                weights = np.load(os.path.join(base_dir, folder, "weights.npy"))
                print(f"\rmodel loaded ← {folder}", end="")
                return weights, os.path.join(base_dir, folder)
        # fall back to most recently modified folder
        if folders:
            latest = max(
                folders, key=lambda f: os.path.getmtime(os.path.join(base_dir, f))
            )
            weights_path = os.path.join(base_dir, latest, "weights.npy")
            if os.path.exists(weights_path):
                weights = np.load(weights_path)
                print(f"\rmodel loaded (latest) ← {latest}", end="")
                return weights, os.path.join(base_dir, latest)
        return None, None

    def save_data(self, data_train, labels_train, data_test, labels_test, parameters, base_dir="data/sdata"):
        os.makedirs(base_dir, exist_ok=True)
        rng = np.random.default_rng()
        tag = str(rng.integers(10000, 99999))
        while tag in os.listdir(base_dir):
            tag = str(rng.integers(10000, 99999))
        data_dir = os.path.join(base_dir, tag)
        os.makedirs(data_dir, exist_ok=True)
        np.save(os.path.join(data_dir, "data_train.npy"), data_train)
        np.save(os.path.join(data_dir, "labels_train.npy"), labels_train)
        np.save(os.path.join(data_dir, "data_test.npy"), data_test)
        np.save(os.path.join(data_dir, "labels_test.npy"), labels_test)
        with open(os.path.join(data_dir, "data_parameters.json"), "w") as f:
            json.dump(parameters, f)
        print("\rdata saved", end="")

    def load_data(self, parameters, base_dir="data/sdata"):
        if not os.path.exists(base_dir):
            return None
        for folder in os.listdir(base_dir):
            param_path = os.path.join(base_dir, folder, "data_parameters.json")
            if not os.path.exists(param_path):
                continue
            with open(param_path) as f:
                stored = json.load(f)
            if stored == parameters:
                print("\rdata loaded", end="")
                return {
                    "data_train": np.load(os.path.join(base_dir, folder, "data_train.npy")),
                    "labels_train": np.load(os.path.join(base_dir, folder, "labels_train.npy")),
                    "data_test": np.load(os.path.join(base_dir, folder, "data_test.npy")),
                    "labels_test": np.load(os.path.join(base_dir, folder, "labels_test.npy")),
                }
        return None

    def save_phi(self, model_dir, sleep_scores, phi_all_scores):
        data_dir = os.path.join(model_dir, str(sleep_scores))
        os.makedirs(data_dir, exist_ok=True)
        np.save(os.path.join(data_dir, "phi_all_scores.npy"), phi_all_scores)
        print("\rphi data has been saved", end="")

    def load_phi(self, model_dir, sleep_scores):
        data_dir = os.path.join(model_dir, str(sleep_scores))
        phi_path = os.path.join(data_dir, "phi_all_scores.npy")
        if os.path.exists(phi_path):
            print("\rphi data has been loaded", end="")
            return np.load(phi_path)
        return None
