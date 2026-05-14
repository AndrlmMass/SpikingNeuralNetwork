import numpy as np
from datetime import datetime

from neurosnn._network.init_weights import WeightFactory
from neurosnn._data.get_data import ImageDataStreamer


class SNNModel:
    def __init__(
        self,
        N_exc: int = 200,
        N_inh: int = 50,
        N_x: int = 225,
        classes: list = None,
        random_state: int = 0,
        ts_spec: str = None,
    ):
        if classes is None:
            classes = list(range(10))
        self.N_exc = N_exc
        self.N_inh = N_inh
        self.N_x = N_x
        self.pixel_size = int(np.sqrt(N_x))
        self.rng = np.random.default_rng(random_state)
        self.N_classes = len(classes)
        self.classes = classes
        self.st = N_x
        self.ex = self.st + N_exc
        self.ih = self.ex + N_inh
        self.N = N_exc + N_inh + N_x
        self.ts_spec = ts_spec
        self.ts = datetime.now().strftime("%Y.%m.%d")
        self.weights = None
        self._factory = None
        self.resting_potential = -70.0
        self.w_dense_ee = self.w_dense_se = self.w_dense_ei = self.w_dense_ie = None
        self.se_weights = self.ee_weights = self.ei_weights = self.ie_weights = None
        self.image_streamer = None
        self.image_dataset = None
        self.num_steps = None
        self.batch_train = self.batch_test = self.batch_val = None
        self.all_train = self.all_test = self.all_val = None
        self.n_train_batches = self.n_val_batches = self.n_test_batches = None

    def prepare_data(
        self,
        num_steps: int,
        all_images_train: int,
        batch_image_train: int,
        all_images_test: int,
        batch_image_test: int,
        all_images_val: int,
        batch_image_val: int,
        image_dataset: str = "mnist",
        max_rate_hz: float = 90.0,
        gain: float = 1.0,
        **kwargs,
    ):
        self.num_steps = num_steps
        self.batch_train = batch_image_train
        self.batch_test = batch_image_test
        self.batch_val = batch_image_val
        self.all_train = all_images_train
        self.all_test = all_images_test
        self.all_val = all_images_val
        self.image_dataset = image_dataset
        self.n_train_batches = max(1, all_images_train // batch_image_train)
        self.n_val_batches = max(1, all_images_val // batch_image_val)
        self.n_test_batches = max(1, all_images_test // batch_image_test)
        self.image_streamer = ImageDataStreamer(
            data_dir="data",
            pixel_size=self.pixel_size,
            num_steps=num_steps,
            max_rate_hz=max_rate_hz,
            gain=gain,
            train_count=all_images_train,
            val_count=all_images_val,
            test_count=all_images_test,
            dataset=image_dataset,
        )

    def prepare_weights(
        self,
        w_dense_ee: float = 0.01,
        w_dense_se: float = 0.05,
        w_dense_ei: float = 0.05,
        w_dense_ie: float = 0.05,
        se_weights: float = 0.1,
        ee_weights: float = 0.3,
        ei_weights: float = 0.3,
        ie_weights: float = -0.2,
        random_weights: bool = False,
        plot_weights: bool = False,
        resting_membrane: float = -70.0,
        **kwargs,
    ):
        self.resting_potential = resting_membrane
        self._factory = WeightFactory(
            N=self.N,
            N_x=self.N_x,
            N_exc=self.N_exc,
            N_inh=self.N_inh,
            rng=self.rng,
            w_dense_se=w_dense_se,
            w_dense_ee=w_dense_ee,
            w_dense_ei=w_dense_ei,
            w_dense_ie=w_dense_ie,
            se_weights=se_weights,
            ee_weights=ee_weights,
            ei_weights=ei_weights,
            ie_weights=ie_weights,
            random_weights=random_weights,
        )
        self.weights = self._factory.build()
        self.w_dense_ee = w_dense_ee
        self.w_dense_ei = w_dense_ei
        self.w_dense_se = w_dense_se
        self.w_dense_ie = w_dense_ie
        self.se_weights = se_weights
        self.ee_weights = ee_weights
        self.ei_weights = ei_weights
        self.ie_weights = ie_weights
        if plot_weights:
            self._factory.plot(plot_weights=True)

    def sparse_indices(self) -> dict:
        return self._factory.sparse_indices(self.weights)

    def initial_sums(self, reg_mode: str) -> tuple:
        return self._factory.initial_sums(self.weights, reg_mode)

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
