from datetime import datetime
from typing import Generator, List, Optional

from neurosnn.layer import Layer
from neurosnn.results import EvalResult, TrainResult
from neurosnn._network.io import CheckpointManager
from neurosnn._network.model import SNNModel
from neurosnn._network.runner import Runner
from neurosnn._utils.logger import HistoryTracker


class Model:
    """User-facing entry point for bio-inspired SNN training.

    Data configuration is fixed at construction time. Layer architecture
    and training hyperparameters are passed to train().

    Example
    -------
    model = Model(input_size=225, num_steps=350, all_images_train=1000, ...)

    for result in model.train(layers=[layer], epochs=10, train_weights=True):
        print(result.epoch, result.batch, result.accuracy)
        if result.batch % 20 == 0:
            val = model.validate()
            print(val.accuracy)

    test_result = model.test()
    """

    def __init__(
        self,
        input_size: int = 225,
        classes: list = None,
        random_state: int = 0,
        ts_spec: str = None,
        num_steps: int = 350,
        all_images_train: int = 1000,
        batch_image_train: int = 100,
        all_images_test: int = 200,
        batch_image_test: int = 100,
        all_images_val: int = 200,
        batch_image_val: int = 100,
        image_dataset: str = "mnist",
        max_rate_hz: float = 90.0,
        gain: float = 1.0,
    ):
        self.input_size = input_size
        self.classes = classes if classes is not None else list(range(10))
        self.random_state = random_state
        self.ts_spec = ts_spec or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.num_steps = num_steps
        self.all_images_train = all_images_train
        self.batch_image_train = batch_image_train
        self.all_images_test = all_images_test
        self.batch_image_test = batch_image_test
        self.all_images_val = all_images_val
        self.batch_image_val = batch_image_val
        self.image_dataset = image_dataset
        self.max_rate_hz = max_rate_hz
        self.gain = gain

        self._runner: Optional[Runner] = None

    def train(
        self,
        layers: List[Layer],
        epochs: int = 1,
        train_weights: bool = True,
        save_model: bool = True,
        accuracy_method: str = "pca_lr",
        use_LR: bool = True,
        use_phi: bool = True,
        use_pca: bool = True,
        pca_variance: float = 0.95,
        learning_rate: float = 0.0008,
        tau_trace: int = 25,
        A_plus: float = 0.5,
        A_minus: float = 0.5,
        tau_LTP: float = 10.0,
        tau_LTD: float = 10.0,
        w_max: float = 10.0,
        mu_weight: float = 0.6,
        dt: float = 1.0,
        min_weight_exc: float = 0.01,
        max_weight_exc: float = 25.0,
        min_weight_inh: float = -25.0,
        max_weight_inh: float = -0.01,
        clip_weights: bool = False,
        normalize_weights: bool = False,
        sleep: bool = False,
        reg_mode: str = "static",
        reg_frequency: int = 1050,
        sleep_duration: int = 300,
        update_weights_freq: int = 100,
        track_stats: bool = False,
        track_weights: bool = False,
        stat_tracking_frequency: int = 1000,
        profile: bool = False,
        PCA_plot: bool = False,
        gif_pca_plot: bool = False,
        heatmap_plot: bool = False,
    ) -> Generator[TrainResult, None, None]:
        """Build the network from layer specs and return a training generator.

        All membrane dynamics parameters (tau, resistance, thresholds, noise,
        adaptation) are read directly from the Layer spec and do not need to
        be repeated here.

        Only a single Layer is currently supported.
        """
        if len(layers) != 1:
            raise NotImplementedError(
                "Multi-layer not yet supported; pass exactly one Layer."
            )

        layer = layers[0]
        mem = layer.membrane

        snn = SNNModel(
            N_exc=layer.N_exc,
            N_inh=layer.N_inh,
            N_x=self.input_size,
            classes=self.classes,
            random_state=self.random_state,
            ts_spec=self.ts_spec,
        )
        snn.prepare_data(
            num_steps=self.num_steps,
            all_images_train=self.all_images_train,
            batch_image_train=self.batch_image_train,
            all_images_test=self.all_images_test,
            batch_image_test=self.batch_image_test,
            all_images_val=self.all_images_val,
            batch_image_val=self.batch_image_val,
            image_dataset=self.image_dataset,
            max_rate_hz=self.max_rate_hz,
            gain=self.gain,
        )
        snn.prepare_weights(
            resting_membrane=mem.resting_potential,
            **layer.weights._to_factory_kwargs(),
        )

        checkpoint = CheckpointManager()
        logger = HistoryTracker(
            ts_spec=self.ts_spec,
            image_dataset=self.image_dataset,
        )
        self._runner = Runner(model=snn, checkpoint=checkpoint, logger=logger)

        return self._runner.train(
            epochs=epochs,
            tau_m_exc=mem.tau_m_exc,
            tau_m_inh=mem.tau_m_inh,
            membrane_resistance_exc=mem.membrane_resistance_exc,
            membrane_resistance_inh=mem.membrane_resistance_inh,
            tau_syn_exc=mem.tau_syn_exc,
            tau_syn_inh=mem.tau_syn_inh,
            reset_potential=mem.reset_potential,
            spike_threshold_default=mem.spike_threshold,
            min_mp=mem.min_mp,
            max_mp=mem.max_mp,
            var_noise=mem.var_noise,
            mean_noise=mem.mean_noise,
            spike_adaption=mem.spike_adaptation,
            tau_adaption=mem.tau_adaptation,
            delta_adaption=mem.delta_adaptation,
            train_weights=train_weights,
            save_model=save_model,
            accuracy_method=accuracy_method,
            use_LR=use_LR,
            use_phi=use_phi,
            use_pca=use_pca,
            pca_variance=pca_variance,
            learning_rate=learning_rate,
            tau_trace=tau_trace,
            A_plus=A_plus,
            A_minus=A_minus,
            tau_LTP=tau_LTP,
            tau_LTD=tau_LTD,
            w_max=w_max,
            mu_weight=mu_weight,
            dt=dt,
            min_weight_exc=min_weight_exc,
            max_weight_exc=max_weight_exc,
            min_weight_inh=min_weight_inh,
            max_weight_inh=max_weight_inh,
            clip_weights=clip_weights,
            normalize_weights=normalize_weights,
            sleep=sleep,
            reg_mode=reg_mode,
            reg_frequency=reg_frequency,
            sleep_duration=sleep_duration,
            update_weights_freq=update_weights_freq,
            track_stats=track_stats,
            track_weights=track_weights,
            stat_tracking_frequency=stat_tracking_frequency,
            profile=profile,
            PCA_plot=PCA_plot,
            gif_pca_plot=gif_pca_plot,
            heatmap_plot=heatmap_plot,
        )

    def validate(self) -> EvalResult:
        """Validate using the current weights. Call from inside the train() loop."""
        if self._runner is None:
            raise RuntimeError("call train() before validate()")
        return self._runner.validate()

    def test(self) -> EvalResult:
        """Run the test partition. Call after training is complete."""
        if self._runner is None:
            raise RuntimeError("call train() before test()")
        return self._runner.test()
