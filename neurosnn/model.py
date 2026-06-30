from datetime import datetime
from typing import Generator, List, Optional

from neurosnn.layer import Layer
from neurosnn.learner import TraceSTDP, TripletSTDP, VogelsSTDP
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
        gabor: bool = False,
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
        self.gabor = gabor
        self._runner: Optional[Runner] = None

    def train(
        self,
        layers: List[Layer],
        learner: "TraceSTDP | TripletSTDP | None" = None,
        inh_learner: "VogelsSTDP | None" = None,
        regularizer=None,
        epochs: int = 1,
        train_weights: bool = True,
        run_full_stream: bool = False,
        save_model: bool = True,
        accuracy_method: str = "pca_lr",
        use_LR: bool = True,
        use_phi: bool = True,
        use_pca: bool = True,
        pca_variance: float = 0.95,
        dt: float = 1.0,
        track_stats: bool = False,
        track_weights: bool = False,
        stat_tracking_frequency: int = 1000,
        profile: bool = False,
        PCA_plot: bool = False,
        gif_pca_plot: bool = False,
        heatmap_plot: bool = False,
        return_spikes: bool = False,
        record_fn_awake_se: "callable | None" = None,
        record_fn_awake_ee: "callable | None" = None,
    ) -> Generator[TrainResult, None, None]:
        """Build the network from layer specs and return a training generator.

        Membrane dynamics are read from the Layer spec.  Learning rule and
        regularization are configured via the learner and regularizer objects.
        If omitted, TraceSTDP and no regularization are used as defaults.

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
            gabor=self.gabor,
        )
        snn.prepare_weights(
            resting_membrane=mem.resting_potential,
            **layer.weights._to_factory_kwargs(),
        )

        if learner is None:
            learner = TraceSTDP()
        if regularizer is None:
            reg_kwargs = dict(
                sleep=False,
                normalize_weights=False,
                sleep_duration=0,
                reg_frequency=1050,
                reg_mode="static",
            )
        else:
            reg_kwargs = regularizer._to_runner_kwargs()

        checkpoint = CheckpointManager()
        logger = HistoryTracker(
            ts_spec=self.ts_spec,
            image_dataset=self.image_dataset,
        )
        self._runner = Runner(model=snn, checkpoint=checkpoint, logger=logger)

        inh_kwargs = inh_learner._to_runner_kwargs() if inh_learner is not None else {}

        return self._runner.train(
            epochs=epochs,
            return_spikes=return_spikes,
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
            run_full_stream=run_full_stream,
            save_model=save_model,
            accuracy_method=accuracy_method,
            use_LR=use_LR,
            use_phi=use_phi,
            use_pca=use_pca,
            pca_variance=pca_variance,
            dt=dt,
            track_stats=track_stats,
            track_weights=track_weights,
            stat_tracking_frequency=stat_tracking_frequency,
            profile=profile,
            PCA_plot=PCA_plot,
            gif_pca_plot=gif_pca_plot,
            heatmap_plot=heatmap_plot,
            **learner._to_runner_kwargs(),
            **reg_kwargs,
            **inh_kwargs,
            record_fn_awake_se=record_fn_awake_se,
            record_fn_awake_ee=record_fn_awake_ee,
        )

    def validate(self, return_spikes: bool = False) -> EvalResult:
        """Validate using the current weights. Call from inside the train() loop."""
        if self._runner is None:
            raise RuntimeError("call train() before validate()")
        return self._runner.validate(return_spikes=return_spikes)

    def test(self, return_spikes: bool = False) -> EvalResult:
        """Run the test partition. Call after training is complete."""
        if self._runner is None:
            raise RuntimeError("call train() before test()")
        return self._runner.test(return_spikes=return_spikes)
