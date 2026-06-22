import gc
import os
from copy import deepcopy
from typing import Generator, Optional

import numpy as np
from tqdm import tqdm

from neurosnn.results import EvalResult, TrainResult
from neurosnn._core.neurons import seed_numba_rng
from neurosnn._core.trainer import Trainer
from neurosnn._evaluation.evaluation import Evaluator
from neurosnn._network.io import CheckpointManager
from neurosnn._network.model import SNNModel
from neurosnn._plot.training import PCAScatterDisplay, plot_accuracy
from neurosnn._utils.logger import HistoryTracker
from neurosnn._utils.performance import start_plot_worker, stop_plot_worker


class Runner:
    """Runs training, validation, and test loops for an SNNModel.

    Persistent neural state (membrane potential, thresholds, synaptic currents)
    carries across train() calls so epochs build on each other.

    Usage
    -----
    for result in runner.train(epochs=10, train_weights=True, ...):
        print(result.epoch, result.batch, result.accuracy)
        if result.batch % 5 == 0:
            val = runner.validate()
            print(val.accuracy)

    test_result = runner.test()
    """

    def __init__(
        self,
        model: SNNModel,
        checkpoint: CheckpointManager,
        logger: HistoryTracker,
    ):
        self.model = model
        self.checkpoint = checkpoint
        self.logger = logger
        self._state = None

        self._trainer: Optional[Trainer] = None
        self._evaluator: Optional[Evaluator] = None
        self._evaluator_fitted: bool = False

        self._accuracy_method: Optional[str] = None
        self._spike_threshold_default: Optional[float] = None
        self._PCA_plot: bool = False
        self._save_model: bool = True
        self._best_val: float = 0.0
        self._pca_plotter: Optional[PCAScatterDisplay] = None
        self._validate_call_count: int = 0
        self._global_batch: int = 0

    def train(
        self,
        epochs: int = 1,
        return_spikes: bool = False,
        train_weights: bool = False,
        tau_trace: int = 25,
        learning_rate: float = 0.0008,
        var_noise: float = 1.0,
        min_weight_inh: float = -25.0,
        max_weight_inh: float = -0.01,
        max_weight_exc: float = 25.0,
        min_weight_exc: float = 0.01,
        spike_threshold_default: float = -55.0,
        min_mp: float = -100.0,
        max_mp: float = 40.0,
        sleep: bool = False,
        normalize_weights: bool = False,
        save_model: bool = True,
        spike_adaption: bool = True,
        delta_adaption: float = 1.0,
        tau_adaption: float = 100.0,
        clip_weights: bool = False,
        A_plus: float = 0.5,
        A_minus: float = 0.5,
        tau_LTD: float = 10.0,
        tau_LTP: float = 10.0,
        w_max: float = 10.0,
        dt: float = 1.0,
        membrane_resistance_exc: float = 30.0,
        membrane_resistance_inh: float = 30.0,
        reset_potential: float = -80.0,
        pca_variance: float = 0.95,
        mean_noise: float = 0.0,
        mu_weight: float = 0.6,
        tau_syn_exc: float = 30.0,
        tau_syn_inh: float = 30.0,
        tau_m_exc: float = 30.0,
        tau_m_inh: float = 30.0,
        accuracy_method: str = "top",
        use_LR: bool = True,
        use_phi: bool = True,
        use_pca: bool = True,
        reg_mode: str = "static",
        x_tar_mode: str = "mean",
        x_tar_pct_se: float = 60.0,
        x_tar_pct_ee: float = 30.0,
        track_stats: bool = False,
        track_weights: bool = False,
        PCA_plot: bool = True,
        gif_pca_plot: bool = True,
        heatmap_plot: bool = False,
        profile: bool = False,
        reg_frequency: int = 1050,
        sleep_duration: int = 300,
        stat_tracking_frequency: int = 1000,
        update_weights_freq: int = 100,
        record_fn_se: "callable | None" = None,
        record_fn_ee: "callable | None" = None,
        record_fn_awake_se: "callable | None" = None,
        record_fn_awake_ee: "callable | None" = None,
    ) -> Generator[TrainResult, None, None]:
        if accuracy_method != "pca_lr" and PCA_plot:
            raise ValueError("PCA_plot requires accuracy_method='pca_lr'")

        # Seed Numba's internal per-thread RNG so membrane noise is reproducible.
        # This must be called inside a Python function before any @njit code runs;
        # it is separate from numpy's global RNG and model.rng.
        seed_numba_rng(self.model.random_state)

        model = self.model
        sparse = model.sparse_indices()
        initial_sums_se, initial_sums_ee = model.initial_sums(reg_mode)

        self._accuracy_method = accuracy_method
        self._spike_threshold_default = spike_threshold_default
        self._PCA_plot = PCA_plot
        self._save_model = save_model
        self._best_val = 0.0
        self._pca_plotter = None
        self._evaluator_fitted = False
        self._validate_call_count = 0
        self._global_batch = 0

        self._trainer = Trainer(
            resting_potential=model.resting_potential,
            membrane_resistance_exc=membrane_resistance_exc,
            membrane_resistance_inh=membrane_resistance_inh,
            update_weights_freq=update_weights_freq,
            reg_frequency=reg_frequency,
            sleep_duration=sleep_duration,
            stat_tracking_frequency=stat_tracking_frequency,
            track_stats=track_stats,
            track_weights=track_weights,
            min_weight_exc=min_weight_exc,
            max_weight_exc=max_weight_exc,
            min_weight_inh=min_weight_inh,
            max_weight_inh=max_weight_inh,
            train_weights=train_weights,
            N_x=model.N_x,
            N_inh=model.N_inh,
            N_exc=model.N_exc,
            learning_rate=learning_rate,
            tau_LTP=tau_LTP,
            tau_LTD=tau_LTD,
            max_mp=max_mp,
            min_mp=min_mp,
            w_target_se=model.se_weights,
            w_target_ee=model.ee_weights,
            w_max=w_max,
            clip_weights=clip_weights,
            dt=dt,
            run=model.ts_spec,
            A_plus=A_plus,
            A_minus=A_minus,
            tau_m_exc=tau_m_exc,
            tau_m_inh=tau_m_inh,
            sleep=sleep,
            spike_adaption=spike_adaption,
            tau_adaption=tau_adaption,
            delta_adaption=delta_adaption,
            spike_threshold_default=spike_threshold_default,
            save_plots=heatmap_plot,
            reset_potential=reset_potential,
            initial_sums_se=initial_sums_se,
            initial_sums_ee=initial_sums_ee,
            dataset=model.image_dataset,
            plot_iterations=model.num_steps,
            tau_trace=tau_trace,
            tau_syn_exc=tau_syn_exc,
            tau_syn_inh=tau_syn_inh,
            mean_noise=mean_noise,
            var_noise=var_noise,
            mu_weight=mu_weight,
            st=model.st,
            ex=model.ex,
            ih=model.ih,
            mp_new=np.zeros(model.N_exc + model.N_inh),
            time_per_item=model.num_steps,
            normalize_weights=normalize_weights,
            nonzero_pre_idx=sparse["nonzero_pre_idx"],
            reg_mode=reg_mode,
            x_tar_mode=x_tar_mode,
            x_tar_pct_se=x_tar_pct_se,
            x_tar_pct_ee=x_tar_pct_ee,
            nz_rows_se=sparse["nz_rows_se"],
            nz_cols_se=sparse["nz_cols_se"],
            nz_rows_ee=sparse["nz_rows_ee"],
            nz_cols_ee=sparse["nz_cols_ee"],
            nz_cols_exc=sparse["nz_cols_exc"],
            nz_rows_exc=sparse["nz_rows_exc"],
            record_fn_se=record_fn_se,
            record_fn_ee=record_fn_ee,
            record_fn_awake_se=record_fn_awake_se,
            record_fn_awake_ee=record_fn_awake_ee,
        )

        self._evaluator = Evaluator(
            xp_var_or_comps=pca_variance,
            num_classes=model.N_classes,
            do_phi=use_phi,
            do_LR=use_LR,
            do_pca=use_pca,
            seed=model.random_state,
        )

        self.logger.log_config(dict(
            ts_spec=model.ts_spec,
            seed=model.random_state,
            dataset=model.image_dataset,
            epochs=epochs,
            train_weights=train_weights,
            sleep=sleep,
            normalize_weights=normalize_weights,
            sleep_duration=sleep_duration,
            reg_mode=reg_mode,
            reg_frequency=reg_frequency,
            var_noise=var_noise,
            mean_noise=mean_noise,
            learning_rate=learning_rate,
            A_plus=A_plus,
            A_minus=A_minus,
            tau_LTP=tau_LTP,
            tau_LTD=tau_LTD,
            tau_m_exc=tau_m_exc,
            tau_m_inh=tau_m_inh,
            spike_threshold_default=spike_threshold_default,
            N_exc=model.N_exc,
            N_inh=model.N_inh,
            N_x=model.N_x,
            num_steps=model.num_steps,
            all_images_train=model.all_train,
            all_images_val=model.all_val,
            all_images_test=model.all_test,
        ))

        if self._state is None:
            self._state = self._init_state(spike_threshold_default)

        n_train = model.n_train_batches

        pbar = tqdm(
            total=n_train * epochs if train_weights else epochs,
            desc="Training" if train_weights else "Test-only",
            unit="batch",
            ncols=80,
            bar_format="{desc} [{bar}] ETA: {remaining} |{postfix}",
        )

        if heatmap_plot:
            start_plot_worker()

        try:
            for epoch in range(epochs):
                model.image_streamer.reset_partition("train")

                for b in range(n_train if train_weights else 1):
                    data_train, labels_train = model.image_streamer.get_batch(
                        0, model.batch_train, "train"
                    )
                    if data_train is None:
                        model.image_streamer.reset_partition("train")
                        data_train, labels_train = model.image_streamer.get_batch(
                            0, model.batch_train, "train"
                        )
                    if data_train is None:
                        break

                    T_train = data_train.shape[0]
                    spikes_train = np.zeros((T_train, model.N), dtype=np.int8)
                    spikes_train[:, : model.st] = data_train
                    del data_train

                    if profile:
                        import cProfile
                        profiler = cProfile.Profile()
                        profiler.enable()

                    (
                        model.weights,
                        spikes_tr_out,
                        self._state["mp"],
                        self._state["spike_threshold"],
                        labels_tr_out,
                        self._state["I_syn_exc"],
                        self._state["I_syn_inh"],
                        self._state["a"],
                        self._state["spike_trace"],
                        batch_stats,
                    ) = self._trainer.step(
                        weights=model.weights,
                        mp=self._state["mp"],
                        spikes=spikes_train,
                        spike_labels=labels_train,
                        spike_trace=self._state["spike_trace"],
                        training_mode="train",
                        spike_threshold=self._state["spike_threshold"],
                        I_syn_exc=self._state["I_syn_exc"],
                        I_syn_inh=self._state["I_syn_inh"],
                        a=self._state["a"],
                    )

                    if profile:
                        profiler.disable()
                        prof_dir = os.path.join("profile", model.image_dataset, model.ts)
                        os.makedirs(prof_dir, exist_ok=True)
                        profiler.dump_stats(
                            os.path.join(prof_dir, f"{model.ts_spec}.prof")
                        )

                    del spikes_train, labels_train

                    tr_accuracy = tr_phi = None
                    tr_spikes = None
                    if accuracy_method == "pca_lr" and spikes_tr_out is not None:
                        X_tr, y_tr = model.spikes_per_item(
                            spikes_tr_out[:, model.st : model.ex], labels_tr_out
                        )
                        if X_tr.size > 0:
                            self._evaluator.fit(X=X_tr, Y=y_tr)
                            self._evaluator_fitted = True
                            if self._global_batch == 0 and PCA_plot:
                                self._pca_plotter = PCAScatterDisplay(
                                    scaler=deepcopy(self._evaluator.scaler),
                                    pca=deepcopy(self._evaluator.pca),
                                )
                            tr_accuracy, tr_phi = self._evaluator.score(
                                X=X_tr, Y=y_tr
                            )
                            self.logger._record_accuracy(
                                "train", tr_accuracy,
                                epoch=self._global_batch + 1,
                                method="pca_lr",
                            )
                            self.logger._record_phi(
                                "train", tr_phi, epoch=self._global_batch + 1
                            )
                            if return_spikes:
                                tr_spikes = X_tr

                    del spikes_tr_out, labels_tr_out
                    gc.collect()

                    self._global_batch += 1
                    pbar.set_description(f"Epoch {epoch + 1}/{epochs} batch {b + 1}/{n_train}")
                    pbar.set_postfix(
                        acc=f"{tr_accuracy:.3f}" if tr_accuracy is not None else "N/A",
                        phi=f"{tr_phi:.2f}" if tr_phi is not None else "N/A",
                    )
                    pbar.update(1)

                    yield TrainResult(
                        epoch=epoch,
                        batch=b,
                        weights=model.weights,
                        accuracy=tr_accuracy,
                        phi=tr_phi,
                        spikes=tr_spikes,
                        stats=batch_stats,
                    )

        finally:
            pbar.close()
            if heatmap_plot:
                stop_plot_worker()
                from datetime import datetime
                from neurosnn._plot.spikes import gif_spike_rate_by_label
                ts = datetime.now().strftime("%Y.%m.%d")
                frame_folder = os.path.join(
                    "plots", "spikes", model.image_dataset, "all", ts, str(model.ts_spec)
                )
                gif_out = os.path.join(frame_folder, "evolution.gif")
                gif_spike_rate_by_label(frame_folder, output_filename=gif_out)
            if gif_pca_plot and PCA_plot and self._pca_plotter is not None:
                from neurosnn._plot.spikes import GenerateGif
                gif = GenerateGif(
                    frame_folder=self._pca_plotter.dir,
                    output_filename=f"{model.ts_spec}.gif",
                )
                gif.create()

    def validate(self, return_spikes: bool = False) -> EvalResult:
        if self._trainer is None:
            raise RuntimeError("call train() before validate()")
        if not self._evaluator_fitted:
            return EvalResult(accuracy=None, phi=None, split="val")

        acc, phi, val_spikes = self._validate(
            trainer=self._trainer,
            evaluator=self._evaluator,
            pca_plotter=self._pca_plotter if self._PCA_plot else None,
            spike_threshold_default=self._spike_threshold_default,
            accuracy_method=self._accuracy_method,
            PCA_plot=self._PCA_plot,
            validate_call_idx=self._validate_call_count,
            return_spikes=return_spikes,
        )
        self._validate_call_count += 1

        if acc is not None:
            self.logger._record_accuracy(
                "val", acc, epoch=self._global_batch, method="pca_lr"
            )
            self.logger._record_phi("val", phi, epoch=self._global_batch)
            if self._accuracy_method == "pca_lr" and self.logger._acc_log_file:
                plot_accuracy(
                    self.logger,
                    wta=False,
                    mcc=False,
                    phi=True,
                    pca=True,
                    acc_log_file=self.logger._acc_log_file,
                    read_jsonl=self.logger._read_jsonl,
                )
            if self._save_model and acc > self._best_val:
                self._best_val = acc
                self.checkpoint.save_model(self.model.weights, {}, run_id=self.model.ts_spec)

        return EvalResult(accuracy=acc, phi=phi, split="val", spikes=val_spikes)

    def test(self, return_spikes: bool = False) -> EvalResult:
        if self._trainer is None:
            raise RuntimeError("call train() before test()")

        acc, phi, te_spikes = self._test(
            trainer=self._trainer,
            evaluator=self._evaluator,
            spike_threshold_default=self._spike_threshold_default,
            accuracy_method=self._accuracy_method,
            return_spikes=return_spikes,
        )
        return EvalResult(accuracy=acc, phi=phi, split="test", spikes=te_spikes)

    def _init_state(self, spike_threshold_default: float) -> dict:
        m = self.model
        mp = np.full(m.N - m.st, m.resting_potential)
        return {
            "mp": mp,
            "I_syn_exc": np.zeros(m.N_exc),
            "I_syn_inh": np.zeros(m.N_inh),
            "a": np.zeros(m.N_exc + m.N_inh),
            "spike_trace": np.zeros(m.N - m.N_inh),
            "spike_threshold": np.full(m.N - m.st, spike_threshold_default, dtype=float),
        }

    def _validate(
        self,
        trainer: Trainer,
        evaluator: Evaluator,
        pca_plotter,
        spike_threshold_default: float,
        accuracy_method: str,
        PCA_plot: bool,
        validate_call_idx: int = 0,
        return_spikes: bool = False,
    ):
        model = self.model
        acc_sum = phi_sum = 0.0
        count = 0
        spikes_out_all = []

        model.image_streamer.reset_partition("val")

        for _ in range(model.n_val_batches):
            data_val, labels_val = model.image_streamer.get_batch(
                0, model.batch_val, "val"
            )
            if data_val is None:
                break

            T_val = data_val.shape[0]
            spikes_val = np.zeros((T_val, model.N), dtype=np.int8)
            spikes_val[:, : model.st] = data_val[:, : model.st]
            del data_val

            mp_val = np.full(model.N - model.st, model.resting_potential)
            spike_trace_val = np.zeros(model.N - model.N_inh)
            a_val = np.zeros(model.N_exc + model.N_inh)
            I_syn_exc_val = np.zeros(model.N_exc)
            I_syn_inh_val = np.zeros(model.N_inh)
            spike_threshold_val = np.full(
                model.N_exc + model.N_inh, spike_threshold_default, dtype=float
            )
            (
                _,
                spikes_val_out,
                _,
                _,
                labels_val_out,
                _,
                _,
                _,
                _,
                _,
            ) = trainer.step(
                weights=model.weights,
                mp=mp_val,
                spikes=spikes_val,
                spike_labels=labels_val,
                spike_trace=spike_trace_val,
                training_mode="val",
                spike_threshold=spike_threshold_val,
                I_syn_exc=I_syn_exc_val,
                I_syn_inh=I_syn_inh_val,
                a=a_val,
            )

            del spikes_val, labels_val
            gc.collect()

            if accuracy_method == "pca_lr" and spikes_val_out is not None:
                X_val, y_val = model.spikes_per_item(
                    spikes_val_out[:, model.st : model.ex], labels_val_out
                )
                if X_val.size > 0:
                    acc, phi = evaluator.score(X=X_val, Y=y_val)
                    if PCA_plot and pca_plotter is not None:
                        pca_plotter.plot(
                            X=X_val, Y=y_val,
                            epoch=validate_call_idx,
                            run=model.ts_spec,
                            dataset=model.image_dataset,
                            phi=phi,
                        )
                    acc_sum += acc
                    phi_sum += phi
                    count += 1
                    if return_spikes:
                        spikes_out_all.append(X_val)

            del spikes_val_out, labels_val_out
            gc.collect()

        if count == 0:
            return None, None, None
        collected = np.concatenate(spikes_out_all, axis=0) if return_spikes and spikes_out_all else None
        return acc_sum / count, phi_sum / count, collected

    def _test(
        self,
        trainer: Trainer,
        evaluator: Evaluator,
        spike_threshold_default: float,
        accuracy_method: str,
        return_spikes: bool = False,
    ):
        model = self.model
        model.image_streamer.reset_partition("test")
        acc_sum = phi_sum = 0.0
        count = 0
        spikes_out_all = []

        for _ in range(model.n_test_batches):
            data_test, labels_test = model.image_streamer.get_batch(
                0, model.batch_test, "test"
            )
            if data_test is None:
                break

            T_test = data_test.shape[0]
            spikes_test = np.zeros((T_test, model.N), dtype=np.int8)
            spikes_test[:, : model.st] = data_test
            del data_test

            mp_test = np.full(model.N - model.st, model.resting_potential)
            a_test = np.zeros(model.N_exc + model.N_inh)
            I_syn_exc_test = np.zeros(model.N_exc)
            I_syn_inh_test = np.zeros(model.N_inh)
            spike_threshold_test = np.full(
                model.N_exc + model.N_inh, spike_threshold_default, dtype=float
            )
            spike_trace_te = np.zeros(model.N - model.N_inh)
            (
                _,
                spikes_te_out,
                _,
                _,
                labels_te_out,
                _,
                _,
                _,
                _,
                _,
            ) = trainer.step(
                weights=model.weights,
                mp=mp_test,
                spikes=spikes_test,
                spike_labels=labels_test,
                spike_trace=spike_trace_te,
                training_mode="test",
                spike_threshold=spike_threshold_test,
                I_syn_exc=I_syn_exc_test,
                I_syn_inh=I_syn_inh_test,
                a=a_test,
            )

            del spikes_test, labels_test
            gc.collect()

            if accuracy_method == "pca_lr" and spikes_te_out is not None:
                X_te, y_te = model.spikes_per_item(
                    spikes_te_out[:, model.st : model.ex], labels_te_out
                )
                if X_te.size > 0:
                    acc, phi = evaluator.score(X=X_te, Y=y_te)
                    acc_sum += acc
                    phi_sum += phi
                    count += 1
                    if return_spikes:
                        spikes_out_all.append(X_te)

            del spikes_te_out, labels_te_out
            gc.collect()

        if count == 0:
            return None, None, None

        final_acc = acc_sum / count
        final_phi = phi_sum / count
        print(f"\nTest accuracy (PCA+LR): {final_acc:.4f}")
        print(f"Test phi: {final_phi:.4f}")
        self.logger._record_accuracy("test", final_acc, epoch=None, method="pca_lr")
        self.logger._record_phi("test", final_phi, epoch=None)
        collected = np.concatenate(spikes_out_all, axis=0) if return_spikes and spikes_out_all else None
        return final_acc, final_phi, collected
