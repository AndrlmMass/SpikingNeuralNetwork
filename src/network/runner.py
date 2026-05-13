import gc
import os

import numpy as np
from copy import deepcopy
from tqdm import tqdm

from src.core.trainer import Trainer
from src.evaluation.evaluation import Evaluator
from src.network.io import CheckpointManager
from src.network.model import SNNModel
from src.plot.training import PCAScatterDisplay, plot_accuracy
from src.utils.logger import HistoryTracker


class Runner:
    """Runs training, validation, and test loops for an SNNModel.

    Persistent neural state (membrane potential, thresholds, synaptic currents)
    carries across calls to run() so epochs build on each other.
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
        self._state = None  # initialized on first run()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
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
    ):
        if accuracy_method != "pca_lr" and PCA_plot:
            raise ValueError("PCA_plot requires accuracy_method='pca_lr'")

        model = self.model
        sparse = model.sparse_indices()
        initial_sums_se, initial_sums_ee = model.initial_sums(reg_mode)

        # Persist x_tar between calls (but recomputed inside step anyway)
        x_tar_se = np.zeros(model.N_x)
        x_tar_ee = np.zeros(model.N_exc)

        trainer = Trainer(
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
            nz_rows_se=sparse["nz_rows_se"],
            nz_cols_se=sparse["nz_cols_se"],
            nz_rows_ee=sparse["nz_rows_ee"],
            nz_cols_ee=sparse["nz_cols_ee"],
            nz_cols_exc=sparse["nz_cols_exc"],
            nz_rows_exc=sparse["nz_rows_exc"],
            x_tar_se=x_tar_se,
            x_tar_ee=x_tar_ee,
        )

        evaluator = Evaluator(
            xp_var_or_comps=pca_variance,
            num_classes=model.N_classes,
            do_phi=use_phi,
            do_LR=use_LR,
            do_pca=use_pca,
        )

        if self._state is None:
            self._state = self._init_state(spike_threshold_default)

        n_train = model.n_train_batches
        n_val = model.n_val_batches

        best_val = 0.0
        evaluator_fitted = False
        pca_plotter = None

        pbar = tqdm(
            total=n_train if train_weights else 1,
            desc="Test-only" if not train_weights else f"Batch 0/{n_train}",
            unit="it",
            ncols=80,
            bar_format="{desc} [{bar}] ETA: {remaining} |{postfix}",
        )

        model.image_streamer.reset_partition("val")

        for b in range(n_train if train_weights else 1):
            # --- train pass ---
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
                x_tar_se,
                x_tar_ee,
            ) = trainer.step(
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
                x_tar_se=x_tar_se,
                x_tar_ee=x_tar_ee,
            )

            if profile:
                profiler.disable()
                prof_dir = os.path.join("profile", model.image_dataset, model.ts)
                os.makedirs(prof_dir, exist_ok=True)
                profiler.dump_stats(os.path.join(prof_dir, f"{model.ts_spec}.prof"))

            del spikes_train, labels_train

            # --- training accuracy ---
            final_tr_acc = final_tr_phi = None
            if accuracy_method == "pca_lr" and spikes_tr_out is not None:
                X_tr, y_tr = model.spikes_per_item(
                    spikes_tr_out[:, model.st : model.ex], labels_tr_out
                )
                if X_tr.size > 0:
                    evaluator.fit(X=X_tr, Y=y_tr)
                    evaluator_fitted = True
                    if b == 0 and PCA_plot:
                        pca_plotter = PCAScatterDisplay(
                            scaler=deepcopy(evaluator.scaler),
                            pca=deepcopy(evaluator.pca),
                        )
                    final_tr_acc, final_tr_phi = evaluator.score(X=X_tr, Y=y_tr)
                    self.logger._record_accuracy("train", final_tr_acc, epoch=b + 1, method="pca_lr")
                    self.logger._record_phi("train", final_tr_phi, epoch=b + 1)

            del spikes_tr_out, labels_tr_out
            gc.collect()

            # --- validation pass ---
            final_val_acc = final_val_phi = None
            if evaluator_fitted:
                final_val_acc, final_val_phi = self._validate(
                    batch_idx=b,
                    trainer=trainer,
                    evaluator=evaluator,
                    pca_plotter=pca_plotter if PCA_plot else None,
                    spike_threshold_default=spike_threshold_default,
                    x_tar_se=x_tar_se,
                    x_tar_ee=x_tar_ee,
                    accuracy_method=accuracy_method,
                    PCA_plot=PCA_plot,
                )
                if final_val_acc is not None:
                    self.logger._record_accuracy("val", final_val_acc, epoch=b + 1, method="pca_lr")
                    self.logger._record_phi("val", final_val_phi, epoch=b + 1)
                    if accuracy_method == "pca_lr":
                        plot_accuracy(mcc=False, pca=True, wta=False, phi=True)

                    if final_val_acc > best_val:
                        best_val = final_val_acc
                        if save_model:
                            self.checkpoint.save_model(model.weights, {})

            pbar.set_description(f"Batch {b+1}/{n_train}")
            acc_str = f"{final_val_acc:.3f}" if final_val_acc is not None else "N/A"
            phi_str = f"{final_val_phi:.2f}" if final_val_phi is not None else "N/A"
            pbar.set_postfix(acc=acc_str, phi=phi_str)
            pbar.update(1)

        pbar.close()

        if gif_pca_plot and PCA_plot and pca_plotter is not None:
            from src.plot.spikes import GenerateGif
            gif = GenerateGif(
                frame_folder=pca_plotter.dir,
                output_filename=f"{model.ts_spec}.gif",
            )
            gif.create()

        # --- test pass ---
        self._test(
            trainer=trainer,
            evaluator=evaluator,
            spike_threshold_default=spike_threshold_default,
            x_tar_se=x_tar_se,
            x_tar_ee=x_tar_ee,
            accuracy_method=accuracy_method,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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
        batch_idx,
        trainer,
        evaluator,
        pca_plotter,
        spike_threshold_default,
        x_tar_se,
        x_tar_ee,
        accuracy_method,
        PCA_plot,
    ):
        model = self.model
        acc_sum = phi_sum = 0.0
        count = 0

        model.image_streamer.reset_partition("val")

        for _ in range(model.n_val_batches):
            data_val, labels_val = model.image_streamer.get_batch(0, model.batch_val, "val")
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
            x_tar_se_val = np.zeros(model.N_x)
            x_tar_ee_val = np.zeros(model.N_exc)

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
                x_tar_se=x_tar_se_val,
                x_tar_ee=x_tar_ee_val,
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
                            X=X_val,
                            Y=y_val,
                            epoch=batch_idx,
                            run=model.ts_spec,
                            dataset=model.image_dataset,
                            phi=phi,
                        )
                    acc_sum += acc
                    phi_sum += phi
                    count += 1

            del spikes_val_out, labels_val_out
            gc.collect()

        if count == 0:
            return None, None
        return acc_sum / count, phi_sum / count

    def _test(self, trainer, evaluator, spike_threshold_default, x_tar_se, x_tar_ee, accuracy_method):
        model = self.model
        model.image_streamer.reset_partition("test")
        acc_sum = phi_sum = 0.0
        count = 0

        for _ in range(model.n_test_batches):
            data_test, labels_test = model.image_streamer.get_batch(0, model.batch_test, "test")
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
            x_tar_se_test = np.zeros(model.N_x)
            x_tar_ee_test = np.zeros(model.N_exc)

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
                x_tar_se=x_tar_se_test,
                x_tar_ee=x_tar_ee_test,
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

            del spikes_te_out, labels_te_out
            gc.collect()

        if count > 0:
            final_acc = acc_sum / count
            final_phi = phi_sum / count
            print(f"\nTest accuracy (PCA+LR): {final_acc:.4f}")
            print(f"Test phi: {final_phi:.4f}")
            self.logger._record_accuracy("test", final_acc, epoch=None, method="pca_lr")
            self.logger._record_phi("test", final_phi, epoch=None)
