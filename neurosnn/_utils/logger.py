from dataclasses import dataclass, field
from datetime import datetime
import json
import os

import numpy as np


def tracking_run_dir(dataset, ts_spec) -> str:
    '''
    Single source of truth for a run's output folder:
    results/tracking/<dataset>/<date>/<ts_spec>. The date is derived from the
    YYYYMMDD prefix of ts_spec so every caller (logger, spike worker, PCA display,
    gif assembly) resolves to the same per-run directory without any plumbing.
    Falls back to today's date only if ts_spec is not an 8-digit-prefixed stamp.
    '''
    ts = str(ts_spec)
    if len(ts) >= 8 and ts[:8].isdigit():
        date = f"{ts[:4]}.{ts[4:6]}.{ts[6:8]}"
    else:
        date = datetime.now().strftime("%Y.%m.%d")
    return os.path.join("results", "tracking", str(dataset), date, ts)


@dataclass
class HistoryTracker:
    ts_spec: str
    image_dataset: str = "unknown"
    acc_history: dict = field(default_factory=dict)
    _run_dir: str = None
    _stats_log_file: str = None

    def _ensure_run_dir(self):
        '''Resolve + create the per-run dir and the stats/ subfolder (lazy).'''
        if self._stats_log_file is None:
            self.image_dataset = getattr(self, "image_dataset", "unknown")
            try:
                self._run_dir = tracking_run_dir(self.image_dataset, self.ts_spec)
                os.makedirs(os.path.join(self._run_dir, "stats"), exist_ok=True)
            except Exception:
                pass
            self._stats_log_file = os.path.join(self._run_dir, "stats", "stats.jsonl")

    def log_config(self, config: dict):
        self._ensure_run_dir()
        path = os.path.join(self._run_dir, "config.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Warning: failed to write config ({e})")

    def _append_record(self, rec: dict):
        '''Append one JSON line to the unified stats jsonl (accuracy + phi + mcc
        + per-batch diagnostics all live here now).'''
        try:
            self._ensure_run_dir()
            with open(self._stats_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
        except Exception as e:
            print(f"Warning: failed to persist record ({e})")

    def _record_accuracy(
        self, split: str, value, epoch: int | None = None, method: str | None = None
    ):
        try:
            acc_val = float(value) if value is not None else None
        except Exception:
            acc_val = None
        try:
            if acc_val is not None:
                method_key = method if method is not None else "default"
                self.acc_history.setdefault(split, {}).setdefault(
                    method_key, []
                ).append(acc_val)
        except Exception:
            pass
        rec = {
            "timestamp": datetime.now().isoformat(),
            "split": str(split),
            "epoch": int(epoch) if epoch is not None else None,
            "accuracy": acc_val,
        }
        if method is not None:
            rec["method"] = method
        self._append_record(rec)

    def _record_phi(self, split: str, phi_value, epoch: int | None = None):
        try:
            phi_val = float(phi_value) if phi_value is not None else None
        except Exception:
            phi_val = None
        self._append_record(
            {
                "timestamp": datetime.now().isoformat(),
                "split": str(split),
                "epoch": int(epoch) if epoch is not None else None,
                "phi": phi_val,
            }
        )

    def _record_mcc(self, split: str, mcc_value, epoch: int | None = None):
        try:
            mcc_val = float(mcc_value) if mcc_value is not None else None
        except Exception:
            mcc_val = None
        self._append_record(
            {
                "timestamp": datetime.now().isoformat(),
                "split": str(split),
                "epoch": int(epoch) if epoch is not None else None,
                "mcc": mcc_val,
            }
        )

    def _record_stats(self, stats: dict, epoch: int | None = None):
        '''
        Persist one per-batch diagnostics dict (TrainTracker.to_dict) as a JSON
        line in the unified stats jsonl. Numpy scalars are coerced to float so the
        record is JSON-serialisable. Diagnostics records are distinguished from
        accuracy/phi/mcc records by the absence of those keys (see plot_stats).
        '''
        if not stats:
            return
        clean = {}
        for k, v in stats.items():
            if isinstance(v, (np.floating, np.integer)):
                clean[k] = float(v)
            elif isinstance(v, (int, float)) or v is None:
                clean[k] = v
            else:
                try:
                    clean[k] = float(v)
                except (TypeError, ValueError):
                    clean[k] = None
        self._append_record(
            {
                "timestamp": datetime.now().isoformat(),
                "epoch": int(epoch) if epoch is not None else None,
                **clean,
            }
        )

    def _read_jsonl(self, path):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # skip truncated/corrupted lines from concurrent writes
        return records


# ---------------------------------------------------------------------------
# Diagnostic metrics (RF-collapse harness)
#
# Pure NumPy snapshot metrics computed once per batch from the full weight
# matrix / spike array. Each isolates one axis of the suspected E-layer
# collapse so accuracy alone does not have to disambiguate them:
#   within-neuron concentration -> RF over-sharpening   (synapse)
#   cross-neuron RF diversity   -> redundant RFs         (synapse)
#   functional E/I balance      -> over-inhibition       (synapse)
#   population activity         -> population collapse    (neuron)
#   trace spread                -> threshold drift        (synapse)
# All are defensive (eps guards, empty-array safe) so they never crash a run.
# ---------------------------------------------------------------------------

_EPS = 1e-12


def _gini_columns(W):
    '''
    Gini coefficient per column of a nonnegative matrix, via the sorted formula
    G = 2*Σ(i*x_i)/(n*Σx) - (n+1)/n. Columns summing to ~0 return 0.
    '''
    n = W.shape[0]
    Ws = np.sort(W, axis=0)
    idx = np.arange(1, n + 1, dtype=np.float64)[:, None]
    s = Ws.sum(axis=0)
    g = (2.0 * (idx * Ws).sum(axis=0)) / (n * s + _EPS) - (n + 1.0) / n
    g[s <= _EPS] = 0.0
    return g


def rf_within_concentration(W_se):
    '''
    Within-neuron RF concentration. Each column of W_se (N_x, N_exc) is one
    E-neuron's receptive field; normalize it to a probability vector and report
    the population-mean Shannon entropy (nats; low = peaky RF) and Gini
    (high = concentrated onto few inputs). Diagnoses RF over-sharpening.
    '''
    W = np.asarray(W_se, dtype=np.float64)
    if W.size == 0:
        return 0.0, 0.0
    W = np.clip(W, 0.0, None)
    colsum = W.sum(axis=0)
    P = W / (colsum + _EPS)
    ent = -(P * np.log(P + _EPS)).sum(axis=0)
    gini = _gini_columns(W)
    live = colsum > _EPS
    if not live.any():
        return 0.0, 0.0
    return float(ent[live].mean()), float(gini[live].mean())


def rf_diversity(W_se):
    '''
    Cross-neuron RF diversity for W_se (N_x, N_exc). Returns mean pairwise cosine
    similarity between RFs (->1 means redundant) and the participation ratio
    PR = (Σs^2)^2 / Σs^4 of the centered RF set (->1 means full collapse onto one
    template), plus PR normalized by min(N_exc, N_x) into (0, 1].
    '''
    R = np.asarray(W_se, dtype=np.float64).T  # (N_exc, N_x), one RF per row
    if R.shape[0] < 2 or R.shape[1] == 0:
        return 0.0, 1.0, 0.0
    norms = np.linalg.norm(R, axis=1, keepdims=True)
    Rn = R / (norms + _EPS)
    G = Rn @ Rn.T
    iu = np.triu_indices(G.shape[0], k=1)
    mean_cos = float(G[iu].mean())
    Rc = R - R.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Rc, compute_uv=False)
    s2 = s * s
    pr = float((s2.sum() ** 2) / ((s2 * s2).sum() + _EPS))
    pr_norm = pr / min(R.shape)
    return mean_cos, pr, pr_norm


def ei_balance(weights, spikes, st, ex, ih):
    '''
    Functional inhibition/excitation balance per E neuron. Uses window-mean firing
    rates so silent presynaptic neurons contribute no current:
        exc_i = (W_se^T rate_in)_i + (W_ee^T rate_e)_i
        inh_i = (|W_ie|^T rate_i)_i
    Returns median ratio, 90th-percentile ratio (heavy tail = a strangled subset),
    and the full per-neuron ratio vector. Diagnoses over-inhibition.
    '''
    W_se = weights[:st, st:ex]
    W_ee = weights[st:ex, st:ex]
    W_ie = weights[ex:ih, st:ex]
    if W_se.shape[1] == 0 or spikes.shape[0] == 0:
        return 0.0, 0.0, np.zeros(0)
    rate_in = spikes[:, :st].mean(axis=0)
    rate_e = spikes[:, st:ex].mean(axis=0)
    rate_i = spikes[:, ex:ih].mean(axis=0)
    exc = W_se.T @ rate_in + W_ee.T @ rate_e
    inh = np.abs(W_ie).T @ rate_i
    ratio = inh / (exc + _EPS)
    return float(np.median(ratio)), float(np.percentile(ratio, 90)), ratio


def population_activity(spikes, st, ex, theta=0.0):
    '''
    Population-level activity over the batch window. Returns the fraction of E
    neurons firing above theta, the Treves-Rolls sparseness
    S = (mean r)^2 / mean(r^2) in (0, 1] (->1 even spread, ->0 a few dominate),
    and the per-neuron rate vector. Diagnoses population collapse.
    '''
    if spikes.shape[0] == 0 or ex <= st:
        return 0.0, 0.0, np.zeros(0)
    r = spikes[:, st:ex].mean(axis=0)
    active_frac = float((r > theta).mean())
    S = float(r.mean() ** 2 / ((r * r).mean() + _EPS))
    return active_frac, S, r


def trace_spread(spike_trace, st, ex):
    '''
    Median and 90th-percentile of the excitatory spike-trace. p90 >> p50 means a
    few hyperactive E neurons drag x_tar_ee up and starve the rest -> threshold
    drift. spike_trace has length N_x + N_exc, so [st:ex] are the E entries.
    '''
    if spike_trace is None or ex <= st:
        return 0.0, 0.0
    exc_trace = np.asarray(spike_trace[st:ex], dtype=np.float64)
    if exc_trace.size == 0:
        return 0.0, 0.0
    return float(np.percentile(exc_trace, 50)), float(np.percentile(exc_trace, 90))
