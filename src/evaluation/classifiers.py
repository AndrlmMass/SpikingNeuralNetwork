"""
Classification algorithms for SNN output evaluation.
"""

import numpy as np
from sklearn.pipeline import Pipeline
import os
from datetime import datetime
from typing import Optional
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pathlib import Path

# Define t-SNE cache directory (project root / cache / tsne)
PROJECT_ROOT = Path(__file__).parent.parent.parent
_TSNE_CACHE_DIR = PROJECT_ROOT / "cache" / "tsne" 


def pca_logistic_regression(
    n_components=None,
    variance_ratio=0.95,
    whiten=True,
    standardize=True,
    max_iter=1000,
):
    steps = []
    if standardize:
        steps.append(("scaler", StandardScaler()))
    # PCA: choose either fixed components or variance ratio
    if n_components is None and variance_ratio is not None:
        pca = PCA(n_components=variance_ratio, svd_solver="full", whiten=whiten)
    else:
        pca = PCA(n_components=n_components, svd_solver="full", whiten=whiten)
    steps.append(("pca", pca))
    steps.append(("clf", LogisticRegression(
        multi_class="multinomial", solver="lbfgs", max_iter=max_iter
    )))
    return Pipeline(steps)

def fit_model(model, X_fit, y_fit):
    return model.fit(X_fit, y_fit)

def accuracy(model, X, y):
    return float(accuracy_score(y, model.predict(X)))


def pca_quadratic_discriminant(
    n_components: Optional[int] = None,
    variance_ratio: Optional[float] = 0.95,
    whiten: bool = True,
    standardize: bool = True,
    reg_param: float = 0.1,
):
    """
    Build a pipeline: (optional standardization) -> PCA -> QDA.
    You can then fit and score it like a normal sklearn model:
        model = pca_quadratic_discriminant(...)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
    """
    steps = []

    if standardize:
        steps.append(("scaler", StandardScaler()))

    # PCA configuration
    if n_components is None and variance_ratio is not None:
        pca = PCA(n_components=variance_ratio, svd_solver="full", whiten=whiten)
    elif n_components is not None:
        pca = PCA(n_components=n_components, svd_solver="full", whiten=whiten)
    else:
        pca = PCA(svd_solver="full", whiten=whiten)

    steps.append(("pca", pca))
    steps.append(("qda", QuadraticDiscriminantAnalysis(reg_param=reg_param)))

    return Pipeline(steps)

def _save_tsne_inputs(
    spikes: np.ndarray, labels: np.ndarray, split: str
) -> Optional[str]:
    """Persist raw spike activity and labels for t-SNE replotting without rerunning simulation."""
    if spikes is None or labels is None:
        raise ValueError("Spikes and labels cannot be None")

    T = min(spikes.shape[0], len(labels))
    if T <= 0:
        raise ValueError("Spikes and labels must have at least one sample")

    os.makedirs(_TSNE_CACHE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{split}_tsne_inputs_{T}samples_{timestamp}.npz"
    path = os.path.join(_TSNE_CACHE_DIR, filename)
    np.savez_compressed(path, spikes=spikes[:T], labels=np.asarray(labels[:T]))
    return path


def bin_spikes_by_label_no_breaks(spikes, labels):
    """
    Splits spike data into segments based on contiguous blocks in the labels vector,
    skipping any segments where the label is -1 or -2.

    Parameters:
        spikes (np.array): 2D array with shape (T, N) where T is total time and N is the number of neurons.
        labels (np.array): 1D array of length T indicating the label at each time point.

    Returns:
        features (np.array): 2D array where each row is the average spike activity for a valid segment.
        segment_labels (np.array): 1D array of labels corresponding to each segment.
    """
    T = min(spikes.shape[0], len(labels))
    if T <= 0:
        return np.empty((0, spikes.shape[1])), np.empty((0,), dtype=int)
    if spikes.shape[0] != T:
        spikes = spikes[:T]
    if len(labels) != T:
        labels = labels[:T]

    segments = []
    segment_labels = []
    start = 0

    for t in range(1, len(labels)):
        if labels[t] != labels[t - 1]:
            current_label = labels[t - 1]
            if current_label != -1 and current_label != -2 and t > start:
                segment = spikes[start:t]
                if segment.size > 0:
                    feature_vector = np.mean(segment, axis=0)
                    segments.append(feature_vector)
                    segment_labels.append(current_label)
            start = t

    # Handle the final segment
    if start < len(labels):
        current_label = labels[-1]
        if current_label != -1 and current_label != -2 and len(spikes[start:]) > 0:
            segment = spikes[start:]
            if segment.size > 0:
                feature_vector = np.mean(segment, axis=0)
                segments.append(feature_vector)
                segment_labels.append(current_label)

    return np.array(segments), np.array(segment_labels)


def t_SNE(
    spikes,
    labels_spike,
    n_components=2,
    perplexity=30,
    max_iter=1000,
    random_state=48,
    train=False,
    show_plot=False,
):
    """
    Apply t-SNE dimensionality reduction to spike data binned by label.
    """
    features, segment_labels = bin_spikes_by_label_no_breaks(spikes, labels_spike)

    if features.shape[0] < 3:
        raise ValueError(f"Insufficient samples for t-SNE ({features.shape[0]} < 3)")

    # Remove zero-variance features
    feature_var = np.var(features, axis=0)
    valid_features = feature_var >= 1e-10

    if np.sum(valid_features) < 2:
        raise ValueError(f"Insufficient non-zero variance features for t-SNE ({np.sum(valid_features)} < 2)")

    features_clean = features[:, valid_features]

    # Persist raw inputs for test data
    if not train:
        try:
            saved_path = _save_tsne_inputs(spikes, labels_spike, split="test")
            if saved_path:
                print(f"t-SNE test inputs cached at {saved_path}")
        except Exception as exc:
            raise ValueError(f"Failed to save t-SNE test inputs ({exc})")

    # Ensure perplexity is valid
    perplexity = min(30, len(features_clean) - 1)
    if perplexity < 1:
        raise ValueError("Perplexity must be at least 1")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            max_iter=max_iter,
            random_state=random_state,
        )
        tsne_results = tsne.fit_transform(features_clean)

    if np.any(~np.isfinite(tsne_results)):
        raise ValueError("t-SNE produced invalid values")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()

    marker_list = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "X"]
    unique_labels = np.unique(segment_labels)

    for i, label in enumerate(unique_labels):
        indices = segment_labels == label
        marker = marker_list[i % len(marker_list)]
        ax.scatter(
            tsne_results[indices, 0],
            tsne_results[indices, 1],
            label=f"Class {label}",
            marker=marker,
            color="black",
            s=60,
        )

    plt.xlabel("t-SNE dimension 1", fontsize=26)
    plt.ylabel("t-SNE dimension 2", fontsize=26)
    os.makedirs("plots", exist_ok=True)
    suffix = "train" if train else "test"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tsne_path = os.path.join("plots", f"tsne_{suffix}_{timestamp}.pdf")
    plt.tight_layout()
    plt.savefig(tsne_path, bbox_inches="tight")
    print(f"t-SNE plot saved to {tsne_path}")
    if show_plot:
        plt.show()
    plt.close(fig)

def _compute_spike_rates(spikes, labels, num_steps, require_all=False):
    """Helper to compute mean spike rates per stimulus presentation."""
    spike_rates = []
    unique_labels = []
    sleep_mask = labels != -2
    for i in range(0, labels.shape[0], num_steps):
        current_mask = sleep_mask[i : i + num_steps]
        chunk_spikes = spikes[i : i + num_steps]
        L = min(chunk_spikes.shape[0], current_mask.shape[0])
        if L == 0:
            if require_all:
                raise ValueError("No samples found")
            continue
        chunk_spikes = chunk_spikes[:L]
        current_mask = current_mask[:L]
        if not current_mask.all():
            if require_all:
                raise ValueError("No samples found")
            continue
        mean_spikes = np.mean(chunk_spikes[current_mask, :], axis=0)
        predom_label = np.argmax(np.bincount(labels[i : i + num_steps][current_mask]))
        spike_rates.append(mean_spikes)
        unique_labels.append(predom_label)

    return np.array(spike_rates), np.array(unique_labels)

class Phi:
    def __init__(self):
        pass
    @staticmethod
    def compute_centroids(scores: np.ndarray, labels: np.ndarray, num_classes: int):
        n_components = scores.shape[1]
        centroids = np.zeros((n_components, num_classes), dtype=float)
        present = np.unique(labels).astype(int)
        for c in present:
            idx = np.where(labels == c)[0]
            centroids[:, c] = np.mean(scores[idx], axis=0)
        return centroids, present

    @staticmethod
    def compute_wcss(scores: np.ndarray, labels: np.ndarray, centroids: np.ndarray):
        # scores: (n_samples, n_components)
        # centroids: (n_components, num_classes)
        wcss = 0.0
        for c in np.unique(labels).astype(int):
            idx = np.where(labels == c)[0]
            diffs = scores[idx] - centroids[:, c].T
            wcss += float(np.sum(diffs * diffs))
        return wcss

    @staticmethod
    def compute_bcss(scores: np.ndarray, labels: np.ndarray, centroids: np.ndarray):
        overall = np.mean(scores, axis=0)
        bcss = 0.0
        for c in np.unique(labels).astype(int):
            idx = np.where(labels == c)[0]
            n_c = idx.size
            mu_c = centroids[:, c]
            bcss += n_c * float(np.sum((mu_c - overall) ** 2))
        return bcss

    def fit(
        self,
        spikes: np.ndarray,
        labels: np.ndarray,
        num_steps: int,
        num_classes: int,
        pca_variance: float = 0.95,
        random_state: int = 0,
        var_eps: float = 1e-10,
    ) -> 'Phi':
        rates, item_labels = _compute_spike_rates(
            spikes, labels, num_steps,
            require_any=True
        )

        if rates.shape[0] < 3:
            raise ValueError(f"Insufficient samples for phi fit ({rates.shape[0]} < 3)")

        feature_var = np.var(rates, axis=0)
        valid = feature_var >= var_eps
        if np.sum(valid) < 2:
            raise ValueError(f"Insufficient non-zero variance features ({np.sum(valid)} < 2)")

        rates_clean = rates[:, valid]

        scaler = StandardScaler(with_mean=True, with_std=True)
        rates_std = scaler.fit_transform(rates_clean)

        pca = PCA(n_components=pca_variance, random_state=random_state)
        scores = pca.fit_transform(rates_std)   # (n_samples, n_components)

        centroids, _present = self.compute_centroids(scores, item_labels, num_classes)

        # store fitted state
        self.valid_features = valid
        self.scaler = scaler
        self.pca = pca
        self.centroids = centroids
        self.num_classes = int(num_classes)

        return self

    def score(
        self,
        spikes: np.ndarray,
        labels: np.ndarray,
        *,
        num_steps: int,
        small_num: float = 1e-5,
        require_any: bool = False,
    ):
        # crumb: guard against calling score() before fit()
        if self.valid_features is None or self.scaler is None or self.pca is None or self.centroids is None:
            raise RuntimeError("Phi.score() called before fit(). Call fit() first.")

        rates, item_labels = _compute_spike_rates(
            spikes, labels, num_steps,
            require_any=require_any
        )

        n = rates.shape[0]
        if n < 2:
            return 0.0

        rates_clean = rates[:, self.valid_features]
        rates_std = self.scaler.transform(rates_clean)
        scores = self.pca.transform(rates_std)

        # effective k = number of classes actually present in THIS split
        present = np.unique(item_labels).astype(int)
        k_eff = int(present.size)

        if k_eff < 2:
            return 0.0

        wcss = self.compute_wcss(scores, item_labels, self.centroids)
        bcss = self.compute_bcss(scores, item_labels, self.centroids)

        wcss_scaled = wcss / max(small_num, n - k_eff)
        bcss_scaled = bcss / max(small_num, k_eff - 1)

        return float(bcss_scaled / wcss_scaled) if wcss > small_num else 0.0


