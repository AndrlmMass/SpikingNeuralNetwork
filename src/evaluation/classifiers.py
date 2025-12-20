"""
Classification algorithms for SNN output evaluation.
"""

import numpy as np
from sklearn.pipeline import Pipeline
import os
from datetime import datetime
from typing import Dict, Tuple, Optional
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from src.init import _TSNE_CACHE_DIR 


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
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_components: Optional[int] = None,
    variance_ratio: Optional[float] = 0.95,
    whiten: bool = True,
    standardize: bool = True,
    reg_param: float = 0.1,
) -> Tuple[Dict[str, float], StandardScaler, PCA, QuadraticDiscriminantAnalysis]:
    """
    Reduce high-dimensional features with PCA, then classify with QDA (Quadratic Discriminant Analysis).
    - Regularization via reg_param helps when classes are few or covariance is near-singular.
    """
    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)
    else:
        scaler = StandardScaler(with_mean=False, with_std=False)
        X_train_s, X_val_s, X_test_s = X_train, X_val, X_test

    if n_components is None and variance_ratio is not None:
        pca = PCA(n_components=variance_ratio, svd_solver="full", whiten=whiten)
    elif n_components is not None:
        pca = PCA(n_components=n_components, svd_solver="full", whiten=whiten)
    else:
        pca = PCA(svd_solver="full", whiten=whiten)

    X_train_p = pca.fit_transform(X_train_s)
    X_val_p = pca.transform(X_val_s)
    X_test_p = pca.transform(X_test_s)

    clf = QuadraticDiscriminantAnalysis(reg_param=reg_param)
    clf.fit(X_train_p, y_train)

    accs = {
        "train": float(accuracy_score(y_train, clf.predict(X_train_p))),
        "val": float(accuracy_score(y_val, clf.predict(X_val_p))),
        "test": float(accuracy_score(y_test, clf.predict(X_test_p))),
    }

    return accs, scaler, pca, clf

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


def _compute_wcss(scores, labels, centroids, num_classes):
    """Helper to compute Within-Cluster Sum of Squares."""
    n_components = centroids.shape[0]
    wcss_arr = np.zeros(num_classes)
    present_classes = np.unique(labels).astype(int)
    for c in present_classes:
        indices = np.where(labels == c)[0]
        values = scores[indices]
        for dim in range(n_components):
            delta_wcss = np.sum((values[:, dim] - centroids[dim, c]) ** 2)
            wcss_arr[c] += delta_wcss
    return float(np.sum(wcss_arr))


def phi(
    num_steps_train,
    num_steps_test,
    pca_variance,
    random_state,
    num_classes,
    spikes_train=None,
    spikes_test=None,
    labels_test=None,
    labels_train=None,
):
    """
    Calculate the clustering coefficient (phi) for train and test data.
    
    Phi = (BCSS / (k-1)) / (WCSS / (n-k))
    
    Where BCSS = Between-Cluster Sum of Squares, WCSS = Within-Cluster Sum of Squares,
    k = number of classes, n = number of samples.
    """
    small_num = 1e-5

    # Align and process training arrays
    if spikes_train.shape[0] != labels_train.shape[0]:
        raise ValueError("Spikes and labels must have the same length")

    spike_train_rates, labels_train_unique = _compute_spike_rates(
        spikes_train, labels_train, num_steps_train, require_all=True
    )

    if spike_train_rates.shape[0] < 3:
        raise ValueError(f"Insufficient samples for phi calculation ({spike_train_rates.shape[0]} < 3)")

    feature_var = np.var(spike_train_rates, axis=0)
    if np.all(feature_var < 1e-10):
        raise ValueError("All features have zero variance (no network activity)")

    valid_features = feature_var >= 1e-10
    if np.sum(valid_features) < 2:
        raise ValueError(f"Insufficient non-zero variance features ({np.sum(valid_features)} < 2)")

    spike_train_rates_clean = spike_train_rates[:, valid_features]

    scaler = StandardScaler(with_mean=True, with_std=True)
    spike_train_rates_std = scaler.fit_transform(spike_train_rates_clean)

    pca = PCA(n_components=pca_variance, random_state=random_state)
    pca.fit(spike_train_rates_std)
    n_components = pca.n_components_
    scores_train_pca = pca.transform(spike_train_rates_std)

    # Calculate centroids and WCSS (train)
    centroids = np.zeros((n_components, num_classes))
    present_train_classes = np.unique(labels_train_unique).astype(int)
    
    for c in present_train_classes:
        indices = np.where(labels_train_unique == c)[0]
        centroids[:, c] = np.mean(scores_train_pca[indices], axis=0)
    
    WCSS_train = _compute_wcss(scores_train_pca, labels_train_unique, centroids, num_classes)

    # Calculate BCSS (train)
    overall_mean = np.mean(scores_train_pca, axis=0)
    n_train = scores_train_pca.shape[0]
    k_eff_train = max(1, present_train_classes.size)
    BCSS_train = 0.0
    for c in present_train_classes:
        idx = np.where(labels_train_unique == c)[0]
        n_c = idx.size
        if n_c > 0:
            mu_c = centroids[:, c]
            BCSS_train += n_c * float(np.sum((mu_c - overall_mean) ** 2))

    WCSS_train_scaled = WCSS_train / max(small_num, n_train - k_eff_train)
    BCSS_train_scaled = BCSS_train / max(small_num, k_eff_train - 1)
    phi_train = BCSS_train_scaled / WCSS_train_scaled if WCSS_train > small_num else 0.0


    # Process test data
    if spikes_test is not None and labels_test is not None:
        if spikes_test.shape[0] != labels_test.shape[0]:
            raise ValueError("Spikes and labels must have the same length")

        spike_test_rates, labels_test_unique = _compute_spike_rates(
            spikes_test, labels_test, num_steps_test, require_all=False
        )

        if spike_test_rates.shape[0] < 5:
            raise ValueError(f"Insufficient test data for phi calculation ({spike_test_rates.shape[0]} < 5)")

        feature_var = np.var(spike_test_rates, axis=0)
        if np.all(feature_var < 1e-10):
            raise ValueError("All features have zero variance (no network activity)")

        valid_features = feature_var >= 1e-10
        if np.sum(valid_features) < 2:
            raise ValueError(f"Insufficient non-zero variance features ({np.sum(valid_features)} < 2)")

        spike_test_rates_clean = spike_test_rates[:, valid_features]
        pca_test_rates = pca.transform(spike_test_rates_clean)

        # Calculate WCSS (test)
        WCSS_test = _compute_wcss(pca_test_rates, labels_test_unique, centroids, num_classes)

        # Calculate BCSS (test)
        n_test = pca_test_rates.shape[0]
        k_eff_test = max(1, np.unique(labels_test_unique).size)
        BCSS_test = 0.0
        overall_mean_test = np.mean(pca_test_rates, axis=0)
        for c in np.unique(labels_test_unique).astype(int):
            idx = np.where(labels_test_unique == c)[0]
            n_c = idx.size
            if n_c > 0:
                mu_c_test = np.mean(pca_test_rates[idx], axis=0)
                BCSS_test += n_c * float(np.sum((mu_c_test - overall_mean_test) ** 2))

        WCSS_test_scaled = WCSS_test / max(small_num, n_test - k_eff_test)
        BCSS_test_scaled = BCSS_test / max(small_num, k_eff_test - 1)
        phi_test = BCSS_test_scaled / WCSS_test_scaled if WCSS_test > small_num else 0.0

        return phi_train, phi_test
    else:
        return phi_train