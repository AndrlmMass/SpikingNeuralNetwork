import warnings
import os
from datetime import datetime
from typing import Dict, Tuple, Optional
import numpy as np
import matplotlib
from platform_utils import configure_matplotlib
configure_matplotlib()
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
warnings.filterwarnings("error", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="matplotlib")

_TSNE_CACHE_DIR = os.path.join("data", "tsne")


def _save_tsne_inputs(
    spikes: np.ndarray, labels: np.ndarray, split: str
) -> Optional[str]:
    """
    Persist the raw spike activity and labels that seed the t-SNE plot so they
    can be reloaded later without rerunning a full simulation.
    """

    if spikes is None or labels is None:
        return None

    T = min(spikes.shape[0], len(labels))
    if T <= 0:
        return None

    os.makedirs(_TSNE_CACHE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{split}_tsne_inputs_{T}samples_{timestamp}.npz"
    path = os.path.join(_TSNE_CACHE_DIR, filename)
    np.savez_compressed(
        path,
        spikes=spikes[:T],
        labels=np.asarray(labels[:T]),
    )
    return path


def bin_spikes_by_label_no_breaks(spikes, labels):
    """
    Splits spike data into segments based on contiguous blocks in the labels vector,
    skipping any segments where the label is -1.

    Parameters:
        spikes (np.array): 2D array with shape (T, N) where T is total time and N is the number of neurons.
        labels (np.array): 1D array of length T indicating the label at each time point.

    Returns:
        features (np.array): 2D array where each row is the average spike activity for a valid segment.
        segment_labels (np.array): 1D array of labels corresponding to each segment.
    """
    # Align lengths defensively
    T = min(spikes.shape[0], len(labels))
    if T <= 0:
        return np.empty((0, spikes.shape[1])), np.empty((0,), dtype=int)
    if spikes.shape[0] != T:
        spikes = spikes[:T]
    if len(labels) != T:
        labels = labels[:T]

    segments = []
    segment_labels = []
    start = 0  # start index for the current segment

    # Iterate through the labels to detect change points.
    for t in range(1, len(labels)):
        if labels[t] != labels[t - 1]:
            # End of the current segment.
            current_label = labels[t - 1]
            # Process only if the current label is not -1.
            if current_label != -1 and current_label != -2 and t > start:
                segment = spikes[start:t]
                if segment.size > 0:
                    # Compute feature for the segment (here, the mean firing rate for each neuron).
                    feature_vector = np.mean(segment, axis=0)
                    segments.append(feature_vector)
                    segment_labels.append(current_label)
            # Update the start index for the next segment.
            start = t

    # Handle the final segment.
    if start < len(labels):
        current_label = labels[-1]
        if current_label != -1 and current_label != -2 and len(spikes[start:]) > 0:
            segment = spikes[start:]
            if segment.size > 0:
                feature_vector = np.mean(segment, axis=0)
                segments.append(feature_vector)
                segment_labels.append(current_label)

    return np.array(segments), np.array(segment_labels)


def pca_logistic_regression(
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
    max_iter: int = 1000,
) -> Tuple[Dict[str, float], StandardScaler, PCA, LogisticRegression]:
    """
    Reduce high-dimensional features with PCA, then classify with multinomial Logistic Regression.

    Notes:
    - PCA is fit on train only, then applied to val/test.
    - Use either n_components (int) or variance_ratio (0-1) to set dimensionality.
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

    clf = LogisticRegression(
        multi_class="multinomial", solver="lbfgs", max_iter=max_iter
    )
    clf.fit(X_train_p, y_train)

    accs = {
        "train": float(accuracy_score(y_train, clf.predict(X_train_p))),
        "val": float(accuracy_score(y_val, clf.predict(X_val_p))),
        "test": float(accuracy_score(y_test, clf.predict(X_test_p))),
    }

    return accs, scaler, pca, clf


def t_SNE(
    spikes,
    labels_spike,
    n_components,
    perplexity,
    max_iter,
    random_state,
    train,
    show_plot=False,
):
    # Now, bin the spikes using the labels, skipping breaks:
    features, segment_labels = bin_spikes_by_label_no_breaks(spikes, labels_spike)

    # Check for sufficient data
    if features.shape[0] < 3:
        print(f"Warning: Insufficient samples for t-SNE ({features.shape[0]} < 3)")
        return

    # Remove zero-variance features to prevent numerical issues
    feature_var = np.var(features, axis=0)
    valid_features = feature_var >= 1e-10

    if np.sum(valid_features) < 2:
        print(
            f"Warning: Insufficient non-zero variance features for t-SNE ({np.sum(valid_features)} < 2)"
        )
        return

    features_clean = features[:, valid_features]

    # Persist raw inputs for downstream analysis when working on held-out data.
    if not train:
        try:
            saved_path = _save_tsne_inputs(spikes, labels_spike, split="test")
            if saved_path:
                print(f"t-SNE test inputs cached at {saved_path}")
        except Exception as exc:
            print(f"Warning: Failed to save t-SNE test inputs ({exc})")

    # Apply t-SNE on the computed features:
    # Ensure that perplexity is less than the number of segments.
    perplexity = min(30, len(features_clean) - 1)

    if perplexity < 1:
        print(f"Warning: Perplexity too low for t-SNE")
        return

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            max_iter=max_iter,
            random_state=random_state,
        )
        tsne_results = tsne.fit_transform(features_clean)

    # Validate t-SNE output
    if np.any(~np.isfinite(tsne_results)):
        print(f"Warning: t-SNE produced invalid values")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()

    markers = {0: "o", 1: "s", 2: "^", 3: "D"}
    # Visualize the results:
    for label in np.unique(segment_labels):
        indices = segment_labels == label
        ax.scatter(
            tsne_results[indices, 0],
            tsne_results[indices, 1],
            label=f"Class {label}",
            marker=markers[label],
            color="black",
            s=40,
        )
    if train:
        title = "from training"
    else:
        title = "from testing"

    plt.xlabel("t-SNE dimension 1", fontsize=16)
    plt.ylabel("t-SNE dimension 2", fontsize=16)
    plt.legend(fontsize=14)
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


def PCA_analysis(
    spikes,
    labels_spike,
    pca_variance,
    random_state,
):
    # Bin the spikes using the labels (assuming this function is defined elsewhere)
    features, segment_labels = bin_spikes_by_label_no_breaks(spikes, labels_spike)

    # Check for sufficient data
    if features.shape[0] < 3:
        print(
            f"Warning: Insufficient samples for PCA analysis ({features.shape[0]} < 3)"
        )
        return

    # Remove zero-variance features to prevent numerical issues
    feature_var = np.var(features, axis=0)
    valid_features = feature_var >= 1e-10

    if np.sum(valid_features) < 2:
        print(
            f"Warning: Insufficient non-zero variance features for PCA ({np.sum(valid_features)} < 2)"
        )
        return

    features_clean = features[:, valid_features]

    # Create a PCA instance.
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        pca = PCA(n_components=pca_variance, random_state=random_state)
        pca_results = pca.fit_transform(features_clean)

    # Validate PCA output
    if np.any(~np.isfinite(pca_results)):
        print(f"Warning: PCA produced invalid values")
        return

    clf = LogisticRegression(
        multi_class="auto", solver="lbfgs", random_state=random_state
    )
    clf.fit(pca_results, segment_labels)
    pred_labels = clf.predict(pca_results)
    acc = accuracy_score(segment_labels, pred_labels)

    # Create a mesh grid to plot the decision boundary:
    x_min, x_max = pca_results[:, 0].min() - 1, pca_results[:, 0].max() + 1
    y_min, y_max = pca_results[:, 1].min() - 1, pca_results[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # For multi-class classification, predict the class label for each point in the grid:
    grid_pred = clf.predict(grid).reshape(xx.shape)

    # Plotting:
    plt.figure(figsize=(10, 8))

    # Plot the decision regions
    plt.contourf(xx, yy, grid_pred, alpha=0.3, cmap="coolwarm")

    # Scatter plot for each class:
    for label in np.unique(segment_labels):
        indices = segment_labels == label
        plt.scatter(
            pca_results[indices, 0],
            pca_results[indices, 1],
            label=f"Class {label}",
            edgecolor="k",
            s=60,
        )

    plt.title("2D PCA with Logistic Regression Decision Boundary: accuracy " + str(acc))
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()


def calculate_phi(
    spikes_train,
    spikes_test,
    labels_test,
    labels_train,
    num_steps,
    pca_variance,
    random_state,
    num_classes,
):
    """
    Recipe:

    First, fit training data to a PCA structure and create K-means centroids based on that data.

    Then, project the test-data onto that PCA-structure with its respective K-means centroids.
    """

    """
    First: Fit PCA and K-means to training data
    """

    # Trim training arrays to the same length to avoid misalignment
    try:
        T_tr = min(spikes_train.shape[0], labels_train.shape[0])
        spikes_train = spikes_train[:T_tr]
        labels_train = labels_train[:T_tr]
    except Exception:
        pass

    """Calculate the spiking rates for each item presentation"""

    # Calculate rate for each item
    spike_train_rates = []
    labels_train_unique = []

    """
    Note that we are currently skipping sleep-patterns. 
    Maybe this should be its own array, but then we miss part of the sequence
    """
    sleep_mask = labels_train != -2
    count = 0
    small_num = 10**-5
    for i in range(0, labels_train.shape[0], num_steps):
        # skip non_sleep spiking activity
        current_mask = sleep_mask[i : i + num_steps]
        chunk_spikes = spikes_train[i : i + num_steps]
        # Align chunk and mask
        L = min(chunk_spikes.shape[0], current_mask.shape[0])
        if L == 0:
            continue
        chunk_spikes = chunk_spikes[:L]
        current_mask = current_mask[:L]
        if not current_mask.all():
            count += 1
            continue
        mean_spikes = np.mean(chunk_spikes[current_mask, :], axis=0)
        predom_label = np.argmax(
            np.bincount(labels_train[i : i + num_steps][current_mask])
        )
        spike_train_rates.append(mean_spikes)
        labels_train_unique.append(predom_label)

    """
    create cutoff point, only for training!
    """

    # convert to numpy array
    spike_train_rates = np.array(spike_train_rates)
    labels_train_unique = np.array(labels_train_unique)

    # Check for sufficient samples before any calculations
    if spike_train_rates.shape[0] < 3:
        print(
            f"Warning: Insufficient samples for phi calculation ({spike_train_rates.shape[0]} < 3)"
        )
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Check for zero-variance features before standardization
    feature_var = np.var(spike_train_rates, axis=0)
    if np.all(feature_var < 1e-10):
        print(f"Warning: All features have zero variance (no network activity)")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Remove zero-variance features
    valid_features = feature_var >= 1e-10
    if np.sum(valid_features) < 2:
        print(
            f"Warning: Insufficient non-zero variance features ({np.sum(valid_features)} < 2)"
        )
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    spike_train_rates_clean = spike_train_rates[:, valid_features]

    # standardize rates (fit on train; reuse scaler for test)
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        scaler = StandardScaler(with_mean=True, with_std=True)
        spike_train_rates_std = scaler.fit_transform(spike_train_rates_clean)

    """ Perform PCA on the binned data """
    # Create a PCA instance
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        pca = PCA(n_components=pca_variance, random_state=random_state)
        pca.fit(spike_train_rates_std)
        n_components = pca.n_components_
        scores_train_pca = pca.transform(spike_train_rates_std)

    # Calculate centroids and WCSS (train)
    centroids = np.zeros((n_components, num_classes))
    wcss_arr = np.zeros(num_classes)
    present_train_classes = np.unique(labels_train_unique).astype(int)
    for c in present_train_classes:
        indices = np.where(labels_train_unique == c)[0]
        centroids[:, c] = np.mean(scores_train_pca[indices], axis=0)
        values = scores_train_pca[indices]
        for dim in range(n_components):
            delta_wcss = np.sum((centroids[dim, c] - values[:, dim]) ** 2)
            wcss_arr[c] += delta_wcss
    WCSS_train = float(np.sum(wcss_arr))

    # Estimate BCSS (train) using only present classes and class counts
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

    # Calculate clustering coefficient
    """
    phi = (BCSS / (k-1)) / (WCSS / (n-k))
    """
    denom1_tr = max(small_num, k_eff_train - 1)
    denom2_tr = max(small_num, n_train - k_eff_train)
    if WCSS_train <= small_num:
        phi_train = 0.0
    else:
        phi_train = (BCSS_train / denom1_tr) / (WCSS_train / denom2_tr)

    """
    Second: Project test data onto precomputed PCA and K-means centroids
    """

    """Calculate the spiking rates for each item presentation"""

    # Calculate rate for each item
    spike_test_rates = []
    labels_test_unique = []

    """
    Note that we are currently skipping sleep-patterns. 
    Maybe this should be its own array, but then we miss part of the sequence
    """
    # Trim test arrays to the same length to avoid misalignment
    try:
        T_te = min(spikes_test.shape[0], labels_test.shape[0])
        spikes_test = spikes_test[:T_te]
        labels_test = labels_test[:T_te]
    except Exception:
        pass

    sleep_mask = labels_test != -2
    for i in range(0, labels_test.shape[0], num_steps):
        # skip non_sleep spiking activity
        current_mask = sleep_mask[i : i + num_steps]
        chunk_spikes = spikes_test[i : i + num_steps]
        # Align chunk and mask
        L = min(chunk_spikes.shape[0], current_mask.shape[0])
        if L == 0:
            continue
        chunk_spikes = chunk_spikes[:L]
        current_mask = current_mask[:L]
        if not current_mask.all():
            continue
        mean_spikes = np.mean(chunk_spikes[current_mask, :], axis=0)
        predom_label = np.argmax(
            np.bincount(labels_test[i : i + num_steps][current_mask])
        )
        spike_test_rates.append(mean_spikes)
        labels_test_unique.append(predom_label)

    """'
    create cutoff point, only for training!
    """

    # convert to numpy array
    spike_test_rates = np.array(spike_test_rates)
    labels_test_unique = np.array(labels_test_unique)

    # Check if we have sufficient test data
    if spike_test_rates.shape[0] < 5:
        print(
            f"Warning: Insufficient test data for phi calculation ({spike_test_rates.shape[0]} samples)"
        )
        return phi_train, 0.0, 0.0, 0.0, 0.0, 0.0

    # Apply same feature selection as training (use valid_features mask)
    spike_test_rates_clean = spike_test_rates[:, valid_features]

    # standardize rates using train-fitted scaler
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        spike_test_rates_std = scaler.transform(spike_test_rates_clean)

    """ Perform PCA on the binned data """
    # Apply PCA transform
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        scores_test_pca = pca.transform(spike_test_rates_std)

    # Calculate WCSS (test) relative to train centroids for continuity
    wcss_arr = np.zeros(num_classes)
    present_test_classes = np.unique(labels_test_unique).astype(int)
    for c in present_test_classes:
        indices = np.where(labels_test_unique == c)[0]
        values = scores_test_pca[indices]
        for dim in range(n_components):
            delta_wcss = np.sum((values[:, dim] - centroids[dim, c]) ** 2)
            wcss_arr[c] += delta_wcss
    WCSS_test = float(np.sum(wcss_arr))

    # Estimate BCSS (test) using test class means and counts
    n_test = scores_test_pca.shape[0]
    overall_mean_test = np.mean(scores_test_pca, axis=0)
    k_eff_test = max(1, present_test_classes.size)
    BCSS_test = 0.0
    for c in present_test_classes:
        idx = np.where(labels_test_unique == c)[0]
        n_c = idx.size
        if n_c > 0:
            mu_c_test = np.mean(scores_test_pca[idx], axis=0)
            BCSS_test += n_c * float(np.sum((mu_c_test - overall_mean_test) ** 2))

    # Calculate clustering coefficient with safe divisions
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        test_denom1 = max(small_num, k_eff_test - 1)
        test_denom2 = max(small_num, n_test - k_eff_test)

        if WCSS_test <= small_num:
            phi_test = 0.0
        else:
            phi_test = (BCSS_test / test_denom1) / (WCSS_test / test_denom2)

        if not np.isfinite(phi_test):
            phi_test = 0.0

        BCSS_test_scaled = BCSS_test / max(small_num, k_eff_test - 1)
        BCSS_train_scaled = BCSS_train / max(small_num, k_eff_train - 1)
        WCSS_test_scaled = WCSS_test / max(small_num, n_test - k_eff_test)
        WCSS_train_scaled = WCSS_train / max(small_num, n_train - k_eff_train)

    return (
        phi_train,
        phi_test,
        WCSS_train_scaled,
        WCSS_test_scaled,
        BCSS_train_scaled,
        BCSS_test_scaled,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="t-SNE utilities")
    parser.add_argument(
        "--tsne-path",
        type=str,
        help="Path to a saved npz containing 'spikes' and 'labels' arrays",
    )
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--perplexity", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=48)
    cli_args = parser.parse_args()

    if cli_args.tsne_path:
        payload = np.load(cli_args.tsne_path)
        spikes = payload["spikes"]
        labels = payload["labels"]
        t_SNE(
            spikes=spikes,
            labels_spike=labels,
            n_components=cli_args.n_components,
            perplexity=cli_args.perplexity,
            max_iter=cli_args.max_iter,
            random_state=cli_args.random_state,
            train=False,
            show_plot=True,
        )
    else:
        parser.print_help()
