import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

matplotlib.use("TkAgg")
warnings.filterwarnings("error", category=RuntimeWarning)


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
    segments = []
    segment_labels = []
    start = 0  # start index for the current segment

    # Iterate through the labels to detect change points.
    for t in range(1, len(labels)):
        if labels[t] != labels[t - 1]:
            # End of the current segment.
            current_label = labels[t - 1]
            # Process only if the current label is not -1.
            if current_label != -1 and current_label != -2:
                segment = spikes[start:t]
                # Compute feature for the segment (here, the mean firing rate for each neuron).
                feature_vector = np.mean(segment, axis=0)
                segments.append(feature_vector)
                segment_labels.append(current_label)
            # Update the start index for the next segment.
            start = t

    # Handle the final segment.
    if start < len(labels):
        current_label = labels[-1]
        if current_label != -1 and current_label != -2:
            segment = spikes[start:]
            feature_vector = np.mean(segment, axis=0)
            segments.append(feature_vector)
            segment_labels.append(current_label)

    return np.array(segments), np.array(segment_labels)


def t_SNE(
    spikes,
    labels_spike,
    n_components,
    perplexity,
    max_iter,
    random_state,
    train,
):
    # Now, bin the spikes using the labels, skipping breaks:
    features, segment_labels = bin_spikes_by_label_no_breaks(spikes, labels_spike)

    # Apply t-SNE on the computed features:

    # Ensure that perplexity is less than the number of segments.
    perplexity = min(30, len(features) - 1)

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=random_state,
    )
    tsne_results = tsne.fit_transform(features)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")

    # Visualize the results:
    for label in np.unique(segment_labels):
        indices = segment_labels == label
        ax.scatter(
            tsne_results[indices, 0],
            tsne_results[indices, 1],
            label=f"Class {label}",
        )
    if train:
        title = "from training"
    else:
        title = "from testing"

    plt.title(f"t-SNE results " + title)
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.legend()
    plt.show()


def PCA_analysis(
    spikes,
    labels_spike,
    pca_variance,
    random_state,
):
    # Bin the spikes using the labels (assuming this function is defined elsewhere)
    features, segment_labels = bin_spikes_by_label_no_breaks(spikes, labels_spike)

    # Create a PCA instance.
    pca = PCA(n_components=pca_variance, random_state=random_state)
    pca_results = pca.fit_transform(features)

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

    """Calculate the spiking rates for each item presentation"""
    # Remove break data
    break_mask = labels_train != -1
    labels_train = labels_train[break_mask]
    spikes_train = spikes_train[break_mask, :]

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
        if not current_mask.all():
            count += 1
            continue
        mean_spikes = np.mean(spikes_train[i : i + num_steps][current_mask, :], axis=0)
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

    # standardize rates
    spike_train_rates_std = StandardScaler().fit_transform(spike_train_rates)

    """ Perform PCA on the binned data """
    # Create a PCA instance
    pca = PCA(n_components=pca_variance, random_state=random_state)
    pca.fit(spike_train_rates_std)
    n_components = pca.n_components_
    scores_train_pca = pca.transform(spike_train_rates_std)

    # Calculate centroids and WCSS
    centroids = np.zeros((n_components, num_classes))
    wcss_arr = np.zeros(num_classes)
    for c in range(num_classes):
        indices = np.where(labels_train_unique == c)[0]
        if indices.size != 0:
            centroids[:, c] = np.mean(scores_train_pca[indices], axis=0)
            values = scores_train_pca[indices]
            for dim in range(n_components):
                delta_wcss = np.sum((centroids[dim, c] - values[:, dim]) ** 2)
                wcss_arr[c] += delta_wcss
    WCSS_train = np.sum(wcss_arr)

    # Estimate BCSS
    overall_mean = np.mean(scores_train_pca, axis=0)
    n_j = scores_train_pca.shape[0]
    BCSS_train = 0
    for c in range(num_classes):
        for dim in range(n_components):
            delta_bcss = n_j * (centroids[dim, c] - overall_mean[dim]) ** 2
            BCSS_train += delta_bcss

    # Calculate clustering coefficient
    """
    phi = (BCSS / (k-1)) / (WCSS / (n-k))
    """
    phi_train = (BCSS_train / (num_classes - 1)) / (
        WCSS_train / (scores_train_pca.shape[0] - num_classes + small_num)
    )

    """
    Second: Project test data onto precomputed PCA and K-means centroids
    """

    """Calculate the spiking rates for each item presentation"""
    # Remove break data
    break_mask = labels_test != -1
    labels_test = labels_test[break_mask]
    spikes_test = spikes_test[break_mask, :]

    # Calculate rate for each item
    spike_test_rates = []
    labels_test_unique = []

    """
    Note that we are currently skipping sleep-patterns. 
    Maybe this should be its own array, but then we miss part of the sequence
    """
    sleep_mask = labels_test != -2
    for i in range(0, labels_test.shape[0], num_steps):
        # skip non_sleep spiking activity
        current_mask = sleep_mask[i : i + num_steps]
        if not current_mask.all():
            continue
        mean_spikes = np.mean(spikes_test[i : i + num_steps][current_mask, :], axis=0)
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

    # standardize rates
    spike_test_rates_std = StandardScaler().fit_transform(spike_test_rates)

    """ Perform PCA on the binned data """
    # Create a PCA instance.
    scores_test_pca = pca.transform(spike_test_rates_std)

    # Calculate centroids and WCSS
    wcss_arr = np.zeros(num_classes)
    for c in range(num_classes):
        indices = np.where(labels_test_unique == c)[0]
        if indices.size != 0:
            values = scores_test_pca[indices]
            for dim in range(n_components):
                delta_wcss = np.sum((values[:, dim] - centroids[dim, c]) ** 2)
                wcss_arr[c] += delta_wcss
    WCSS_test = np.sum(wcss_arr)

    # Estimate bcss
    n_j = scores_test_pca.shape[0]
    overall_mean = np.mean(scores_test_pca, axis=0)
    BCSS_test = 0
    for c in range(num_classes):
        for dim in range(n_components):
            delta_bcss = n_j * (centroids[dim, c] - overall_mean[dim]) ** 2
            BCSS_test += delta_bcss

    # Calculate clustering coefficient
    phi_test = (BCSS_test / (num_classes - 1)) / (
        WCSS_test / (scores_test_pca.shape[0] - num_classes + small_num)
    )

    BCSS_test_scaled = BCSS_test / (num_classes - 1 + small_num)
    BCSS_train_scaled = BCSS_train / (num_classes - 1 + small_num)
    WCSS_test_scaled = WCSS_test / (scores_test_pca.shape[0] - num_classes + small_num)
    WCSS_train_scaled = WCSS_train / (
        scores_train_pca.shape[0] - num_classes + small_num
    )

    return (
        phi_train,
        phi_test,
        WCSS_train_scaled,
        WCSS_test_scaled,
        BCSS_train_scaled,
        BCSS_test_scaled,
    )
