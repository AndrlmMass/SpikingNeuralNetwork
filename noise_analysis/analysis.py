import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

matplotlib.use("TkAgg")


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
    plt.title("t-SNE results")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.legend()
    plt.show()


def PCA_analysis(
    spikes,
    labels_spike,
    n_components,
    random_state,
):
    # Bin the spikes using the labels (assuming this function is defined elsewhere)
    features, segment_labels = bin_spikes_by_label_no_breaks(spikes, labels_spike)

    # Create a PCA instance.
    pca = PCA(n_components=n_components, random_state=random_state)
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
