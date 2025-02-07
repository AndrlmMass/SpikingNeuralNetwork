import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib

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

    # Visualize the results:
    plt.figure(figsize=(10, 8))
    for label in np.unique(segment_labels):
        indices = segment_labels == label
        plt.scatter(
            tsne_results[indices, 0], tsne_results[indices, 1], label=f"Class {label}"
        )
    plt.title("t-SNE results")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.legend()
    plt.show()
