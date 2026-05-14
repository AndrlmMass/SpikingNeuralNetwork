import numpy as np


def get_contiguous_segment(indices):
    if len(indices) == 0:
        return None
    gaps = np.where(np.diff(indices) != 1)[0]
    segments = np.split(indices, gaps + 1)
    return max(segments, key=len)


def block_reduce(spikes, labels, block_size, reduce="sum"):
    if block_size <= 1:
        return spikes, labels

    valid_mask = labels >= 0
    spikes = spikes[valid_mask]
    labels = labels[valid_mask]

    num_blocks = spikes.shape[0] // block_size
    if num_blocks == 0:
        return spikes[:0], labels[:0]

    spikes = spikes[: num_blocks * block_size]
    labels = labels[: num_blocks * block_size]

    blocks = spikes.reshape(num_blocks, block_size, spikes.shape[1])
    if reduce == "mean":
        spikes_b = blocks.mean(axis=1)
    elif reduce == "sum":
        spikes_b = blocks.sum(axis=1)
    else:
        raise ValueError("reduce must be 'mean' or 'sum'")

    labels_b = np.zeros(num_blocks, dtype=int)
    for i in range(num_blocks):
        lab_block = labels[i * block_size : (i + 1) * block_size]
        labels_b[i] = np.argmax(np.bincount(lab_block))

    return spikes_b, labels_b
