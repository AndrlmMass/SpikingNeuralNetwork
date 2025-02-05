from torchvision import datasets, transforms
from plot import plot_floats_and_spikes
from snntorch import spikegen
from tqdm import tqdm
import pickle as pkl
import numpy as np
import torch
import os


def create_data(
    pixel_size,
    num_steps,
    gain,
    offset,
    first_spike_time,
    time_var_input,
    download,
    num_images,
    recreate,
    add_breaks,
    break_lengths,
    noisy_data,
    noise_level,
    classes,
    plot_comparison,
):
    file_name1 = "sdata/MNIST_spiked.pkl"
    file_name2 = "sdata/MNIST_spiked_labels.pkl"

    if recreate:
        # set transform rule
        transform = transforms.Compose(
            [
                transforms.Grayscale(),  # Ensure single channel
                transforms.Resize((pixel_size)),  # Resize to specified pixels
                transforms.ToTensor(),  # Convert to tensor
            ]
        )

        # get dataset with progress bar
        print("Downloading MNIST dataset...")
        with tqdm(total=1, desc="Downloading MNIST") as pbar:
            mnist = datasets.MNIST(
                root="data", train=True, transform=transform, download=download
            )
            pbar.update(1)  # Update progress bar when download completes

        # Filter the dataset so that only samples with allowed labels are included
        filtered_samples = [
            (mnist[i][0], mnist[i][1])
            for i in tqdm(range(len(mnist)), desc="Extracting samples")
            if mnist[i][1] in classes
        ]

        # Now unpack and convert to tensors
        images, labels = zip(*filtered_samples)
        images = torch.stack(images)  # Shape: [num_samples, 1, 28, 28]
        labels = torch.tensor(labels)  # Shape: [num_samples]

        # Limit number of images
        limited_images = images[:num_images]
        limited_labels = labels[:num_images]

        # convert to spikes with progress bar
        print("Converting images to spike trains...")
        spike_data = spikegen.rate(
            limited_images,
            num_steps=num_steps,
            gain=gain,
            offset=offset,
            first_spike_time=first_spike_time,
            time_var_input=time_var_input,
        )

        # extend labels based on num_steps
        spike_labels = limited_labels.numpy().repeat(num_steps)

        # remove unnecessary dimensions
        spike_data_corrected = spike_data.squeeze(2).flatten(start_dim=2)

        # merge second and first dimensions
        S_data = np.reshape(
            spike_data_corrected,
            (spike_data.shape[0] * spike_data.shape[1], spike_data_corrected.shape[2]),
        )

        # convert S_data to numpy array
        S_data_conv = S_data.numpy()

        if plot_comparison:
            plot_floats_and_spikes(images, S_data_conv, spike_labels, labels)

        if add_breaks:
            # Keep track of how much the array has grown so that the index is correct
            offset = 0

            for img in range(num_images):
                # choose random break length
                length = np.random.choice(break_lengths, size=1)[0]

                # define break spiking activity
                break_activity = np.zeros((length, int(pixel_size**2)), dtype=int)

                # create break labels
                break_labels = np.full(length, fill_value=-1)

                # compute the start position for inserting break
                start = img * num_steps + offset

                # IMPORTANT: reassign the output of np.insert
                S_data_conv = np.insert(S_data_conv, start, break_activity, axis=0)
                spike_labels = np.insert(spike_labels, start, break_labels, axis=0)

                # update offset since we have inserted 'length' steps of break
                offset += length
        if noisy_data:
            # Convert the float array to an integer array first
            S_data_conv = S_data_conv.astype(int)
            # create break activity
            break_activity = (
                np.random.random(size=S_data_conv.shape) < noise_level
            ).astype(int)
            S_data_conv = S_data_conv | break_activity

        # save training data in binary format with progress bar
        print("Saving spike data and labels...")
        os.makedirs("sdata", exist_ok=True)

        with tqdm(total=2, desc="Saving Data") as pbar:
            with open(file_name1, "wb") as file:
                pkl.dump(S_data_conv, file)
            pbar.update(1)

            with open(file_name2, "wb") as file:
                pkl.dump(spike_labels, file)
            pbar.update(1)

        return S_data_conv, spike_labels

    elif os.path.exists(file_name1):
        print("\rLoading existing data...", end="")
        with open(file_name1, "rb") as file:
            data = pkl.load(file)

        with open(file_name2, "rb") as file:
            labels = pkl.load(file)

        return data, labels
