from torchvision import datasets, transforms
from snntorch import spikegen
import pickle as pkl
import numpy as np
import torch
import os
from tqdm import tqdm  # Import tqdm for progress bars


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

        # Extract image tensors and labels
        print("Processing images and labels...")
        images, labels = zip(
            *[
                (mnist[i][0], mnist[i][1])
                for i in tqdm(range(len(mnist)), desc="Extracting samples")
            ]
        )
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
        spike_data_corrected = np.reshape(
            spike_data_corrected,
            (spike_data.shape[0] * spike_data.shape[1], spike_data_corrected.shape[2]),
        )

        # save training data in binary format with progress bar
        print("Saving spike data and labels...")
        os.makedirs("sdata", exist_ok=True)

        with tqdm(total=2, desc="Saving Data") as pbar:
            with open(file_name1, "wb") as file:
                pkl.dump(spike_data_corrected, file)
            pbar.update(1)

            with open(file_name2, "wb") as file:
                pkl.dump(spike_labels, file)
            pbar.update(1)

        return spike_data_corrected, spike_labels

    elif os.path.exists(file_name1):
        print("\rLoading existing data...", end="")
        with open(file_name1, "rb") as file:
            data = pkl.load(file)

        with open(file_name2, "rb") as file:
            labels = pkl.load(file)

        return data, labels
