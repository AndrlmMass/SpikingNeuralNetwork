from torchvision import datasets, transforms
from plot import plot_floats_and_spikes
import torch.nn.functional as F
from snntorch import spikegen
import numpy as np
import torch

import os

os.environ["TQDM_DISABLE"] = "True"


def normalize_image(img, target_sum=1.0):
    current_sum = img.sum()
    return img * (target_sum / current_sum) if current_sum > 0 else img


def create_data(
    pixel_size,
    num_steps,
    gain,
    offset,
    first_spike_time,
    time_var_input,
    num_images_train,
    num_images_test,
    add_breaks,
    break_lengths,
    noisy_data,
    noise_level,
    plot_comparison,
    download,
    data_dir,
    idx_train,
    idx_test,
):
    # set transform rule
    transform = transforms.Compose(
        [
            transforms.Grayscale(),  # Ensure single channel
            transforms.Resize((pixel_size)),  # Resize to specified pixels
            transforms.ToTensor(),  # Convert to tensor
        ]
    )

    # GET TRAINING DATA

    # Fetch MNIST dataset
    # TRAIN#
    mnist_train = datasets.MNIST(
        root=data_dir, train=True, transform=transform, download=download
    )
    # TEST#
    mnist_test = datasets.MNIST(
        root=data_dir, train=False, transform=transform, download=download
    )

    # Extract data
    # TRAIN#
    samples_train = [
        (mnist_train[i][0], mnist_train[i][1]) for i in range(len(mnist_train))
    ]
    # TEST#
    samples_test = [
        (mnist_test[i][0], mnist_test[i][1]) for i in range(len(mnist_test))
    ]

    # Unpack and convert to tensors
    # TRAIN#
    images_train, labels_train = zip(*samples_train)
    images_train = torch.stack(
        images_train[idx_train : idx_train + num_images_train]
    )  # Shape: [num_samples, 1, 28, 28]
    labels_train = torch.tensor(
        labels_train[idx_train : idx_train + num_images_train]
    )  # Shape: [num_samples]
    # TEST#
    images_test, labels_test = zip(*samples_test)
    images_test = torch.stack(
        images_test[idx_test : idx_test + num_images_test]
    )  # Shape: [num_samples, 1, 28, 28]
    labels_test = torch.tensor(
        labels_test[idx_test : idx_test + num_images_test]
    )  # Shape: [num_samples]
    k = 4

    # Normalize spike intensity for each image
    target_sum = (
        pixel_size**2
    ) * 0.1  # THIS NEEDS TO BE REVISED. IDK WHY I CHOSE 0.1...
    # TRAIN#
    norm_images_train = torch.stack(
        [normalize_image(img=img, target_sum=target_sum) for img in images_train]
    )
    # TEST#
    norm_images_test = torch.stack(
        [normalize_image(img=img, target_sum=target_sum) for img in images_test]
    )

    # Convert floats to poisson sequences
    # TRAIN#
    S_data_train = torch.zeros(size=norm_images_train.shape)
    S_data_train = S_data_train.repeat(num_steps, 1, 1, 1)
    for i in range(num_images_train):
        S_data_train[i * num_steps : (i + 1) * num_steps] = spikegen.rate(
            norm_images_train[i],
            num_steps=num_steps,
            gain=gain,
            offset=offset,
            first_spike_time=first_spike_time,
            time_var_input=time_var_input,
        )
    # TEST#
    S_data_test = torch.zeros(size=norm_images_test.shape)
    S_data_test = S_data_test.repeat(num_steps, 1, 1, 1)
    for i in range(num_images_test):
        S_data_test[i * num_steps : (i + 1) * num_steps] = spikegen.rate(
            norm_images_test[i],
            num_steps=num_steps,
            gain=gain,
            offset=offset,
            first_spike_time=first_spike_time,
            time_var_input=time_var_input,
        )

    # Extend labels based on num_steps
    # TRAIN#
    spike_labels_train = labels_train.numpy().repeat(num_steps)
    # TEST#
    spike_labels_test = labels_test.numpy().repeat(num_steps)

    # Remove unnecessary dimensions
    # TRAIN#
    S_data_train_corrected = S_data_train.squeeze(2).flatten(start_dim=2)
    # TEST#
    S_data_test_corrected = S_data_test.squeeze(2).flatten(start_dim=2)

    # Merge second and first dimensions
    # TRAIN#
    S_data_train = np.reshape(
        S_data_train_corrected,
        (
            S_data_train.shape[0] * S_data_train.shape[1],
            S_data_train_corrected.shape[2],
        ),
    )
    # TEST#
    S_data_test = np.reshape(
        S_data_test_corrected,
        (
            S_data_test.shape[0] * S_data_test.shape[1],
            S_data_test_corrected.shape[2],
        ),
    )

    # convert S_data to numpy array
    S_data_train = S_data_train.numpy()
    S_data_test = S_data_test.numpy()

    if plot_comparison:
        plot_floats_and_spikes(
            images_train, S_data_train, spike_labels_train, labels_train, num_steps
        )

    if add_breaks:
        train_data_list = []
        train_labels_list = []

        offset = 0
        for img in range(num_images_train):
            # 1. Get the original slice for this image
            image_slice = S_data_train[img * num_steps : (img + 1) * num_steps]
            label_slice = spike_labels_train[img * num_steps : (img + 1) * num_steps]

            # 2. Add them to our list
            train_data_list.append(image_slice)
            train_labels_list.append(label_slice)

            # 3. Create the break
            length = np.random.choice(break_lengths, size=1)[0]
            break_activity = np.zeros((length, pixel_size**2), dtype=int)
            break_labels = np.full(length, fill_value=-1)

            # 4. Append the break
            train_data_list.append(break_activity)
            train_labels_list.append(break_labels)

        # Finally, concatenate once
        S_data_train = np.concatenate(train_data_list, axis=0)
        spike_labels_train = np.concatenate(train_labels_list, axis=0)

        test_data_list = []
        test_labels_list = []

        offset = 0
        for img in range(num_images_test):
            # 1. Get the original slice for this image
            image_slice = S_data_test[img * num_steps : (img + 1) * num_steps]
            label_slice = spike_labels_test[img * num_steps : (img + 1) * num_steps]

            # 2. Add them to our list
            test_data_list.append(image_slice)
            test_labels_list.append(label_slice)

            # 3. Create the break
            length = np.random.choice(break_lengths, size=1)[0]
            break_activity = np.zeros((length, pixel_size**2), dtype=int)
            break_labels = np.full(length, fill_value=-1)

            # 4. Append the break
            test_data_list.append(break_activity)
            test_labels_list.append(break_labels)

        # Finally, concatenate once
        S_data_test = np.concatenate(test_data_list, axis=0)
        spike_labels_test = np.concatenate(test_labels_list, axis=0)

    if noisy_data:
        # Convert the float array to an integer array first
        S_data_train = S_data_train.astype(int)

        # create break activity
        break_activity = (
            np.random.random(size=S_data_train.shape) < noise_level
        ).astype(int)
        S_data_train = S_data_train | break_activity

    return (
        S_data_train,
        spike_labels_train,
        S_data_test,
        spike_labels_test,
    )
