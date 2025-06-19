from torchvision import datasets, transforms
from plot import plot_floats_and_spikes
import torch.nn.functional as F
from snntorch import spikegen
import numpy as np
import torch
import os
import json

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
    use_validation_data=False,
    validation_split=0.2,
):
    """
    Optionally splits off a validation set from the training data if use_validation_data is True.
    Returns (train_data, train_labels, test_data, test_labels) as before, or
    (train_data, train_labels, val_data, val_labels, test_data, test_labels) if validation is enabled.
    When validation is enabled, the split is persistent and batches are indexed from the training set only.
    """
    # set transform rule
    transform = transforms.Compose(
        [
            transforms.Grayscale(),  # Ensure single channel
            transforms.Resize((pixel_size)),  # Resize to specified pixels
            transforms.ToTensor(),  # Convert to tensor
        ]
    )

    # Fetch MNIST dataset
    mnist_train = datasets.MNIST(
        root=data_dir, train=True, transform=transform, download=download
    )
    mnist_test = datasets.MNIST(
        root=data_dir, train=False, transform=transform, download=download
    )

    # Extract data
    samples_train = [
        (mnist_train[i][0], mnist_train[i][1]) for i in range(len(mnist_train))
    ]
    samples_test = [
        (mnist_test[i][0], mnist_test[i][1]) for i in range(len(mnist_test))
    ]
    np.random.seed(42)

    if use_validation_data:
        split_path = os.path.join(data_dir, "train_val_split.json")
        total_train = len(samples_train)
        val_size = int(total_train * validation_split)
        train_size = total_train - val_size
        # Try to load split from disk
        if os.path.exists(split_path):
            with open(split_path, "r") as f:
                split = json.load(f)
            train_indices = split["train_indices"]
            val_indices = split["val_indices"]
        else:
            indices = np.arange(total_train)
            np.random.shuffle(indices)
            train_indices = indices[:train_size].tolist()
            val_indices = indices[train_size:].tolist()
            with open(split_path, "w") as f:
                json.dump(
                    {"train_indices": train_indices, "val_indices": val_indices}, f
                )
        # For batching, select a slice from the training indices only
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        batch_train_indices = train_indices[idx_train : idx_train + num_images_train]
        images_train, labels_train = zip(
            *[samples_train[i] for i in batch_train_indices]
        )
        images_train = torch.stack(images_train)
        labels_train = torch.tensor(labels_train)
        # Validation set is always the same (use first num_images_test for batch size)
        batch_val_indices = val_indices[:num_images_test]
        images_val, labels_val = zip(*[samples_train[i] for i in batch_val_indices])
        images_val = torch.stack(images_val)
        labels_val = torch.tensor(labels_val)
    else:
        images_train, labels_train = zip(*samples_train)
        images_train = torch.stack(
            images_train[idx_train : idx_train + num_images_train]
        )
        labels_train = torch.tensor(
            labels_train[idx_train : idx_train + num_images_train]
        )
    images_test, labels_test = zip(*samples_test)
    images_test = torch.stack(images_test[idx_test : idx_test + num_images_test])
    labels_test = torch.tensor(labels_test[idx_test : idx_test + num_images_test])

    # Normalize spike intensity for each image
    target_sum = (pixel_size**2) * 0.1
    norm_images_train = torch.stack(
        [normalize_image(img=img, target_sum=target_sum) for img in images_train]
    )
    norm_images_test = torch.stack(
        [normalize_image(img=img, target_sum=target_sum) for img in images_test]
    )
    if use_validation_data:
        norm_images_val = torch.stack(
            [normalize_image(img=img, target_sum=target_sum) for img in images_val]
        )

    # Convert floats to poisson sequences
    S_data_train = torch.zeros(size=norm_images_train.shape)
    S_data_train = S_data_train.repeat(num_steps, 1, 1, 1)
    for i in range(norm_images_train.shape[0]):
        S_data_train[i * num_steps : (i + 1) * num_steps] = spikegen.rate(
            norm_images_train[i],
            num_steps=num_steps,
            gain=gain,
            offset=offset,
            first_spike_time=first_spike_time,
            time_var_input=time_var_input,
        )
    S_data_test = torch.zeros(size=norm_images_test.shape)
    S_data_test = S_data_test.repeat(num_steps, 1, 1, 1)
    for i in range(norm_images_test.shape[0]):
        S_data_test[i * num_steps : (i + 1) * num_steps] = spikegen.rate(
            norm_images_test[i],
            num_steps=num_steps,
            gain=gain,
            offset=offset,
            first_spike_time=first_spike_time,
            time_var_input=time_var_input,
        )
    if use_validation_data:
        S_data_val = torch.zeros(size=norm_images_val.shape)
        S_data_val = S_data_val.repeat(num_steps, 1, 1, 1)
        for i in range(norm_images_val.shape[0]):
            S_data_val[i * num_steps : (i + 1) * num_steps] = spikegen.rate(
                norm_images_val[i],
                num_steps=num_steps,
                gain=gain,
                offset=offset,
                first_spike_time=first_spike_time,
                time_var_input=time_var_input,
            )

    # Extend labels based on num_steps
    spike_labels_train = labels_train.numpy().repeat(num_steps)
    spike_labels_test = labels_test.numpy().repeat(num_steps)
    if use_validation_data:
        spike_labels_val = labels_val.numpy().repeat(num_steps)

    # Remove unnecessary dimensions
    S_data_train_corrected = S_data_train.squeeze(2).flatten(start_dim=2)
    S_data_test_corrected = S_data_test.squeeze(2).flatten(start_dim=2)
    if use_validation_data:
        S_data_val_corrected = S_data_val.squeeze(2).flatten(start_dim=2)

    # Merge second and first dimensions
    S_data_train = np.reshape(
        S_data_train_corrected,
        (
            S_data_train.shape[0] * S_data_train.shape[1],
            S_data_train_corrected.shape[2],
        ),
    )
    S_data_test = np.reshape(
        S_data_test_corrected,
        (
            S_data_test.shape[0] * S_data_test.shape[1],
            S_data_test_corrected.shape[2],
        ),
    )
    if use_validation_data:
        S_data_val = np.reshape(
            S_data_val_corrected,
            (
                S_data_val.shape[0] * S_data_val.shape[1],
                S_data_val_corrected.shape[2],
            ),
        )

    S_data_train = S_data_train.numpy()
    S_data_test = S_data_test.numpy()
    if use_validation_data:
        S_data_val = S_data_val.numpy()

    if plot_comparison:
        plot_floats_and_spikes(
            images_train, S_data_train, spike_labels_train, labels_train, num_steps
        )

    def add_breaks_to_data(
        S_data, spike_labels, num_images, num_steps, pixel_size, break_lengths
    ):
        data_list = []
        labels_list = []
        for img in range(num_images):
            image_slice = S_data[img * num_steps : (img + 1) * num_steps]
            label_slice = spike_labels[img * num_steps : (img + 1) * num_steps]
            data_list.append(image_slice)
            labels_list.append(label_slice)
            length = np.random.choice(break_lengths, size=1)[0]
            break_activity = np.zeros((length, pixel_size**2), dtype=int)
            break_labels = np.full(length, fill_value=-1)
            data_list.append(break_activity)
            labels_list.append(break_labels)
        S_data_out = np.concatenate(data_list, axis=0)
        spike_labels_out = np.concatenate(labels_list, axis=0)
        return S_data_out, spike_labels_out

    if add_breaks:
        S_data_train, spike_labels_train = add_breaks_to_data(
            S_data_train,
            spike_labels_train,
            num_images_train,
            num_steps,
            pixel_size,
            break_lengths,
        )
        S_data_test, spike_labels_test = add_breaks_to_data(
            S_data_test,
            spike_labels_test,
            num_images_test,
            num_steps,
            pixel_size,
            break_lengths,
        )
        if use_validation_data:
            S_data_val, spike_labels_val = add_breaks_to_data(
                S_data_val,
                spike_labels_val,
                num_images_test,
                num_steps,
                pixel_size,
                break_lengths,
            )

    if noisy_data:
        S_data_train = S_data_train.astype(int)
        break_activity = (
            np.random.random(size=S_data_train.shape) < noise_level
        ).astype(int)
        S_data_train = S_data_train | break_activity

    if use_validation_data:
        return (
            S_data_train,
            spike_labels_train,
            S_data_val,
            spike_labels_val,
            S_data_test,
            spike_labels_test,
        )
    else:
        return (
            S_data_train,
            spike_labels_train,
            S_data_test,
            spike_labels_test,
        )
