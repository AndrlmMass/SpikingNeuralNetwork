from torchvision import datasets, transforms
from plot import plot_floats_and_spikes
from snntorch import spikegen
from tqdm import tqdm
import numpy as np
import json
import os
import torch


def create_data(
    pixel_size,
    num_steps,
    gain,
    offset,
    first_spike_time,
    time_var_input,
    num_images,
    add_breaks,
    break_lengths,
    noisy_data,
    noise_level,
    classes,
    plot_comparison,
    test_data_ratio,
    train_,
):
    # Define data parameters
    data_parameters = {"pixel_size": pixel_size, "train_": train_}

    # Define folder to load data
    folders = os.listdir("data/mdata")

    # Search for existing data gens
    if len(folders) > 0:
        for folder in folders:
            json_file_path = os.path.join(
                "data", "mdata", folder, "data_parameters.json"
            )

            with open(json_file_path, "r") as j:
                ex_params = json.loads(j.read())

            # Check if parameters are the same as the current ones
            if ex_params == data_parameters:
                data_dir = os.path.join("data/mdata", folder)
                download = False
                break
        else:
            download = True
    else:
        download = True

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
    if download == True:
        # generate random number to create unique folder
        rand_nums = np.random.randint(low=0, high=9, size=5)

        # Check if folder already exists
        while any(item in os.listdir("data") for item in rand_nums):
            rand_nums = np.random.randint(low=0, high=9, size=5)

        # Create folder to store data
        data_dir = os.path.join("data/mdata", str(rand_nums))
        os.makedirs(data_dir)

        # Save data parameters
        filepath = os.path.join(data_dir, "data_parameters.json")

        with open(filepath, "w") as outfile:
            json.dump(data_parameters, outfile)

    # Fetch MNIST dataset
    with tqdm(total=1, desc="Downloading MNIST") as pbar:
        mnist = datasets.MNIST(
            root=data_dir, train=train_, transform=transform, download=download
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
    test_images = int(num_images * test_data_ratio)

    # Limit number of images
    limited_images_train = images[:num_images]
    limited_labels_train = labels[:num_images]
    limited_images_test = images[num_images : test_images + num_images]
    limited_labels_test = labels[num_images : test_images + num_images]

    # convert to spikes with progress bar
    print("Converting images to spike trains...")
    spike_data_train = torch.zeros(size=limited_images_train.shape)
    spike_data_train = spike_data_train.repeat(num_steps, 1, 1, 1)
    for i in range(num_images):
        spike_data_train[i * num_steps : (i + 1) * num_steps] = spikegen.rate(
            limited_images_train[i],
            num_steps=num_steps,
            gain=gain,
            offset=offset,
            first_spike_time=first_spike_time,
            time_var_input=time_var_input,
        )

    spike_data_test = torch.zeros(size=limited_images_test.shape)
    spike_data_test = spike_data_test.repeat(num_steps, 1, 1, 1)
    for i in range(test_images):
        spike_data_test[i * num_steps : (i + 1) * num_steps] = spikegen.rate(
            limited_images_test[i],
            num_steps=num_steps,
            gain=gain,
            offset=offset,
            first_spike_time=first_spike_time,
            time_var_input=time_var_input,
        )

    # create true spiking labels
    spike_data_train = torch.zeros(size=limited_images_train.shape)
    spike_data_train = spike_data_train.repeat(num_steps, 1, 1, 1)
    for i in range(num_images):
        spike_data_train[i * num_steps : (i + 1) * num_steps] = spikegen.rate(
            limited_images_train[i],
            num_steps=num_steps,
            gain=gain,
            offset=offset,
            first_spike_time=first_spike_time,
            time_var_input=time_var_input,
        )

    # extend labels based on num_steps
    spike_labels_train = limited_labels_train.numpy().repeat(num_steps)
    spike_labels_test = limited_labels_test.numpy().repeat(num_steps)

    # remove unnecessary dimensions
    spike_data_train_corrected = spike_data_train.squeeze(2).flatten(start_dim=2)
    spike_data_test_corrected = spike_data_test.squeeze(2).flatten(start_dim=2)

    # merge second and first dimensions
    S_data_train = np.reshape(
        spike_data_train_corrected,
        (
            spike_data_train.shape[0] * spike_data_train.shape[1],
            spike_data_train_corrected.shape[2],
        ),
    )
    S_data_test = np.reshape(
        spike_data_test_corrected,
        (
            spike_data_test.shape[0] * spike_data_test.shape[1],
            spike_data_test_corrected.shape[2],
        ),
    )

    # convert S_data to numpy array
    S_data_train_conv = S_data_train.numpy()
    S_data_test_conv = S_data_test.numpy()

    if plot_comparison:
        plot_floats_and_spikes(
            images, S_data_train_conv, spike_labels_train, labels, num_steps
        )

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
            S_data_train_conv = np.insert(
                S_data_train_conv, start, break_activity, axis=0
            )
            spike_labels_train = np.insert(
                spike_labels_train, start, break_labels, axis=0
            )

            # update offset since we have inserted 'length' steps of break
            offset += length
    if noisy_data:
        # Convert the float array to an integer array first
        S_data_train_conv = S_data_train_conv.astype(int)
        # create break activity
        break_activity = (
            np.random.random(size=S_data_train_conv.shape) < noise_level
        ).astype(int)
        S_data_train_conv = S_data_train_conv | break_activity

        # save training data in binary format with progress bar

    return (
        S_data_train_conv,
        spike_labels_train,
        S_data_test_conv,
        spike_labels_test,
    )
