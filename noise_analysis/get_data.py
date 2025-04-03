from torchvision import datasets, transforms
from plot import plot_floats_and_spikes
import torch.nn.functional as F
from snntorch import spikegen
from tqdm import tqdm
import numpy as np
import torch


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
    num_images,
    add_breaks,
    gain_labels,
    break_lengths,
    noisy_data,
    noise_level,
    classes,
    N_classes,
    plot_comparison,
    test_data_ratio,
    true_labels,
    download,
    data_dir,
    train_,
):
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
    if test_images % N_classes != 0:
        test_images = N_classes * round(test_images / N_classes)

    # Build a balanced dataset
    balanced_images = []
    balanced_labels = []
    num_images_per_class = num_images // N_classes

    for c in classes:
        # indices for samples belonging to class c
        c_indices = (labels == c).nonzero().flatten()
        # shuffle these indices
        c_indices = c_indices[torch.randperm(len(c_indices))]
        # select the first N
        c_indices = c_indices[:num_images_per_class]

        balanced_images.append(images[c_indices])
        balanced_labels.append(labels[c_indices])

    # Concatenate images and labels from all classesbalanced_labels
    balanced_images_train = torch.cat(balanced_images, dim=0)
    balanced_labels_train = torch.cat(balanced_labels, dim=0)

    # Shuffle everything again one last time
    perm = torch.randperm(balanced_images_train.size(0))
    limited_images_train = balanced_images_train[perm]
    limited_labels_train = balanced_labels_train[perm]

    # Build a balanced dataset
    balanced_images2 = []
    balanced_labels2 = []
    num_images_per_class_test = test_images // N_classes

    for c in classes:
        # indices for samples belonging to class c
        c_indices = (labels == c).nonzero().flatten()
        # shuffle these indices
        c_indices = c_indices[torch.randperm(len(c_indices))]
        # select the first N
        c_indices = c_indices[num_images : num_images + num_images_per_class_test]

        balanced_images2.append(images[c_indices])
        balanced_labels2.append(labels[c_indices])

    # Concatenate images and labels from all classesbalanced_labels
    balanced_images_test = torch.cat(balanced_images2, dim=0)
    balanced_labels_test = torch.cat(balanced_labels2, dim=0)

    # Shuffle everything again one last time
    perm = torch.randperm(balanced_images_test.size(0))
    limited_images_test = balanced_images_test[perm]
    limited_labels_test = balanced_labels_test[perm]

    # normalize spike intensity for each image
    target_sum = (pixel_size**2) * 0.1
    limited_images_train = torch.stack(
        [
            normalize_image(img=img, target_sum=target_sum)
            for img in limited_images_train
        ]
    )
    limited_images_test = torch.stack(
        [normalize_image(img=img, target_sum=target_sum) for img in limited_images_test]
    )

    # one-hot encode labels
    if true_labels:

        one_hot_train_pos = F.one_hot(limited_labels_train, num_classes=N_classes)
        one_hot_train_neg = 1 - one_hot_train_pos
        labels_true = torch.concatenate([one_hot_train_pos, one_hot_train_neg], axis=1)

        # create true spiking labels train
        labels_true_r_train = np.zeros((num_images * num_steps, 2 * N_classes))
        for i in range(num_images):
            labels_true_r_train[i * num_steps : (i + 1) * num_steps] = spikegen.rate(
                labels_true[i],
                num_steps=num_steps,
                gain=gain_labels,
                offset=offset,
                first_spike_time=first_spike_time,
                time_var_input=time_var_input,
            )

        # create true spiking labels train
        one_hot_train_pos = F.one_hot(limited_labels_test, num_classes=N_classes)
        one_hot_train_neg = 1 - one_hot_train_pos
        labels_true = torch.concatenate([one_hot_train_pos, one_hot_train_neg], axis=1)

        labels_true_r_test = np.zeros((test_images * num_steps, N_classes * 2))
        for i in range(test_images):
            labels_true_r_test[i * num_steps : (i + 1) * num_steps] = spikegen.rate(
                labels_true[i],
                num_steps=num_steps,
                gain=gain_labels,
                offset=offset,
                first_spike_time=first_spike_time,
                time_var_input=time_var_input,
            )
    else:
        labels_true_r_train = None
        labels_true_r_test = None

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
    S_data_train = S_data_train.numpy()
    S_data_test = S_data_test.numpy()

    # print sum spikes for each class to compare
    """
    This will take the sum of all the training, not just the first example
    """

    # Check that all classes are presented equally in frequency

    for cl in classes:
        indices = spike_labels_train == cl
        sum_ = np.sum(S_data_train[indices])
        print(cl, sum_)

    limited_images_train_ = limited_images_train.squeeze(1).flatten(start_dim=2).numpy()
    limited_labels_train_ = limited_labels_train.numpy()
    for cl in classes:
        indices = limited_labels_train_ == cl
        ind = np.where(indices == True)[0]
        sum_ = np.sum(limited_images_train_[ind])
        print(cl, sum_)

    if plot_comparison:
        plot_floats_and_spikes(
            images, S_data_train, spike_labels_train, labels, num_steps
        )

    if add_breaks:
        train_data_list = []
        train_labels_list = []
        if true_labels:
            train_true_labels_list = []

        offset = 0
        for img in range(num_images):
            # 1. Get the original slice for this image
            image_slice = S_data_train[img * num_steps : (img + 1) * num_steps]
            label_slice = spike_labels_train[img * num_steps : (img + 1) * num_steps]

            # 2. Add them to our list
            train_data_list.append(image_slice)
            train_labels_list.append(label_slice)
            if true_labels:
                train_true_labels_list.append(
                    labels_true_r_train[img * num_steps : (img + 1) * num_steps]
                )

            # 3. Create the break
            length = np.random.choice(break_lengths, size=1)[0]
            break_activity = np.zeros((length, pixel_size**2), dtype=int)
            break_labels = np.full(length, fill_value=-1)

            # 4. Append the break
            train_data_list.append(break_activity)
            train_labels_list.append(break_labels)
            if true_labels:
                break_true_labels1 = np.zeros((length, N_classes))
                break_true_labels2 = np.ones((length, N_classes))
                break_true_labels = np.concatenate(
                    (break_true_labels1, break_true_labels2), axis=1
                )
                train_true_labels_list.append(break_true_labels)

        # Finally, concatenate once
        S_data_train = np.concatenate(train_data_list, axis=0)
        spike_labels_train = np.concatenate(train_labels_list, axis=0)
        if true_labels:
            labels_true_r_train = np.concatenate(train_true_labels_list, axis=0)

        test_data_list = []
        test_labels_list = []
        if true_labels:
            test_true_labels_list = []

        offset = 0
        for img in range(test_images):
            # 1. Get the original slice for this image
            image_slice = S_data_test[img * num_steps : (img + 1) * num_steps]
            label_slice = spike_labels_test[img * num_steps : (img + 1) * num_steps]

            # 2. Add them to our list
            test_data_list.append(image_slice)
            test_labels_list.append(label_slice)
            if true_labels:
                test_true_labels_list.append(
                    labels_true_r_test[img * num_steps : (img + 1) * num_steps]
                )

            # 3. Create the break
            length = np.random.choice(break_lengths, size=1)[0]
            break_activity = np.zeros((length, pixel_size**2), dtype=int)
            break_labels = np.full(length, fill_value=-1)

            # 4. Append the break
            test_data_list.append(break_activity)
            test_labels_list.append(break_labels)
            if true_labels:
                break_true_labels1 = np.zeros((length, N_classes))
                break_true_labels2 = np.ones((length, N_classes))
                break_true_labels = np.concatenate(
                    (break_true_labels1, break_true_labels2), axis=1
                )
                test_true_labels_list.append(break_true_labels)

        # Finally, concatenate once
        S_data_test = np.concatenate(test_data_list, axis=0)
        spike_labels_test = np.concatenate(test_labels_list, axis=0)
        if true_labels:
            labels_true_r_test = np.concatenate(test_true_labels_list, axis=0)
        else:
            labels_true_r_test = None

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
        labels_true_r_train,
        labels_true_r_test,
    )
