import numpy as np


def replace_duplicates_with_unique(input_indices, num_neurons):
    unique, counts = np.unique(input_indices, return_counts=True)
    duplicates = unique[counts > 1]

    for dup in duplicates:
        dup_indices = np.where(input_indices == dup)[
            0
        ]  # Indices of duplicates in input_indices
        for d_idx in dup_indices[1:]:  # Skip the first occurrence, replace the rest
            new_val = np.random.choice(
                [i for i in range(num_neurons) if i not in input_indices]
            )
            input_indices[d_idx] = new_val

    # Ensuring all elements are unique now
    assert len(np.unique(input_indices)) == len(input_indices), "Duplicates remain!"
    return input_indices


def generate_small_world_network_power_law(
    num_neurons, excit_inhib_ratio, alpha, num_input_neurons, num_timesteps, num_items
):
    n_rows, n_cols = num_neurons, num_neurons

    # Generate power-law distribution for the probability of connections
    connection_probabilities = np.random.power(alpha, size=n_rows * n_cols)

    # Initialize the arrays for weights and signs
    weight_array = np.zeros((n_rows, n_cols, num_items))
    weight_array[:, :, 0] = np.ones((n_rows, n_cols))
    # Fill diagonal for 3d array
    for j in range(num_items):
        np.fill_diagonal(weight_array[:, :, j], 0)

    # Add weights to input neurons
    input_indices = np.random.choice(np.arange(num_neurons), num_input_neurons)
    input_indices = replace_duplicates_with_unique(input_indices, num_neurons)
    weight_array[input_indices, :, 0] = np.zeros(shape=num_neurons)

    # Assign weights and signs based on connection probability
    for i in range(n_rows):
        for j in range(n_cols):
            if weight_array[i, j, 0] != 0:
                if connection_probabilities[i * n_cols + j] > np.random.rand():
                    # Assign weights
                    const = 1 if np.random.rand() < excit_inhib_ratio else -1
                    weight_array[i, j, 0] = round(np.random.rand() / 10 * const, 4)
                    weight_array[j, i, 0] = 0

                else:
                    weight_array[i, j, 0] = 0

    # Add connections to neurons without post-synaptic connections
    if np.any(np.all(weight_array[:, :, 0] == 0, axis=0)):
        for j in range(num_neurons):
            if np.all(weight_array[:, j, 0] == 0):
                idx = np.random.choice(num_neurons)
                if idx == j:
                    idx = np.random.choice(num_neurons)
                else:
                    const = 1 if np.random.rand() < excit_inhib_ratio else -1
                    weight_array[idx, j, 0] = round(np.random.rand() / 10 * const, 4)
            if np.any(np.all(weight_array[:, :, 0] == 0, axis=0)) == False:
                break

    # Calculate ratio of excitatory to inhibitory connections
    var2 = round(
        np.sum(weight_array[:, :, 0] > 0) / np.sum(weight_array[:, :, 0] != 0), 2
    )
    print(f"This is the current ratio of positive edges to all edges: {var2}")
    return weight_array, input_indices


import numpy as np


def encode_input_poisson(
    input_data, num_timesteps, num_neurons, num_items, dt, input_scaler
):
    # Extract labels and input features
    labels = input_data[:, -1]
    input_features = input_data[:, 0:num_neurons]

    # Correct the dimensions for the 3D-array: timesteps x neurons x items
    poisson_input = np.zeros((num_timesteps, num_neurons, num_items))

    for i in range(num_items):  # Iterating over items
        for j in range(num_neurons):  # Iterating over neurons
            # Calculate the mean spike count for the Poisson distribution
            lambda_poisson = input_features[i, j] * dt * input_scaler

            # Generate spikes using Poisson distribution
            for t in range(num_timesteps):
                spike_count = np.random.poisson(lambda_poisson)
                poisson_input[t, j, i] = 1 if spike_count > 0 else 0

    return poisson_input, labels


def generate_multidimensional_data(
    num_classes,
    base_mean,
    mean_increment,
    variance,
    num_samples_per_class,
    features,
    num_timesteps,
    dt,
    input_scaler,
):
    print(num_samples_per_class)
    combined_pts = []
    combined_labels = []

    for class_id in range(num_classes):
        mean = base_mean + mean_increment * class_id
        cov = np.eye(features) * variance
        mean_vector = np.full(features, mean)

        # Generate data points for this class
        pts = np.random.multivariate_normal(mean_vector, cov, num_samples_per_class)
        labels = np.full(num_samples_per_class, class_id, dtype=int)

        combined_pts.append(pts)
        combined_labels.append(labels)

    # Combine and shuffle the datasets
    combined_pts = np.vstack(combined_pts)
    combined_labels = np.concatenate(combined_labels)
    combined_data = np.column_stack((combined_pts, combined_labels))
    np.random.shuffle(combined_data)

    # Adjust num_items to match the total number of samples
    num_items = num_classes * num_samples_per_class
    # Convert float-based array to spike-based
    data, classes = encode_input_poisson(
        combined_data, num_timesteps, features, num_items, dt, input_scaler
    )

    return data, classes.astype(int)
