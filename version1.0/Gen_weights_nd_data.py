import numpy as np


class gen_weights:
    def gen_StimE(self, radius, N_input_neurons, N_excit_neurons):
        input_shape = int(np.sqrt(N_input_neurons))
        circle_pos = np.arange(N_input_neurons).reshape(input_shape, input_shape)
        circle_pos_valid = circle_pos

        if circle_pos_valid.size == 0:
            raise ValueError("circle_pos_valid has invalid shape")

        circle_pos_flat = circle_pos_valid.flatten()
        circle_draws = np.random.choice(a=circle_pos_flat, size=N_excit_neurons)
        StimE_weights = np.zeros((N_input_neurons, N_excit_neurons))

        for j in range(N_excit_neurons):
            center_idx = np.argwhere(circle_pos == circle_draws[j])[
                0
            ]  # Find the 2D index of the center

            # Calculate the bounds for slicing around the center with the given radius
            # Ensure bounds are within the array limits
            row_start = max(0, center_idx[0] - radius)
            row_end = min(input_shape, center_idx[0] + radius + 1)
            col_start = max(0, center_idx[1] - radius)
            col_end = min(input_shape, center_idx[1] + radius + 1)

            # Example operation: for each selected position, set a weight in EE_weights
            for row in range(row_start, row_end):
                for col in range(col_start, col_end):
                    StimE_weights[circle_pos[row, col], j] = np.random.uniform(
                        low=0, high=1
                    )

        return StimE_weights

    def Gen_EE(N_excit_neurons, alpha, time):
        # Define n_rows and n_cols
        n_rows, n_cols = N_excit_neurons, N_excit_neurons

        # Generate power-law distribution for the probability of connections
        connection_probabilities = np.random.power(alpha, size=n_rows * n_cols)

        # Initialize the arrays for weights and signs
        EE_weights = np.zeros((n_rows, n_cols, time))
        EE_weights[:, :, 0] = np.ones((n_rows, n_cols))

        # Fill diagonal for 3d array with zeros
        for j in range(time):
            np.fill_diagonal(EE_weights[:, :, j], 0)

        # Assign weights and signs based on connection probability
        for i in range(n_rows):
            for j in range(n_cols):
                if EE_weights[i, j, 0] != 0:
                    if connection_probabilities[i * n_cols + j] > np.random.rand():
                        # Assign weight
                        EE_weights[i, j, 0] = round(np.random.rand(), 4)

                        # Set inverse weight to zero
                        EE_weights[j, i, 0] = 0

                    else:
                        EE_weights[i, j, 0] = 0

        return W_ee

    def gen_EI(self, N_excit_neurons, N_inhib_neurons, excit_to_inhib, excit_inhib_std):
        # Create weight array for EI
        W_ei = np.zeros((N_excit_neurons, N_inhib_neurons))

        # Determine the receptor field size
        recept_fields = np.random.normal(
            loc=excit_to_inhib, scale=excit_inhib_std, size=N_inhib_neurons
        )

        # Check if any of the recept fields are outside of the limits
        if np.any(recept_fields < 0 or recept_fields > int(excit_to_inhib * 2)):

            # Get index of the values outside the range
            idx = np.where(
                recept_fields < 1 or recept_fields > int(excit_to_inhib * 2)
            )[0]

            # Assign these indexes with new values inside the range
            recept_fields[idx] = np.clip(recept_fields[idx], 1, int(excit_to_inhib * 2))

        # Fill the weight array W_ei with the determined receptor field sizes
        for base in range(N_inhib_neurons):

            # Get first weight
            dist = recept_fields[base] // 2

            # Check if recept_fields is an even number
            if recept_fields[base] % 2 == 0:
                correction = 1
            else:
                correction = 0

            # Check if dist is outside of weight array
            if base - dist - correction < 0:

                # Define interval of excit neurons to receive weights
                start_row = 0
                end_row = recept_fields[base] - 1

                # Set weights for interval
                W_ei[start_row:end_row, base] = np.random.random(size=recept_fields)

            # Check if the total distance exceeds the weight array
            elif base + dist > N_excit_neurons or base + dist + 1 > N_excit_neurons:

                # Define interval of excit neurons to receive weights
                start_row = N_excit_neurons - recept_fields[base]
                end_row = N_excit_neurons

                # Set weights for interval
                W_ei[start_row:end_row, base] = np.random.random(size=recept_fields)

            # If the receptor field does not exceed the array dims, do this
            else:

                # Define interval of excit neurons to receive weights
                start_row = base - dist - 1
                end_row = base + dist

                # Set weights for interval
                W_ei[start_row:end_row, base] = np.random.random(size=recept_fields)

        return W_ei

    def gen_IE_II_weights(self, N_excit_neurons, N_inhib_neurons):
        # Inhibitory to excitatory synapses
        IE_weights = np.random.rand(N_inhib_neurons, N_excit_neurons)

        # Inhibitory to inhibitory synapses
        II_weights = np.random.rand(N_inhib_neurons, N_inhib_neurons)

        return W_ie, W_ii


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
