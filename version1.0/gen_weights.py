import numpy as np


class gen_weights:
    def gen_SE(radius, N_input_neurons, N_excit_neurons):
        # Calculate the side length of the square grid of input neurons
        input_shape = int(np.sqrt(N_input_neurons))

        # Initialize the weight matrix with zeros (for all input to excitatory neuron connections)
        W_se = np.zeros((N_input_neurons, N_excit_neurons))

        # Compute the 2D grid positions for each excitatory neuron based on their index
        excitatory_positions = [
            (mu % input_shape, mu // input_shape) for mu in range(N_excit_neurons)
        ]

        # Iterate over each excitatory neuron to define and assign its receptive field
        for mu, (ex_col, ex_row) in enumerate(excitatory_positions):

            # Iterate through rows within the receptive field radius
            for row in range(
                max(0, ex_row - radius), min(input_shape, ex_row + radius + 1)
            ):
                # Calculate the vertical distance from the current row to the excitatory neuron's row
                distance_from_center_row = abs(row - ex_row)

                # Calculate the maximum horizontal distance within the circular receptive field
                # for the current row, using the Pythagorean theorem
                max_column_distance = int(
                    np.sqrt(radius**2 - distance_from_center_row**2)
                )

                # Determine the start and end columns for the receptive field in the current row
                start_col = max(0, ex_col - max_column_distance)
                end_col = min(input_shape, ex_col + max_column_distance + 1)

                # Assign random weights to the connections within the receptive field bounds
                for col in range(start_col, end_col):
                    # Calculate the linear index for the input neuron at the current row and column
                    p = row * input_shape + col

                    # Assign a random weight to the connection
                    W_se[p, mu] = np.random.random()

        return W_se

    def gen_EE(N_excit_neurons, prob, time):
        # Initialize the arrays for weights
        W_ee = np.zeros((N_excit_neurons, N_excit_neurons, time))

        # Initial weights at time 0 set to random for prob portion
        W_ee[:, :, 0] = np.random.rand(N_excit_neurons, N_excit_neurons)
        W_ee[:, :, 0] *= np.random.rand(N_excit_neurons, N_excit_neurons) < prob

        # Ensure no self-connections at the initial time
        np.fill_diagonal(W_ee[:, :, 0], 0)

        return W_ee

    def gen_EI(N_excit_neurons, N_inhib_neurons, time):
        # Calculate probability of connection
        prob = N_excit_neurons / (N_excit_neurons + N_inhib_neurons)

        # Create weight array for EI
        W_ei = np.zeros((N_excit_neurons, N_inhib_neurons, time))

        # Assign random weights to N inhibitory neurons
        W_ei[:, :, 0] = np.random.rand(N_excit_neurons, N_inhib_neurons)
        W_ei[:, :, 0] *= np.random.rand(N_excit_neurons, N_inhib_neurons) < prob

        return W_ei

    def gen_IE(N_inhib_neurons, N_excit_neurons, W_ei, prob, time):
        # Initialize the weight array for IE connections
        W_ie = np.zeros((N_inhib_neurons, N_excit_neurons, time))

        # Set initial weights by transposing the EI weights for the initial time step
        W_ie[:, :, 0] = W_ei[:, :, 0].T

        # Identify positions in the transposed matrix (IE) that correspond to non-zero weights in EI
        non_zero_positions = W_ie[:, :, 0] != 0

        # Generate a mask for potential connections based on the specified probability
        W_idx = np.random.rand(N_inhib_neurons, N_excit_neurons) < prob

        # Ensure that new connections (where W_idx is true) do not override existing non-zero connections from EI
        # This step respects the established connections by not allowing feedback loops
        valid_new_connections = np.logical_and(W_idx, ~non_zero_positions)

        # Assign random weights only to valid new connection positions
        initial_random_weights = np.random.rand(N_inhib_neurons, N_excit_neurons)
        W_ie[:, :, 0] = np.where(
            valid_new_connections, initial_random_weights, W_ie[:, :, 0]
        )

        return W_ie


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
