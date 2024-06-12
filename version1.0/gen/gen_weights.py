import numpy as np
import random


class gen_weights:
    def gen_SE(self, radius, N_input_neurons, N_excit_neurons, time, basenum):
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

        # Define ideal weights set to base_num
        W_se_ideal = np.full((N_input_neurons, N_excit_neurons), basenum)

        # Create a representative 2D array of the weights for visualization
        W_se_2d = np.zeros((time, 10))

        # Find non-zero indices in 'W_se'
        non_zero_indices = np.transpose(np.nonzero(W_se))

        # Select 10 random non-zero indices
        selected_indices = non_zero_indices[
            np.random.choice(len(non_zero_indices), 10, replace=False)
        ]

        # Extract the values corresponding to the selected indices
        W_se_2d[0] = W_se[selected_indices[:, 0], selected_indices[:, 1]]

        return W_se, W_se_ideal, W_se_2d, selected_indices

    def gen_EE(self, N_excit_neurons, prob, time, basenum):
        # Initialize the arrays for weights
        W_ee = np.zeros((N_excit_neurons, N_excit_neurons))

        # Initial weights at time 0 set to random for prob portion
        W_ee = np.random.rand(N_excit_neurons, N_excit_neurons)
        W_ee *= np.random.rand(N_excit_neurons, N_excit_neurons) < prob

        # Ensure no self-connections at the initial time
        np.fill_diagonal(W_ee, 0)

        # Define ideal weights set to base_num
        W_ee_ideal = np.full((N_excit_neurons, N_excit_neurons), basenum)

        # Create a representative 2D array of the weights for visualization
        W_ee_2d = np.zeros((time, 10))

        # Find non-zero indices in 'W_se'
        non_zero_indices = np.transpose(np.nonzero(W_ee))

        # Select 10 random non-zero indices
        selected_indices = non_zero_indices[
            np.random.choice(len(non_zero_indices), 10, replace=False)
        ]

        # Extract the values corresponding to the selected indices
        W_ee_2d[0] = W_ee[selected_indices[:, 0], selected_indices[:, 1]]

        return W_ee, W_ee_ideal, W_ee_2d, selected_indices

    def gen_EI(self, N_excit_neurons, N_inhib_neurons, time, prob):
        # Create weight array for EI
        W_ei = np.zeros((N_excit_neurons, N_inhib_neurons))

        # Assign random weights to N inhibitory neurons
        W_ei = np.random.rand(N_excit_neurons, N_inhib_neurons)
        W_ei *= np.random.rand(N_excit_neurons, N_inhib_neurons) < prob

        return W_ei

    def gen_IE(
        self, N_inhib_neurons, N_excit_neurons, W_ei, radius, time, N_ws, weight_val
    ):
        # Initialize the weight array for IE connections
        W_ie = np.zeros((N_inhib_neurons, N_excit_neurons))

        # Use the transpose of W_ei to indicate where weights cannot be placed
        not_allowed_indices = np.logical_not(W_ei.T).astype(int)

        # Loop through each receiving neuron
        for n in range(W_ie.shape[0]):
            # Define neighborhood of connections based on mean presynaptic connections
            indices = np.where(W_ei[:, n] == 0)[0]
            if len(indices) == 0:
                continue  # Skip if no zero indices found

            mid_point = np.random.randint(low=0, high=len(indices))
            if mid_point + radius > len(indices):
                diff_min = mid_point + radius - len(indices)
            else:
                diff_min = 0

            if mid_point - radius < 0:
                diff_max = abs(mid_point - radius)
            else:
                diff_max = 0

            range_ = (
                int(max(mid_point - radius - diff_min, 0)),
                int(min(mid_point + radius + diff_max, len(indices))),
            )

            # Find zero-valued positions in the allowed range
            nu_connects = list(indices[range_[0] : range_[1]])

            # Draw N_ws samples from index-list
            if len(nu_connects) >= N_ws:
                nz_indices = random.sample(nu_connects, N_ws)
            else:
                nz_indices = nu_connects

            # Create synapses with random values between 0 and 1
            for idx in nz_indices:
                if not_allowed_indices[n, idx] == 1:
                    W_ie[n, idx] = np.random.random()

        # Create array of ideal weights
        W_ie_ideal = np.full((N_inhib_neurons, N_excit_neurons), weight_val)

        # Create a representative 2D array of the weights for visualization
        W_ie_2d = np.zeros((time, 10))

        # Find non-zero indices in 'W_ie'
        non_zero_indices = np.transpose(np.nonzero(W_ie))

        # Ensure there are enough non-zero indices to select from
        if len(non_zero_indices) < 10:
            raise ValueError("Not enough non-zero indices to select 10 unique ones.")

        # Select 10 random non-zero indices
        selected_indices = non_zero_indices[
            np.random.choice(non_zero_indices.shape[0], 10, replace=False)
        ]

        # Extract the values corresponding to the selected indices
        W_ie_2d[0] = W_ie[selected_indices[:, 0], selected_indices[:, 1]]

        return W_ie, W_ie_ideal, W_ie_2d, selected_indices
