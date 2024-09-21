import numpy as np


class gen_weights:
    def gen_SE(self, N_input_neurons, N_excit_neurons, w_prob, w_val, radius):
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

                    # Check if a weight will be applied
                    """
                    This assumes that Zenke meant a 0.5% chance of a synapse formation between
                    a given stimulation cell and an excitation cell. This leads to very sparse
                    connections, so it might be wrong. Not sure.  
                    """
                    if np.random.random() < w_prob:

                        # Assign weight to the connection
                        W_se[p, mu] = w_val

        # Define the ideal weights for stimulation-excitation
        W_se_ideal = np.zeros(shape=(N_input_neurons, N_excit_neurons))

        return W_se, W_se_ideal

    def gen_EE(self, N_excit_neurons, w_prob, w_val):
        # Initialize the weight matrix with zeros (for all input to excitatory neuron connections)
        mask = np.random.random(size=(N_excit_neurons, N_excit_neurons)) < w_prob

        # Define ideal weights set to base_num
        W_ee = np.zeros(shape=(N_excit_neurons, N_excit_neurons))
        W_ee[mask] = w_val

        # Define the ideal weights for stimulation-excitation
        W_ee_ideal = np.zeros(shape=(N_excit_neurons, N_excit_neurons))

        return W_ee, W_ee_ideal

    def gen_EI(self, N_excit_neurons, N_inhib_neurons, w_prob, w_val):
        # Initialize the weight matrix with zeros (for all input to excitatory neuron connections)
        mask = np.random.random(size=(N_excit_neurons, N_inhib_neurons)) < w_prob

        # Define ideal weights set to base_num
        W_ei = np.zeros(shape=(N_excit_neurons, N_inhib_neurons))
        W_ei[mask] = w_val

        return W_ei

    def gen_II(self, N_inhib_neurons, w_prob, w_val):
        # Initialize the weight matrix with zeros (for all input to excitatory neuron connections)
        mask = np.random.random(size=(N_inhib_neurons, N_inhib_neurons)) < w_prob

        # Define ideal weights set to base_num
        W_ii = np.zeros(shape=(N_inhib_neurons, N_inhib_neurons))
        W_ii[mask] = w_val

        return W_ii

    def gen_IE(self, N_inhib_neurons, N_excit_neurons, time, w_prob, w_val):
        # Create mask for weight initialization
        mask = np.random.random(size=(N_inhib_neurons, N_excit_neurons)) < w_prob

        # Define ideal weights set to base_num
        W_ie = np.zeros(shape=(N_inhib_neurons, N_excit_neurons))
        W_ie[mask] = w_val

        # Create a representative 2D array of the weights for visualization
        W_ie_plt = np.zeros((time - 1, 3))  # 3: 1=mean, 2=high, 3=low]

        return W_ie, W_ie_plt
