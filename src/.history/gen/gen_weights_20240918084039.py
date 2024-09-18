import numpy as np
import random


class gen_weights:
    def gen_SE(self, N_input_neurons, N_excit_neurons, time, w_prob, w_val):

        # Initialize the weight matrix with zeros (for all input to excitatory neuron connections)
        mask = np.random.random(size=(N_input_neurons, N_excit_neurons)) < w_prob

        # Define ideal weights set to base_num
        W_se = np.zeros(shape=(N_input_neurons, N_excit_neurons))
        W_se[mask] = w_val

        # Define the ideal weights for stimulation-excitation
        W_se_ideal = np.zeros(size=(N_input_neurons, N_excit_neurons))

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

    def gen_EE(self, N_excit_neurons, time, w_prob, w_val):
        # Initialize the weight matrix with zeros (for all input to excitatory neuron connections)
        mask = np.random.random(size=(N_excit_neurons, N_excit_neurons)) < w_prob

        # Define ideal weights set to base_num
        W_ee = np.zeros(shape=(N_excit_neurons, N_excit_neurons))
        W_ee[mask] = w_val

        # Define the ideal weights for stimulation-excitation
        W_ee_ideal = np.zeros(size=(N_excit_neurons, N_excit_neurons))

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

        return W_ie, W_ie_2d, selected_indices
