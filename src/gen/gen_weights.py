import numpy as np


class gen_weights:
    def gen_SE(self, N_input_neurons, N_excit_neurons, w_prob, w_val):

        # Initialize the weight matrix with zeros (for all input to excitatory neuron connections)
        mask = np.random.random(size=(N_input_neurons, N_excit_neurons)) < w_prob

        # Define ideal weights set to base_num
        W_se = np.zeros(shape=(N_input_neurons, N_excit_neurons))
        W_se[mask] = w_val

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
