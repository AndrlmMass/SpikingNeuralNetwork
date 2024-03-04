import numpy as np


def gen_StimE(radius, N_input_neurons, N_excit_neurons):
    # Ensure the input neurons form a square grid
    input_shape = int(np.sqrt(N_input_neurons))

    # Initialize the weights matrix with zeros
    W_se = np.zeros((N_input_neurons, N_excit_neurons))

    # Calculate the 2D positions for each excitatory neuron assuming a linear indexing
    excitatory_positions = [
        (mu % input_shape, mu // input_shape) for mu in range(N_excit_neurons)
    ]

    # Iterate over each excitatory neuron to define and assign its receptive field
    for mu, (ex_col, ex_row) in enumerate(excitatory_positions):
        # Define the bounds of the receptive field in the 2D space
        for row in range(
            max(0, ex_row - radius), min(input_shape, ex_row + radius + 1)
        ):
            for col in range(
                max(0, ex_col - radius), min(input_shape, ex_col + radius + 1)
            ):
                # Calculate the 1D index for the current position in the input space
                pos = row * input_shape + col
                # Assign a random weight to the position within the receptive field
                W_se[pos, mu] = np.random.random()

    return W_se


import numpy as np
import matplotlib.pyplot as plt


def draw_receptive_field_single(weights, N_input_neurons):
    input_shape = int(
        np.sqrt(N_input_neurons)
    )  # Calculate the size of one dimension of the square input grid

    for ex in range(weights.shape[1]):
        fig, ax = plt.subplots(
            figsize=(5, 5)
        )  # Create a plot for the current excitatory neuron
        ax.set_xlim(-0.5, input_shape - 0.5)
        ax.set_ylim(-0.5, input_shape - 0.5)
        ax.set_xticks(np.arange(0, input_shape, 1))
        ax.set_yticks(np.arange(0, input_shape, 1))
        ax.set_title(f"Excitatory Neuron {ex + 1}")
        plt.grid(True)

        # Draw each input neuron as a grey dot
        for i in range(input_shape):
            for j in range(input_shape):
                ax.plot(i, j, "o", color="lightgrey", markersize=10)

        # Find positions where the excitatory neuron has an input synapse (non-zero weights)
        active_synapses = np.where(weights[:, ex] > 0)[0]
        for pos in active_synapses:
            y, x = divmod(pos, input_shape)
            # Draw red boxes around active synapses
            rect = plt.Rectangle(
                (x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2
            )
            ax.add_patch(rect)

        plt.show()
        input("Press Enter to continue to the next neuron...")


draw_receptive_field_single(W_se, N_input_neurons)
