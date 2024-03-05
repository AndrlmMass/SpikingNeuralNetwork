import numpy as np


import numpy as np


def gen_StimE(radius, N_input_neurons, N_excit_neurons):
    input_shape = int(np.sqrt(N_input_neurons))
    W_se = np.zeros((N_input_neurons, N_excit_neurons))
    excitatory_positions = [
        (mu % input_shape, mu // input_shape) for mu in range(N_excit_neurons)
    ]

    for mu, (ex_col, ex_row) in enumerate(excitatory_positions):
        for row in range(
            max(0, ex_row - radius), min(input_shape, ex_row + radius + 1)
        ):
            distance_from_center_row = abs(row - ex_row)
            max_column_distance = int(np.sqrt(radius**2 - distance_from_center_row**2))

            start_col = max(0, ex_col - max_column_distance)
            end_col = min(input_shape, ex_col + max_column_distance + 1)

            for col in range(start_col, end_col):
                p = row * input_shape + col
                W_se[p, mu] = np.random.random()

    return W_se


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


W_se = gen_StimE(radius=3, N_input_neurons=36, N_excit_neurons=36)

draw_receptive_field_single(W_se, N_input_neurons=36)
