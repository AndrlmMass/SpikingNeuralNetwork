# Gen data according to y number of classes
import numpy as np
import matplotlib.pyplot as plt


def gen_data(N_classes, N_input_neurons):
    # Define input shape
    input_dims = int(np.sqrt(N_input_neurons))

    # Assert input space based on input_dims variable
    input_shape = np.zeros((input_dims, input_dims))

    # Define rules for input classes
    rules = []

    # Loop over num_classes to generate rules
    for clas in N_classes:

        # Assert rules
        d = 1


def gen_triangle(input_dims, receptor_size, triangle_thickness):
    if input_dims % 2 == 0:
        raise UserWarning("Invalid input dimensions. Must be an odd value.")
    if not (0 <= triangle_size <= 1):
        raise ValueError("Triangle size must be between 0 and 1.")
    if triangle_thickness <= 0:
        raise ValueError("Triangle thickness must be greater than 0.")

    # Define input space
    input_space = np.zeros((input_dims, input_dims))

    # Define edges of triangle
    basic_unit = input_dims // 2 + 1
    radius = input_dims // 4
    centre = (basic_unit, basic_unit)
    top = (basic_unit - radius, basic_unit)
    bottom_left = (basic_unit + radius, basic_unit - radius)
    bottom_right = (basic_unit + radius, basic_unit + radius)

    # Return rules and starting positions


def plot_input_space(input_space):
    # Create a heatmap to visualize the input_space coverage values
    plt.figure(figsize=(8, 8))  # Set figure size
    plt.imshow(input_space, cmap="viridis", origin="lower", interpolation="nearest")
    plt.colorbar(label="Degree of Coverage")
    plt.title("Input Space Coverage Visualization")
    plt.xlabel("X Dimension")
    plt.ylabel("Y Dimension")
    # Configure ticks to align with each square if needed
    tick_marks = np.arange(len(input_space))
    plt.xticks(tick_marks, [str(i) for i in tick_marks])
    plt.yticks(tick_marks, [str(i) for i in tick_marks])
    plt.grid(False)  # Optionally disable the grid for clarity
    plt.show()


# Example usage: Adjust 'input_dims', 'triangle_size', 'receptor_size', and 'triangle_thickness' as needed
input_space = gen_triangle(
    15, 0.6, 1, 20
)  # For example: input_dims=3, triangle_size=1, receptor_size=1, triangle_thickness=2
plot_input_space(input_space)
