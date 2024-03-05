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


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from skimage.draw import polygon


def gen_triangle(input_dims, triangle_size, receptor_size, triangle_thickness):
    if input_dims % 2 == 0:
        raise UserWarning("Invalid input dimensions. Must be an odd value.")
    if not (0 <= triangle_size <= 1):
        raise ValueError("Triangle size must be between 0 and 1.")
    if triangle_thickness <= 0:
        raise ValueError("Triangle thickness must be greater than 0.")

    # Adjust plotting range to accommodate the outermost receptive fields
    plot_dims = input_dims + 1

    # Create a plot to display the triangle and receptive fields
    fig, ax = plt.subplots()
    plt.xlim(0, plot_dims)
    plt.ylim(0, plot_dims)

    # Plot squares representing receptive fields for each neuron
    for x in range(1, plot_dims):
        for y in range(1, plot_dims):
            bottom_left_x = x - receptor_size / 2
            bottom_left_y = y - receptor_size / 2
            square = patches.Rectangle(
                (bottom_left_x, bottom_left_y),
                receptor_size,
                receptor_size,
                linewidth=1,
                edgecolor="b",
                facecolor="none",
            )
            ax.add_patch(square)

    # Adjust triangle vertices for correct centering and sizing
    center = plot_dims / 2
    half_base_length = (
        input_dims * triangle_size / 2
    )  # Ensure triangle size scales to input_dims

    # Define triangle vertices with corrected centering and size
    top_point = (center, center + half_base_length)
    low_left = (center - half_base_length, center - half_base_length)
    low_right = (center + half_base_length, center - half_base_length)

    # Draw the triangle with specified thickness
    triangle = plt.Polygon(
        [low_left, top_point, low_right],
        closed=True,
        fill=None,
        edgecolor="r",
        linewidth=triangle_thickness,
    )
    ax.add_patch(triangle)

    plt.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray")
    plt.xticks(np.arange(1, plot_dims))
    plt.yticks(np.arange(1, plot_dims))
    plt.show()

    # Calculate the center and the size of the triangle
    center = input_dims // 2
    half_base = input_dims * triangle_size / 2
    height = half_base * np.sqrt(3)  # For an equilateral triangle

    # Vertices of the triangle
    vertices = np.array(
        [
            [center, center + height / 2],  # Top vertex
            [center - half_base, center - height / 2],  # Bottom left vertex
            [center + half_base, center - height / 2],  # Bottom right vertex
        ]
    )

    # Initialize the input space with zeros
    input_space = np.zeros((input_dims, input_dims))

    # High-resolution grid for more accurate overlap estimation
    subgrid_resolution = 100  # Number of pixels per square side
    subgrid_size = subgrid_resolution**2  # Total number of pixels in the square

    # Rasterize the triangle into a high-resolution binary mask
    rr, cc = polygon(
        vertices[:, 1] * subgrid_resolution, vertices[:, 0] * subgrid_resolution
    )
    high_res_triangle = np.zeros(
        (input_dims * subgrid_resolution, input_dims * subgrid_resolution)
    )
    high_res_triangle[rr, cc] = 1  # Fill the triangle area

    # Estimate the overlap for each square in the grid
    for i in range(input_dims):
        for j in range(input_dims):
            # Calculate the bounding box for the current square in high-resolution space
            top = i * subgrid_resolution
            left = j * subgrid_resolution
            bottom = (i + 1) * subgrid_resolution
            right = (j + 1) * subgrid_resolution

            # Extract the subgrid for the current square
            subgrid = high_res_triangle[top:bottom, left:right]

            # The coverage is the sum of the subgrid values (number of pixels inside the triangle)
            # divided by the total number of pixels in the subgrid
            coverage = np.sum(subgrid) / subgrid_size
            input_space[i, j] = coverage

    return input_space


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
