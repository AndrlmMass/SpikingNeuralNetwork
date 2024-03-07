import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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

    from skimage.draw import line

    # High-resolution grid for more accurate edge drawing
    subgrid_resolution = 100  # Number of pixels per square side
    subgrid_size = subgrid_resolution**2

    # Initialize the high-resolution binary mask for the hollow triangle
    high_res_triangle = np.zeros(
        (input_dims * subgrid_resolution, input_dims * subgrid_resolution)
    )

    # Function to draw a line on the high-resolution grid
    def draw_line_high_res(v0, v1):
        rr, cc = line(
            int(v0[1] * subgrid_resolution),
            int(v0[0] * subgrid_resolution),
            int(v1[1] * subgrid_resolution),
            int(v1[0] * subgrid_resolution),
        )
        high_res_triangle[rr, cc] = 1

    # Draw the triangle's edges on the high-resolution grid
    draw_line_high_res(vertices[0], vertices[1])  # Edge from top vertex to bottom left
    draw_line_high_res(
        vertices[1], vertices[2]
    )  # Edge from bottom left to bottom right
    draw_line_high_res(vertices[2], vertices[0])  # Edge from bottom right to top vertex

    # Estimate the overlap for each square in the grid, focusing on edges
    for i in range(input_dims):
        for j in range(input_dims):
            top = i * subgrid_resolution
            left = j * subgrid_resolution
            bottom = (i + 1) * subgrid_resolution
            right = (j + 1) * subgrid_resolution

            # Extract the subgrid for the current square
            subgrid = high_res_triangle[top:bottom, left:right]

            # Calculate coverage based on edge presence
            coverage = np.sum(subgrid) / subgrid_size
            input_space[i, j] = coverage if np.sum(subgrid) > 0 else 0

    # Step 1: Find the maximum value in input_space
    max_value = np.max(input_space)

    # Step 2: Normalize values in input_space
    input_space_normalized = input_space / max_value if max_value > 0 else input_space

    return input_space_normalized


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from skimage.draw import line

import numpy as np
from skimage.draw import line


def gen_square(input_dims, square_size, receptor_size, square_thickness):
    if input_dims % 2 == 0:
        raise UserWarning("Invalid input dimensions. Must be an odd value.")
    if not (0 <= square_size <= 1):
        raise ValueError("Square size must be between 0 and 1.")
    if square_thickness <= 0:
        raise ValueError("Square thickness must be greater than 0.")

    # Initialize the input space with zeros
    input_space = np.zeros((input_dims, input_dims))

    # High-resolution grid for more accurate edge drawing
    subgrid_resolution = 100  # Number of pixels per square side
    subgrid_size = subgrid_resolution**2

    # Calculate center of the input space
    center = input_dims // 2

    # Calculate the size of the square in terms of input dimensions
    square_half_length = int(square_size * input_dims / 2)

    # Initialize the high-resolution binary mask for the hollow square
    high_res_square = np.zeros(
        (input_dims * subgrid_resolution, input_dims * subgrid_resolution)
    )

    # Define square vertices centered within the high-resolution grid
    top_left = (center - square_half_length, center + square_half_length)
    top_right = (center + square_half_length, center + square_half_length)
    bottom_left = (center - square_half_length, center - square_half_length)
    bottom_right = (center + square_half_length, center - square_half_length)
    vertices = [top_left, top_right, bottom_right, bottom_left]

    # Function to draw lines on the high-resolution grid for square edges
    def draw_line_high_res(v0, v1):
        rr, cc = line(
            int(v0[1] * subgrid_resolution),
            int(v0[0] * subgrid_resolution),
            int(v1[1] * subgrid_resolution),
            int(v1[0] * subgrid_resolution),
        )
        high_res_square[rr, cc] = 1

    # Draw the square's edges on the high-resolution grid
    for i in range(len(vertices)):
        draw_line_high_res(
            vertices[i], vertices[(i + 1) % len(vertices)]
        )  # Connect vertices in order

    # Estimate the overlap for each square in the grid, focusing on edges
    for i in range(input_dims):
        for j in range(input_dims):
            top = i * subgrid_resolution
            left = j * subgrid_resolution
            bottom = (i + 1) * subgrid_resolution
            right = (j + 1) * subgrid_resolution

            # Extract the subgrid for the current square
            subgrid = high_res_square[top:bottom, left:right]

            # Calculate coverage based on edge presence
            coverage = np.sum(subgrid) / subgrid_size
            input_space[i, j] = coverage if np.sum(subgrid) > 0 else 0

    # Normalize values in input_space
    max_value = np.max(input_space)
    input_space_normalized = input_space / max_value if max_value > 0 else input_space

    return input_space_normalized


def gen_x_symbol(input_dims, x_size, receptor_size, x_thickness):
    if input_dims % 2 == 0:
        raise UserWarning("Invalid input dimensions. Must be an odd value.")
    if not (0 <= x_size <= 1):
        raise ValueError("X size must be between 0 and 1.")
    if x_thickness <= 0:
        raise ValueError("X thickness must be greater than 0.")

    # Adjust plotting range to accommodate the outermost receptive fields
    plot_dims = input_dims + 1

    # Create a plot to display the X symbol and receptive fields
    fig, ax = plt.subplots()
    plt.xlim(0, plot_dims)
    plt.ylim(0, plot_dims)

    # Plot squares representing receptive fields for each neuron
    for x in range(1, plot_dims):
        for y in range(1, plot_dims):
            bottom_left_x = x - receptor_size / 2
            bottom_left_y = y - receptor_size / 2
            receptive_field_square = patches.Rectangle(
                (bottom_left_x, bottom_left_y),
                receptor_size,
                receptor_size,
                linewidth=1,
                edgecolor="b",
                facecolor="none",
            )
            ax.add_patch(receptive_field_square)

    # Adjust X symbol dimensions for correct centering and sizing
    center = plot_dims / 2
    half_diagonal = input_dims * x_size / 2

    # Define end points for the two lines that form the X
    line1_start = (center - half_diagonal, center + half_diagonal)
    line1_end = (center + half_diagonal, center - half_diagonal)
    line2_start = (center + half_diagonal, center + half_diagonal)
    line2_end = (center - half_diagonal, center - half_diagonal)

    # Draw the X symbol with specified thickness
    ax.plot(
        [line1_start[0], line1_end[0]],
        [line1_start[1], line1_end[1]],
        color="r",
        linewidth=x_thickness,
    )
    ax.plot(
        [line2_start[0], line2_end[0]],
        [line2_start[1], line2_end[1]],
        color="r",
        linewidth=x_thickness,
    )

    plt.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray")
    plt.xticks(np.arange(1, plot_dims))
    plt.yticks(np.arange(1, plot_dims))
    plt.show()

    # Initialize the input space with zeros
    input_space = np.zeros((input_dims, input_dims))

    # High-resolution grid for more accurate edge drawing
    subgrid_resolution = 100  # Number of pixels per square side
    subgrid_size = subgrid_resolution**2

    # Initialize the high-resolution binary mask for the hollow X
    high_res_x = np.zeros(
        (input_dims * subgrid_resolution, input_dims * subgrid_resolution)
    )

    # Function to draw lines on the high-resolution grid for X edges
    def draw_line_high_res(v0, v1):
        rr, cc = line(
            int(v0[1] * subgrid_resolution),
            int(v0[0] * subgrid_resolution),
            int(v1[1] * subgrid_resolution),
            int(v1[0] * subgrid_resolution),
        )
        high_res_x[rr, cc] = 1

    # Draw the X's edges on the high-resolution grid
    draw_line_high_res(line1_start, line1_end)
    draw_line_high_res(line2_start, line2_end)

    # Estimate the overlap for each square in the grid, focusing on edges
    for i in range(input_dims):
        for j in range(input_dims):
            top = i * subgrid_resolution
            left = j * subgrid_resolution
            bottom = (i + 1) * subgrid_resolution
            right = (j + 1) * subgrid_resolution

            # Extract the subgrid for the current square
            subgrid = high_res_x[top:bottom, left:right]

            # Calculate coverage based on edge presence
            coverage = np.sum(subgrid) / subgrid_size
            input_space[i, j] = coverage if np.sum(subgrid) > 0 else 0

    # Normalize values in input_space
    max_value = np.max(input_space)
    input_space_normalized = input_space / max_value if max_value > 0 else input_space

    return input_space_normalized


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from skimage.draw import circle_perimeter


def gen_circle(input_dims, circle_size, receptor_size, circle_thickness):
    if input_dims % 2 == 0:
        raise UserWarning("Invalid input dimensions. Must be an odd value.")
    if not (0 <= circle_size <= 1):
        raise ValueError("Circle size must be between 0 and 1.")
    if circle_thickness <= 0:
        raise ValueError("Circle thickness must be greater than 0.")

    # Adjust plotting range to accommodate the outermost receptive fields
    plot_dims = input_dims + 1

    # Create a plot to display the circle and receptive fields
    fig, ax = plt.subplots()
    plt.xlim(0, plot_dims)
    plt.ylim(0, plot_dims)

    # Plot squares representing receptive fields for each neuron
    for x in range(1, plot_dims):
        for y in range(1, plot_dims):
            bottom_left_x = x - receptor_size / 2
            bottom_left_y = y - receptor_size / 2
            receptive_field_square = patches.Rectangle(
                (bottom_left_x, bottom_left_y),
                receptor_size,
                receptor_size,
                linewidth=1,
                edgecolor="b",
                facecolor="none",
            )
            ax.add_patch(receptive_field_square)

    # Define circle center and radius for correct centering and sizing
    center = plot_dims / 2
    radius = input_dims * circle_size / 2

    # Draw the circle with specified thickness
    circle = patches.Circle(
        (center, center),
        radius,
        fill=None,
        edgecolor="r",
        linewidth=circle_thickness,
    )
    ax.add_patch(circle)

    plt.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray")
    plt.xticks(np.arange(1, plot_dims))
    plt.yticks(np.arange(1, plot_dims))
    plt.show()

    # Initialize the input space with zeros
    input_space = np.zeros((input_dims, input_dims))

    # High-resolution grid for more accurate edge drawing
    subgrid_resolution = 100  # Number of pixels per square side
    subgrid_size = subgrid_resolution**2

    # Initialize the high-resolution binary mask for the hollow circle
    high_res_circle = np.zeros(
        (input_dims * subgrid_resolution, input_dims * subgrid_resolution)
    )

    # Draw the circle's edge on the high-resolution grid
    rr, cc = circle_perimeter(
        int(center * subgrid_resolution),
        int(center * subgrid_resolution),
        int(radius * subgrid_resolution),
    )
    high_res_circle[rr, cc] = 1

    # Estimate the overlap for each square in the grid, focusing on edges
    for i in range(input_dims):
        for j in range(input_dims):
            top = i * subgrid_resolution
            left = j * subgrid_resolution
            bottom = (i + 1) * subgrid_resolution
            right = (j + 1) * subgrid_resolution

            # Extract the subgrid for the current square
            subgrid = high_res_circle[top:bottom, left:right]

            # Calculate coverage based on edge presence
            coverage = np.sum(subgrid) / subgrid_size
            input_space[i, j] = coverage if np.sum(subgrid) > 0 else 0

    # Normalize values in input_space
    max_value = np.max(input_space)
    input_space_normalized = input_space / max_value if max_value > 0 else input_space

    return input_space_normalized


def plot_input_space(input_space, input_dims):
    # Create a heatmap to visualize the input_space coverage values
    plt.figure(figsize=(8, 8))  # Set figure size
    plt.imshow(input_space, cmap="viridis", origin="lower", interpolation="nearest")
    plt.colorbar(label="Degree of Coverage")
    plt.title("Input Space Coverage Visualization")
    plt.xlabel("X Dimension")
    plt.ylabel("Y Dimension")
    # Configure ticks to align with each square if needed
    tick_marks = np.arange(len(input_space - 1))
    plt.xticks(tick_marks, [str(i) for i in tick_marks])
    plt.yticks(tick_marks, [str(i) for i in tick_marks])
    plt.xlim(0, input_dims)  # Assuming input_dims is the max value you want to show
    plt.ylim(0, input_dims)
    plt.grid(False)  # Optionally disable the grid for clarity
    plt.show()


input_space = gen_triangle(
    7, 0.6, 1, 20
)  # input_dims, triangle_size, receptor_size, triangle_thickness
plot_input_space(input_space, 6)
print(np.round(input_space, 2))

input_space_2 = gen_square(11, 0.6, 1, 20)
plot_input_space(input_space_2, 11)
print(np.round(input_space_2, 2))

input_space_3 = gen_x_symbol(11, 0.5, 1, 20)
plot_input_space(input_space_3, 11)
print(input_space_3)

input_space_4 = gen_circle(11, 0.5, 1, 20)
plot_input_space(input_space_4, 11)
print(input_space_4)
