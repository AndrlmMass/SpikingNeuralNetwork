# Import libraries
import numpy as np
from skimage.draw import line


# Generate triangle-shaped input space
def gen_triangle(input_dims, triangle_size, triangle_thickness):
    if input_dims % 2 == 0:
        raise ValueError("Invalid input size. Should be an odd number")
    if not (0 <= triangle_size <= 1):
        raise ValueError("Triangle size must be between 0 and 1.")
    if triangle_thickness <= 0:
        raise ValueError("Triangle thickness must be greater than 0.")

    # Initialize the input space with zeros
    input_space = np.zeros((input_dims, input_dims))

    # Calculate the center of the input space
    center = input_dims / 2

    # Calculate base length and height of the triangle
    base_length = triangle_size * input_dims
    triangle_height = (np.sqrt(3) / 2) * base_length

    # Define the vertices of the triangle, centered within the input space
    top_vertex = (center, center + triangle_height / 2)
    bottom_left_vertex = (center - base_length / 2, center - triangle_height / 2)
    bottom_right_vertex = (center + base_length / 2, center - triangle_height / 2)

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

    # Draw the triangle's edges on the high-resolution grid using new vertices
    vertices = np.array([top_vertex, bottom_left_vertex, bottom_right_vertex])
    draw_line_high_res(vertices[0], vertices[1])
    draw_line_high_res(vertices[1], vertices[2])
    draw_line_high_res(vertices[2], vertices[0])

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

    # Normalize and flip the input space vertically
    max_value = np.max(input_space)
    input_space_normalized = input_space / max_value if max_value > 0 else input_space
    input_space_flipped = np.flipud(input_space_normalized)

    return input_space_flipped
