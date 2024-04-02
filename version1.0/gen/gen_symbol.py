import numpy as np
from skimage.draw import line
from skimage.draw import circle_perimeter
import matplotlib.pyplot as plt


def generate_uniform_random(mean, variance):
    # Calculate a and b from mean and variance
    a = mean - np.sqrt(3 * variance)
    b = mean + np.sqrt(3 * variance)

    # Generate a random number from U(a,b)
    random_number = np.random.uniform(a, b)

    return random_number


def gen_triangle(
    input_dims: int,
    triangle_size: float,
    triangle_thickness: int,
    noise_rand: bool,
    noise_variance: float | int,
):

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

    # Adjust the line drawing function to include thickness
    def draw_line_high_res(v0, v1, thickness):
        # Calculate the normal vector to the line
        dx = v1[0] - v0[0]
        dy = v1[1] - v0[1]
        normal = np.array([-dy, dx])
        normal_unit = normal / np.linalg.norm(normal)

        # Draw lines parallel to the original line to achieve thickness
        for offset in np.linspace(
            -thickness / 2, thickness / 2, num=thickness * subgrid_resolution
        ):
            offset_vector = offset * normal_unit
            v0_offset = v0 + offset_vector / subgrid_resolution
            v1_offset = v1 + offset_vector / subgrid_resolution
            rr, cc = line(
                int(v0_offset[1] * subgrid_resolution),
                int(v0_offset[0] * subgrid_resolution),
                int(v1_offset[1] * subgrid_resolution),
                int(v1_offset[0] * subgrid_resolution),
            )
            high_res_triangle[rr, cc] = 1

    # Draw the triangle's edges with thickness on the high-resolution grid using new vertices
    vertices = np.array([top_vertex, bottom_left_vertex, bottom_right_vertex])
    for i in range(3):
        draw_line_high_res(vertices[i], vertices[(i + 1) % 3], triangle_thickness)

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
    max_value = np.mean(input_space)
    input_space_normalized = input_space / max_value if max_value > 0 else input_space
    input_space_flipped = np.flipud(input_space_normalized)

    # Add noise to input_space_flipped based on rand_lvl variable
    if noise_rand:
        for j in range(input_space_flipped.shape[0]):
            for l in range(input_space_flipped.shape[1]):
                mean, variance = input_space_flipped[j, l], noise_variance

                fuzzy_val = generate_uniform_random(mean=mean, variance=variance)

                if fuzzy_val > 1:
                    fuzzy_val = 1
                elif fuzzy_val < 0:
                    fuzzy_val = 0

                input_space_flipped[j, l] = fuzzy_val

    return input_space_flipped


def gen_square(
    input_dims: int,
    square_size: float,
    square_thickness: int,
    noise_rand: bool,
    noise_variance: float | int,
):

    # Define the input_space
    input_space = np.zeros((input_dims, input_dims))

    # Calculate the size of the square in terms of pixels
    square_length = int(square_size * input_dims)

    # Define the thickness of the lines
    thickness = int(square_thickness)  # This example uses a fixed thickness value

    # Calculate the starting and ending indices of the square
    start_idx = (input_dims - square_length) // 2
    end_idx = start_idx + square_length

    # Draw horizontal lines
    for i in range(thickness):
        input_space[start_idx + i, start_idx:end_idx] = 1
        input_space[end_idx - 1 - i, start_idx:end_idx] = 1

    # Draw vertical lines with
    for i in range(thickness):
        input_space[start_idx:end_idx, start_idx + i] = 1
        input_space[start_idx:end_idx, end_idx - 1 - i] = 1

    # Add noise to input_space_flipped based on rand_lvl variable
    if noise_rand:
        for j in range(input_space.shape[0]):
            for l in range(input_space.shape[1]):
                mean, variance = input_space[j, l], noise_variance

                fuzzy_val = generate_uniform_random(mean=mean, variance=variance)

                if fuzzy_val > 1:
                    fuzzy_val = 1
                elif fuzzy_val < 0:
                    fuzzy_val = 0

                input_space[j, l] = fuzzy_val

    return input_space


def gen_x(
    input_dims: int,
    x_size: float,
    x_thickness: float,
    noise_rand: bool,
    noise_variance: float | int,
):

    # if input_dims % 2 == 0:
    #    raise UserWarning("Invalid input dimensions. Must be an odd value.")
    if not (0 <= x_size <= 1):
        raise ValueError("X size must be between 0 and 1.")
    if x_thickness <= 0:
        raise ValueError("X thickness must be greater than 0.")

    # Adjust X symbol dimensions for correct centering and sizing
    center = input_dims / 2
    half_diagonal = input_dims * x_size / 2

    # Define end points for the two lines that form the X
    line1_start = (center - half_diagonal, center + half_diagonal)
    line1_end = (center + half_diagonal, center - half_diagonal)
    line2_start = (center + half_diagonal, center + half_diagonal)
    line2_end = (center - half_diagonal, center - half_diagonal)

    # Initialize the input space with zeros
    input_space = np.zeros((input_dims, input_dims))

    # High-resolution grid for more accurate edge drawing
    subgrid_resolution = 100  # Number of pixels per square side
    subgrid_size = subgrid_resolution**2

    # Initialize the high-resolution binary mask for the hollow X
    high_res_x = np.zeros(
        (input_dims * subgrid_resolution, input_dims * subgrid_resolution)
    )

    # Function to draw lines on the high-resolution grid for X edges, adjusted for thickness
    def draw_line_high_res(v0, v1, thickness):
        # Calculate the line's direction vector
        dx = v1[0] - v0[0]
        dy = v1[1] - v0[1]
        normal = np.array([-dy, dx])
        normal_unit = normal / np.linalg.norm(normal)

        # Draw lines parallel to the original line to create thickness
        for offset in np.linspace(
            -thickness / 2, thickness / 2, num=int(thickness * subgrid_resolution)
        ):
            offset_vector = offset * normal_unit / subgrid_resolution
            v0_offset = v0 + offset_vector
            v1_offset = v1 + offset_vector
            rr, cc = line(
                int(v0_offset[1] * subgrid_resolution),
                int(v0_offset[0] * subgrid_resolution),
                int(v1_offset[1] * subgrid_resolution),
                int(v1_offset[0] * subgrid_resolution),
            )
            high_res_x[rr, cc] = 1

    # Draw the X's edges with thickness on the high-resolution grid
    draw_line_high_res(line1_start, line1_end, x_thickness)
    draw_line_high_res(line2_start, line2_end, x_thickness)

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
    max_value = np.mean(input_space)
    input_space_normalized = input_space / max_value if max_value > 0 else input_space

    # Add noise to input_space_flipped based on rand_lvl variable
    if noise_rand:
        for j in range(input_space_normalized.shape[0]):
            for l in range(input_space_normalized.shape[1]):
                mean, variance = input_space_normalized[j, l], noise_variance

                fuzzy_val = generate_uniform_random(mean=mean, variance=variance)

                if fuzzy_val > 1:
                    fuzzy_val = 1
                elif fuzzy_val < 0:
                    fuzzy_val = 0

                input_space_normalized[j, l] = fuzzy_val

    return input_space_normalized


def gen_circle(
    input_dims: int,
    circle_size: float,
    circle_thickness: int,
    noise_rand: bool,
    noise_variance: float | int,
):
    if not (0 <= circle_size <= 1):
        raise ValueError("Circle size must be between 0 and 1.")
    if circle_thickness <= 0:
        raise ValueError("Circle thickness must be greater than 0.")

    # Initialize the input space with zeros
    input_space = np.zeros((input_dims, input_dims))

    # Define circle center and radius for correct centering and sizing within the input space
    center = input_dims // 2
    radius = int(circle_size * input_dims / 2)

    # High-resolution grid for more accurate edge drawing
    subgrid_resolution = 100
    subgrid_size = subgrid_resolution**2

    # Initialize the high-resolution binary mask for the hollow circle
    high_res_circle = np.zeros(
        (input_dims * subgrid_resolution, input_dims * subgrid_resolution)
    )

    def draw_circle_with_thickness(center, radius, thickness):
        # Calculate the range of radii for the concentric circles
        min_radius = radius - thickness // 2
        max_radius = radius + thickness // 2

        for r in range(min_radius, max_radius + 1):
            rr, cc = circle_perimeter(
                center * subgrid_resolution,
                center * subgrid_resolution,
                r * subgrid_resolution,
            )
            # Ensure that the indices are within the bounds of the array
            rr = rr[(rr >= 0) & (rr < high_res_circle.shape[0])]
            cc = cc[(cc >= 0) & (cc < high_res_circle.shape[1])]
            high_res_circle[rr, cc] = 1

    # Draw the circle's edge with specified thickness
    draw_circle_with_thickness(center, radius, circle_thickness)

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
    max_value = np.mean(input_space)
    input_space_normalized = input_space / max_value if max_value > 0 else input_space

    # Add noise to input_space_flipped based on rand_lvl variable
    if noise_rand:
        for j in range(input_space_normalized.shape[0]):
            for l in range(input_space_normalized.shape[1]):
                mean, variance = input_space_normalized[j, l], noise_variance

                fuzzy_val = generate_uniform_random(mean=mean, variance=variance)

                if fuzzy_val > 1:
                    fuzzy_val = 1
                elif fuzzy_val < 0:
                    fuzzy_val = 0

                input_space_normalized[j, l] = fuzzy_val

    return input_space_normalized


def generate_uniform_random_clipped(mean, variance):
    a = mean - np.sqrt(3 * variance)
    b = mean + np.sqrt(3 * variance)

    # Generate a random number from U(a,b)
    random_number = np.random.uniform(a, b)

    # Clip the random number to be within [0, 1]
    random_number_clipped = np.clip(random_number, 0, 1)

    return random_number_clipped


def gen_blank(input_dims, noise_rand, noise_variance, mean):
    if noise_rand:
        # Use the modified function with clipping
        func = lambda i, j: generate_uniform_random_clipped(
            mean=mean, variance=noise_variance
        )

        # Generate an input_dims*input_dims array
        input_space = np.fromfunction(np.vectorize(func), (input_dims, input_dims))
    else:
        input_space = np.zeros(shape=(input_dims, input_dims))

    return input_space

data = gen_triangle(input_dims=40,
            triangle_size=0.7,
            triangle_thickness=230,
            noise_rand=True,
            noise_variance=0.1)

def input_space_plotted_single(data):

    # The function receives a 2D array of values
    sqr_side = int(np.sqrt(data.shape))

    # Convert 1D array to 2D
    data = np.reshape(data, (sqr_side, sqr_side))

    # Create a plt subplot
    fig, ax = plt.subplots()

    # Create plot
    ax.imshow(data, cmap="Greys", interpolation="nearest")

    plt.grid(visible=True, which="both")
    plt.show()

data = gen_blank(input_dims=40, noise_rand=True, noise_variance=0.05, mean=0.1)

data = data.flatten()


input_space_plotted_single(data)