import numpy as np
from skimage.draw import circle_perimeter, line


def generate_normal_value(mean=0, variance=0.1):
    std_dev = np.sqrt(variance)
    value = np.random.normal(mean, std_dev)
    return value


def draw_line_high_res(high_res_grid, v0, v1, thickness, subgrid_resolution):
    dx = v1[0] - v0[0]
    dy = v1[1] - v0[1]
    length = np.sqrt(dx**2 + dy**2)
    normal_unit = np.array([-dy / length, dx / length])

    for offset in range(-thickness // 2, thickness // 2):
        offset_vector = offset * normal_unit
        v0_offset = v0 + offset_vector / subgrid_resolution
        v1_offset = v1 + offset_vector / subgrid_resolution
        rr, cc = line(
            int(v0_offset[1] * subgrid_resolution),
            int(v0_offset[0] * subgrid_resolution),
            int(v1_offset[1] * subgrid_resolution),
            int(v1_offset[0] * subgrid_resolution),
        )
        for r, c in zip(rr, cc):
            if 0 <= r < high_res_grid.shape[0] and 0 <= c < high_res_grid.shape[1]:
                high_res_grid[r, c] = 1


def draw_circle_with_thickness(
    high_res_grid, center, radius, thickness, subgrid_resolution
):
    min_radius = int(radius - thickness // 2)
    max_radius = int(radius + thickness // 2)
    for r in range(min_radius, max_radius + 1):
        rr, cc = circle_perimeter(
            int(center * subgrid_resolution),
            int(center * subgrid_resolution),
            int(r * subgrid_resolution),
        )
        for i in range(len(rr)):
            if (
                0 <= rr[i] < high_res_grid.shape[0]
                and 0 <= cc[i] < high_res_grid.shape[1]
            ):
                high_res_grid[rr[i], cc[i]] = 1


def add_noise(input_space, noise_rand, noise_variance):
    if noise_rand:
        for j in range(input_space.shape[0]):
            for l in range(input_space.shape[1]):
                mean, variance = input_space[j, l], noise_variance
                fuzzy_val = generate_normal_value(mean=mean, variance=variance)
                fuzzy_val = np.clip(fuzzy_val, 0, 1)
                input_space[j, l] = fuzzy_val
    return input_space


def gen_triangle(
    input_dims, triangle_size, triangle_thickness, noise_rand, noise_variance
):
    input_space = np.zeros((input_dims, input_dims))
    center = input_dims / 2
    base_length = triangle_size * input_dims
    triangle_height = (np.sqrt(3) / 2) * base_length
    top_vertex = (center, center + triangle_height / 2)
    bottom_left_vertex = (center - base_length / 2, center - triangle_height / 2)
    bottom_right_vertex = (center + base_length / 2, center - triangle_height / 2)

    subgrid_resolution = 100
    subgrid_size = subgrid_resolution**2
    high_res_triangle = np.zeros(
        (input_dims * subgrid_resolution, input_dims * subgrid_resolution)
    )

    vertices = np.array([top_vertex, bottom_left_vertex, bottom_right_vertex])
    for i in range(3):
        draw_line_high_res(
            high_res_triangle,
            vertices[i],
            vertices[(i + 1) % 3],
            triangle_thickness,
            subgrid_resolution,
        )

    for i in range(input_dims):
        for j in range(input_dims):
            top = i * subgrid_resolution
            left = j * subgrid_resolution
            bottom = (i + 1) * subgrid_resolution
            right = (j + 1) * subgrid_resolution
            subgrid = high_res_triangle[top:bottom, left:right]
            coverage = np.sum(subgrid) / subgrid_size
            input_space[i, j] = coverage if np.sum(subgrid) > 0 else 0

    max_value = np.mean(input_space)
    input_space_normalized = input_space / max_value if max_value > 0 else input_space
    input_space_flipped = np.flipud(input_space_normalized)
    input_space_flipped = np.clip(input_space_flipped, a_min=0, a_max=1)

    return add_noise(input_space_flipped, noise_rand, noise_variance)


def gen_square(input_dims, square_size, square_thickness, noise_rand, noise_variance):
    input_space = np.zeros((input_dims, input_dims))
    square_length = int(square_size * input_dims)
    thickness = square_thickness
    start_idx = (input_dims - square_length) // 2
    end_idx = start_idx + square_length

    for i in range(thickness):
        input_space[start_idx + i, start_idx:end_idx] = 1
        input_space[end_idx - 1 - i, start_idx:end_idx] = 1

    for i in range(thickness):
        input_space[start_idx:end_idx, start_idx + i] = 1
        input_space[start_idx:end_idx, end_idx - 1 - i] = 1

    return add_noise(input_space, noise_rand, noise_variance)


def gen_x(input_dims, x_size, x_thickness, noise_rand, noise_variance):
    if not (0 <= x_size <= 1):
        raise ValueError("X size must be between 0 and 1.")
    if x_thickness <= 0:
        raise ValueError("X thickness must be greater than 0.")

    center = input_dims / 2
    half_diagonal = input_dims * x_size / 2
    line1_start = (center - half_diagonal, center + half_diagonal)
    line1_end = (center + half_diagonal, center - half_diagonal)
    line2_start = (center + half_diagonal, center + half_diagonal)
    line2_end = (center - half_diagonal, center - half_diagonal)

    input_space = np.zeros((input_dims, input_dims))
    subgrid_resolution = 100
    subgrid_size = subgrid_resolution**2
    high_res_x = np.zeros(
        (input_dims * subgrid_resolution, input_dims * subgrid_resolution)
    )

    draw_line_high_res(
        high_res_x, line1_start, line1_end, x_thickness, subgrid_resolution
    )
    draw_line_high_res(
        high_res_x, line2_start, line2_end, x_thickness, subgrid_resolution
    )

    for i in range(input_dims):
        for j in range(input_dims):
            top = i * subgrid_resolution
            left = j * subgrid_resolution
            bottom = (i + 1) * subgrid_resolution
            right = (j + 1) * subgrid_resolution
            subgrid = high_res_x[top:bottom, left:right]
            coverage = np.sum(subgrid) / subgrid_size
            input_space[i, j] = coverage if np.sum(subgrid) > 0 else 0

    max_value = np.mean(input_space)
    input_space_normalized = input_space / max_value if max_value > 0 else input_space
    input_space_normalized = np.clip(input_space_normalized, a_min=0, a_max=1)

    return add_noise(input_space_normalized, noise_rand, noise_variance)


def gen_circle(input_dims, circle_size, circle_thickness, noise_rand, noise_variance):
    if not (0 <= circle_size <= 1):
        raise ValueError("Circle size must be between 0 and 1.")
    if circle_thickness <= 0:
        raise ValueError("Circle thickness must be greater than 0.")

    input_space = np.zeros((input_dims, input_dims))
    center = input_dims // 2
    radius = int(circle_size * input_dims / 2)
    subgrid_resolution = 100
    subgrid_size = subgrid_resolution**2
    high_res_circle = np.zeros(
        (input_dims * subgrid_resolution, input_dims * subgrid_resolution)
    )

    draw_circle_with_thickness(
        high_res_circle, center, radius, circle_thickness, subgrid_resolution
    )

    for i in range(input_dims):
        for j in range(input_dims):
            top = i * subgrid_resolution
            left = j * subgrid_resolution
            bottom = (i + 1) * subgrid_resolution
            right = (j + 1) * subgrid_resolution
            subgrid = high_res_circle[top:bottom, left:right]
            coverage = np.sum(subgrid) / subgrid_size
            input_space[i, j] = coverage if np.sum(subgrid) > 0 else 0

    max_value = np.mean(input_space)
    input_space_normalized = input_space / max_value if max_value > 0 else input_space
    input_space_normalized = np.clip(input_space_normalized, a_min=0, a_max=1)

    return add_noise(input_space_normalized, noise_rand, noise_variance)


def gen_blank(input_dims, blank_variance, mean):
    std_dev = np.sqrt(blank_variance)
    input_space = np.random.normal(mean, std_dev, (input_dims, input_dims))
    return input_space
