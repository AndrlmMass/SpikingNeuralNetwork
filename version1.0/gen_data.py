# Gen data according to y number of classes
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

os.chdir(
    "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\version1.0"
)

from gen_symbol import *


import numpy as np
from functools import partial

# Assuming your import statement for gen_symbol functions is here


def gen_data_(N_classes, N_input_neurons, items, draw_bin=False):
    # Define input shape
    input_dims = int(np.sqrt(N_input_neurons))

    if input_dims**2 != N_input_neurons:
        raise ValueError("N_input_neurons must be a perfect square")

    # Assert input space based on input_dims variable
    input_space = np.zeros((items, input_dims, input_dims))

    # List of lambda functions wrapping the original functions with necessary arguments
    functions = [
        lambda: gen_triangle(
            input_dims=input_dims,
            triangle_size=0.6,
            triangle_thickness=20,
            draw_bin=draw_bin,
        ),
        lambda: gen_circle(
            input_dims=input_dims,
            circle_size=0.6,
            receptor_size=1,
            circle_thickness=20,
            draw_bin=draw_bin,
        ),
        lambda: gen_square(
            input_dims=input_dims,
            square_size=0.6,
            square_thickness=20,
            draw_bin=draw_bin,
        ),
        lambda: gen_x_symbol(
            input_dims=input_dims,
            x_size=0.6,
            receptor_size=1,
            x_thickness=20,
            draw_bin=draw_bin,
        ),
    ]

    # Ensure we have enough functions for the requested classes
    if N_classes > len(functions):
        raise ValueError(
            "Not enough functions to generate symbols for the requested number of classes"
        )

    # Loop over items to generate symbols
    for item in tqdm(range(items), ncols=100):
        class_index = item % N_classes
        # Execute the lambda function for the current class_index and assign its output
        input_space[item] = functions[class_index]()

    return input_space


input_space = gen_data_(N_classes=4, N_input_neurons=4096, items=100, draw_bin=False)

print(input_space.shape)
for j in range(input_space.shape[0]):
    plt.figure(figsize=(8, 8))  # You can adjust the figure size as needed
    plt.imshow(input_space[j, :, :], cmap="gray", vmin=0, vmax=1)
    plt.axis("off")  # To not display axis
    plt.title(f"Item {j + 1}")
    plt.show()

    inp = input("Do you want to continue?")
    if inp == "":
        continue
    else:
        break
