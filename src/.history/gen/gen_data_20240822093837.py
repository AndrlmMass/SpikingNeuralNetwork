# Gen data according to y number of classes
import os
import sys
import numpy as np
from tqdm import tqdm
import threading

if os.path.exists(
    "C:\\Users\\Bruker\\OneDrive\\Documents\\NMBU_\\BONSAI\\SNN\\SpikingNeuralNetwork\\version1.0"
):
    base_path = "C:\\Users\\Bruker\\OneDrive\\Documents\\NMBU_\\BONSAI\\SNN\\SpikingNeuralNetwork\\version1.0"
else:
    base_path = "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\version1.0"

os.chdir(base_path)

sys.path.append(os.path.join(base_path, "gen"))
sys.path.append(os.path.join(base_path, "main"))

from gen_symbol import *
from train import display_animation


def gen_float_data_(
    N_classes: int,
    N_input_neurons: int,
    items: int,
    noise_rand: bool,
    noise_variance: float | int,
    mean: int | float,
    blank_variance: int | float,
):
    # Add print statement to show funct

    # Check if n_classes and items are compatible
    if items % N_classes != 0:
        raise UserWarning(
            "Invalid items or classes value initiated. must be divisible by each other"
        )

    # Define input shape
    input_dims = int(np.sqrt(N_input_neurons))

    if input_dims**2 != N_input_neurons:
        raise ValueError("N_input_neurons must be a perfect square")

    # Assert input space based on input_dims variable
    input_space = np.zeros((items, input_dims, input_dims))
    labels = np.zeros((items, N_classes + 1))

    # List of lambda functions wrapping the original functions with necessary arguments
    functions = [
        lambda: gen_triangle(
            input_dims=input_dims,
            triangle_size=0.7,
            triangle_thickness=250,
            noise_rand=noise_rand,
            noise_variance=noise_variance,
        ),
        lambda: gen_circle(
            input_dims=input_dims,
            circle_size=0.7,
            circle_thickness=3,
            noise_rand=noise_rand,
            noise_variance=noise_variance,
        ),
        lambda: gen_square(
            input_dims=input_dims,
            square_size=0.6,
            square_thickness=4,
            noise_rand=noise_rand,
            noise_variance=noise_variance,
        ),
        lambda: gen_x(
            input_dims=input_dims,
            x_size=0.8,
            x_thickness=350,
            noise_rand=noise_rand,
            noise_variance=noise_variance,
        ),
        lambda: gen_blank(
            input_dims=input_dims, blank_variance=blank_variance, mean=mean
        ),
    ]

    # Ensure we have enough functions for the requested classes
    if N_classes > len(functions):
        raise ValueError(
            "Not enough functions to generate symbols for the requested number of classes"
        )
    # Loop over items to generate symbols
    t = 0
    for item in tqdm(range(0, items, 2), ncols=100):

        # Execute the lambda function for the current class_index and assign its output
        input_space[item] = np.clip(functions[t](), a_min=0, a_max=1)
        labels[item, t] = 1

        # Assign blank part after symbol-input
        input_space[item + 1] = np.clip(functions[4](), a_min=0, a_max=1)
        labels[item + 1, 4] = 1

        if t == N_classes - 1:
            t = 0
        else:
            t += 1

    # Reshape input_dims x input_dims to get time x input_dims**2
    input_space = np.reshape(input_space, (int(items), input_dims**2))

    # return if true
    return input_space, labels


def float_2_pos_spike(
    data: np.ndarray,
    labels: np.ndarray,
    items: int,
    time: int | float,
    timesteps: int,
    dt: float,
    save: bool,
    retur: bool,
    avg_high_freq: float | int,
    avg_low_freq: float | int,
    var_high_freq: float | int,
    var_low_freq: float | int,
    params_dict: dict,
):
    # Assert number of items
    N_input_neurons = data.shape[1]

    # Correct the dimensions for the 2D-array: time x neurons
    poisson_input = np.zeros((time, N_input_neurons))

    # Correct the dimensions for the 2D-array: time x neurons
    for i in range(items):  # Iterating over items
        for j in range(N_input_neurons):  # Iterating over neurons
            if data[i, j] < 0.3:  # Arbitrary limit, might need better reasoning
                lambda_poisson = np.random.normal(
                    avg_low_freq, var_low_freq
                )  # Average firing rate of 1 Hz (mu=1, delta=0.2)
            else:
                lambda_poisson = np.random.normal(
                    avg_high_freq, var_high_freq
                )  # Average firing rate of 10 Hz (mu=10, delta=0.2)
            for t in range(timesteps):
                spike_count = np.random.poisson(lambda_poisson * dt)
                poisson_input[t + i * timesteps, j] = spike_count

    # Extend labels to match the poisson_input
    labels_bin = np.repeat(labels, timesteps, axis=0)

    # save data if true
    if save:
        print("Saving training data and labels", end="\r")

        stop_event = threading.Event()
        animation_thread = threading.Thread(target=display_animation, args=(stop_event))
        animation_thread.start()

        rand_nums = np.random.randint(low=0, high=9, size=5)

        # Check if name is taken
        while rand_nums in os.listdir("data"):
            rand_nums = np.random.randint(low=0, high=9, size=5)

        # Create folder to store data
        os.makedirs(f"data\\{rand_nums}")

        # Save training data and labels
        np.save(f"data\\{rand_nums}\\data_bin.npy", poisson_input)
        np.save(f"data\\{rand_nums}\\labels_bin.npy", labels_bin)
        np.save(f"data\\{rand_nums}\\labels.npy", labels)
        np.save(f"data\\{rand_nums}\\data.npy", data)
        np.save(f"parameters.npy", params_dict)

        stop_event.set()

        animation_thread.join()

        print("Saving complete")

    if retur:
        return poisson_input, labels_bin, data, labels
