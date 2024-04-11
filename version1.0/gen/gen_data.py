# Gen data according to y number of classes
import os
import sys
from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt

if os.path.exists(
    "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork"
):
    os.chdir(
        "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.0"
    )
    sys.path.append(
        "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.0\\gen"
    )
else:
    os.chdir(
        "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\version1.0"
    )
    sys.path.append(
        "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\version1.0\\gen"
    )


from gen_symbol import *


def gen_float_data_(
    run: bool,
    N_classes: int,
    N_input_neurons: int,
    items: int,
    noise_rand: bool,
    noise_variance: float | int,
    mean: int | float,
    blank_variance: int | float,
):
    if not run:
        return

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
        input_space[item] = functions[t]()
        labels[item, t] = 1

        # Assign blank part after symbol-input
        input_space[item + 1] = functions[4]()
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
    timesteps: int,
    dt: float,
    input_scaler: int | float,
    save: bool,
    retur: bool,
    rand_lvl: float,
):
    # Data has shape items x neurons

    # Assert number of items
    items = data.shape[0]
    N_input_neurons = data.shape[1]

    # Set time variable
    time = items * timesteps

    # Correct the dimensions for the 2D-array: time x neurons
    poisson_input = np.zeros((time, N_input_neurons))

    for i in range(items):  # Iterating over items
        for j in range(N_input_neurons):  # Iterating over neurons
            # Calculate the mean spike count for the Poisson distribution
            lambda_poisson = data[i, j] * dt * input_scaler

            if lambda_poisson < 0 or np.isnan(lambda_poisson):
                lambda_poisson = 0

            # Generate spikes using Poisson distribution
            for t in range(timesteps):
                spike_count = np.random.poisson(lambda_poisson)
                index = i * timesteps + t
                poisson_input[index, j] = 1 if spike_count > 0 else 0

    # save data if true
    if save:
        print(
            f"Saving training and testing data with labels for random level {rand_lvl}"
        )
        with open(f"data/training_data/training_data_{rand_lvl}.pkl", "wb") as file:
            pickle.dump(poisson_input, file)
        print("training data is saved in data folder")

        with open(f"data/labels_train/labels_train_{rand_lvl}.pkl", "wb") as file:
            pickle.dump(labels, file)
        print("training labels is saved in data folder")

    if retur:
        return poisson_input, labels


# Plot input_data structure to ensure realistic creation
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


# define function to create a raster plot of the input data
def raster_plot(data, labels):
    labels_name = ["t", "o", "s", "x", " "]
    indices = np.argmax(labels, axis=1)

    # Create raster plot with dots
    plt.figure(figsize=(10, 6))
    for neuron_index in range(data.shape[1]):
        spike_times = np.where(data[:, neuron_index] == 1)[0]
        plt.scatter(
            spike_times, np.ones_like(spike_times) * neuron_index, color="black", s=10
        )
    t = 0

    for item_boundary in range(0, data.shape[0], 100):
        # Get label name
        plt.axvline(x=item_boundary, color="red", linestyle="--")
        plt.text(
            x=item_boundary + 25,
            y=1700,
            s=labels_name[indices[t]],
            size=12,
        )

        t += 1

    # Calculate the frequency of the spikes to check that it is acceptable
    t_counts = sum([np.sum(data[j : j + 100]) for j in range(0, data.shape[0], 800)])
    c_counts = sum([np.sum(data[j : j + 100]) for j in range(200, data.shape[0], 800)])
    s_counts = sum([np.sum(data[j : j + 100]) for j in range(400, data.shape[0], 800)])
    x_counts = sum([np.sum(data[j : j + 100]) for j in range(600, data.shape[0], 800)])
    b_counts = sum([np.sum(data[j : j + 100]) for j in range(100, data.shape[0], 200)])

    # Calculate the frquency of firing according to this formula: (spikes / possible spikes (units)) * timeunit (this is used to convert the unit to seconds, not milieconds)
    t_hz = t_counts / (100 * (t_counts // 100) * 0.001)
    c_hz = c_counts / (100 * (c_counts // 100) * 0.001)
    s_hz = s_counts / (100 * (s_counts // 100) * 0.001)
    x_hz = x_counts / (100 * (x_counts // 100) * 0.001)
    b_hz = b_counts / (100 * (b_counts // 100) * 0.001)

    print(f"t: {t_hz} Hz\nc: {c_hz} Hz\ns: {s_hz} Hz\nx: {x_hz} Hz\nb: {b_hz} Hz")

    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron Index")
    plt.show()
