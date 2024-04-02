# Gen data according to y number of classes
import os
import statistics
from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt

os.chdir(
    "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.0"
)
# os.chdir(
#    "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\version1.0"
# )

from gen.gen_symbol import *


def gen_float_data_(
    N_classes: int,
    N_input_neurons: int,
    items: int,
    noise_rand: bool,
    noise_variance: float | int,
    retur: bool,
    mean: float | int,
):
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
            triangle_thickness=230,
            noise_rand=noise_rand,
            noise_variance=noise_variance,
        ),
        lambda: gen_circle(
            input_dims=input_dims,
            circle_size=0.6,
            circle_thickness=3,
            noise_rand=noise_rand,
            noise_variance=noise_variance,
        ),
        lambda: gen_square(
            input_dims=input_dims,
            square_size=0.9,
            square_thickness=10,
            noise_rand=noise_rand,
            noise_variance=noise_variance,
        ),
        lambda: gen_x(
            input_dims=input_dims,
            x_size=0.6,
            x_thickness=200,
            noise_rand=noise_rand,
            noise_variance=noise_variance,
        ),
        lambda: gen_blank(
            input_dims=input_dims,
            noise_rand=True,
            noise_variance=noise_variance,
            mean=mean,
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
    if retur:
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
    print(items)
    N_input_neurons = data.shape[1]
    print(N_input_neurons)

    # Set time variable
    time = items * timesteps

    # Correct the dimensions for the 2D-array: time x neurons
    poisson_input = np.zeros((time, N_input_neurons))

    for i in range(items):  # Iterating over items
        for j in range(N_input_neurons):  # Iterating over neurons
            # Calculate the mean spike count for the Poisson distribution
            lambda_poisson = data[i, j] * dt * input_scaler

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

        with open(f"data/labels_train/testing_data_{rand_lvl}.pkl", "wb") as file:
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
    sum_ = []
    timepoints = data.shape[0] // 2
    for j in range(0, data.shape[0], 201):
        sum_.append(sum(data[j]))

    print(statistics.mean(sum_), timepoints * 0.001)
    # Calculate the average frequency
    average_frequency = round(statistics.mean(sum_) / ((timepoints * 0.001) / 2), 2)

    # Print the result
    print(f"This is the current spiking frequency: {average_frequency} Hz")

    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron Index")
    plt.show()


# Define randomness levels for the training data in a list
random_lvls = [0, 0.3, 0.5, 0.7]

for rand_lvl in random_lvls:
    print(f"generating training data at {rand_lvl} randomness level:\n")

    data, labels = gen_float_data_(
        N_classes=4,
        N_input_neurons=1600,
        items=40,
        noise_rand=True,
        noise_variance=rand_lvl,
        mean=2,
        retur=True
    )

    training_data, labels_train = float_2_pos_spike(
        data=data,
        labels=labels,
        timesteps=100,
        dt=0.001,
        input_scaler=2,
        save=False,
        retur=True,
        rand_lvl=rand_lvl,
    )

    raster_plot(training_data, labels_train)

    input_space_plotted_single(data[0])
    

data, labels = gen_float_data_(
    N_classes=4,
    N_input_neurons=1600,
    items=40,
    noise_rand=False,
    noise_variance=0.01,
    mean=0.1,
    retur=True
)

training_data, labels_train = float_2_pos_spike(
    data=data,
    labels=labels,
    timesteps=100,
    dt=0.001,
    input_scaler=5,
    save=False,
    retur=True,
    rand_lvl=0,
)

raster_plot(training_data, labels_train)

input_space_plotted_single(data[0])
input_space_plotted_single(data[1])