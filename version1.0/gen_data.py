# Gen data according to y number of classes
import os
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

from gen_symbol import *


def gen_float_data_(
    N_classes,
    N_input_neurons,
    items,
    draw_bin=False,
    retur=False,
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
    labels = np.zeros((items, N_classes))

    # List of lambda functions wrapping the original functions with necessary arguments
    functions = [
        lambda: gen_triangle(
            input_dims=input_dims,
            triangle_size=0.7,
            triangle_thickness=230,
            draw_bin=draw_bin,
        ),
        lambda: gen_circle(
            input_dims=input_dims,
            circle_size=0.6,
            circle_thickness=3,
            draw_bin=draw_bin,
        ),
        lambda: gen_square(
            input_dims=input_dims,
            square_size=0.9,
            square_thickness=10,
            draw_bin=draw_bin,
        ),
        lambda: gen_x_symbol(
            input_dims=input_dims,
            x_size=0.6,
            receptor_size=1,
            x_thickness=200,
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
        labels[item, class_index] = 1

    # Reshape input_dims x input_dims to get time x input_dims**2
    input_space = np.reshape(input_space, (items, input_dims**2))

    # return if true
    if retur:
        return input_space, labels


def float_2_pos_spike(
    data,
    labels,
    N_input_neurons,
    timesteps,
    dt,
    input_scaler,
    train_2_test,
    save=False,
    retur=False,
):
    # Assert number of items
    items = data.shape[0]

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

    # Divide data and labels into training and testing
    training_data = poisson_input[: int(items * train_2_test) * timesteps]
    testing_data = poisson_input[int(items * train_2_test) * timesteps :]
    labels_train = labels[: int(items * train_2_test) * timesteps]
    labels_test = labels[int(items * train_2_test) * timesteps :]

    # save data if true
    if save:
        with open("data/training_data.pkl", "wb") as file:
            pickle.dump(training_data, file)
        print("training data is saved in data folder")

        with open("data/testing_data.pkl", "wb") as file:
            pickle.dump(testing_data, file)
        print("testing data is saved in data folder")

        with open("data/labels_train.pkl", "wb") as file:
            pickle.dump(labels_train, file)
        print("labels train are saved in data folder")

        with open("data/labels_test.pkl", "wb") as file:
            pickle.dump(labels_test, file)
        print("labels test are saved in data folder")

    if retur:
        return training_data, testing_data, labels_train, labels_test

# Plot input_data structure to ensure realistic creation
def input_space_plotted_single(data):
    # The function receives a 2D array of values

    # Convert 1D array to 2D
    data = np.reshape(data, (45, 45))
    
    # Create a plt subplot
    fig, ax = plt.subplots()

    # Create plot
    ax.imshow(data, cmap="Greys", interpolation="nearest")

    plt.grid(visible=True, which="both")
    
    plt.show()

# define function to create a raster plot of the input data
def raster_plot(data, labels):
    labels_name = ["tri", "O", "sq", "X"]
    indices = np.argmax(labels, axis=1)

    # Create raster plot with dots
    plt.figure(figsize=(10, 6))
    for neuron_index in range(2025):
        spike_times = np.where(data[:, neuron_index] == 1)[0]
        plt.scatter(
            spike_times, np.ones_like(spike_times) * neuron_index, color="black", s=10
        )  
    t = 0

    for item_boundary in range(0, data.shape[0], 100 + 1):
        # Get label name
        plt.axvline(x=item_boundary, color="red", linestyle="--")
        plt.text(
            x=item_boundary+25,
            y=1999,
            s=labels_name[indices[t]],
            size=20,
        )
        t += 1
    ax = plt.gca()
    ax.set_xlim([0,1600])
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron Index")
    plt.show()


data, labels = gen_float_data_(
    N_classes=4, N_input_neurons=2025, items=20, draw_bin=False, retur=True
)

training_data, testing_data, labels_train, labels_test = float_2_pos_spike(
    data=data,
    labels=labels,
    N_input_neurons=2025,
    timesteps=100,
    dt=0.001,
    input_scaler=0.5,
    train_2_test=0.8,
    save=True,
    retur=True,
)

raster_plot(training_data, labels_train)

input_space_plotted_single(data[2])