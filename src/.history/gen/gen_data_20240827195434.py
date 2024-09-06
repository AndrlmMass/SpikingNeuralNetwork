# Gen data according to y number of classes
import os
import sys
import numpy as np
from tqdm import tqdm
from snntorch.spikegen import target_rate_code

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


class gen_data_cl:

    def __init__(
        self,
        N_classes: int,
        N_input_neurons: int,
        items: int,
        noise_rand: bool,
        noise_variance: float | int,
        mean: int | float,
        blank_variance: int | float,
        time: int | float,
        timesteps: int,
        dt: float,
        retur: bool,
        avg_high_freq: float | int,
        avg_low_freq: float | int,
        var_high_freq: float | int,
        var_low_freq: float | int,
        params_dict: dict,
    ):

        self.N_classes = N_classes
        self.N_input_neurons = N_input_neurons
        self.items = items
        self.noise_rand = noise_rand
        self.noise_variance = noise_variance
        self.mean = mean
        self.blank_variance = blank_variance
        self.time = time
        self.timesteps = timesteps
        self.dt = dt
        self.retur = retur
        self.avg_high_freq = avg_high_freq
        self.avg_low_freq = avg_low_freq
        self.var_high_freq = var_high_freq
        self.var_low_freq = var_low_freq
        self.params_dict = params_dict

    def gen_float_data_(self):
        # Add print statement to show funct

        # Check if n_classes and items are compatible
        if self.items % self.N_classes != 0:
            raise UserWarning(
                "Invalid items or classes value initiated. must be divisible by each other"
            )

        # Define input shape
        input_dims = int(np.sqrt(self.N_input_neurons))

        if input_dims**2 != self.N_input_neurons:
            raise ValueError("N_input_neurons must be a perfect square")

        # Assert input space based on input_dims variable
        input_space = np.zeros((self.items, input_dims, input_dims))
        self.labels = np.zeros((self.items, self.N_classes + 1))

        # List of lambda functions wrapping the original functions with necessary arguments
        functions = [
            lambda: gen_triangle(
                input_dims=input_dims,
                triangle_size=0.7,
                triangle_thickness=250,
                noise_rand=self.noise_rand,
                noise_variance=self.noise_variance,
            ),
            lambda: gen_circle(
                input_dims=input_dims,
                circle_size=0.7,
                circle_thickness=3,
                noise_rand=self.noise_rand,
                noise_variance=self.noise_variance,
            ),
            lambda: gen_square(
                input_dims=input_dims,
                square_size=0.6,
                square_thickness=4,
                noise_rand=self.noise_rand,
                noise_variance=self.noise_variance,
            ),
            lambda: gen_x(
                input_dims=input_dims,
                x_size=0.8,
                x_thickness=350,
                noise_rand=self.noise_rand,
                noise_variance=self.noise_variance,
            ),
            lambda: gen_blank(
                input_dims=input_dims,
                blank_variance=self.blank_variance,
                mean=self.mean,
            ),
        ]

        # Ensure we have enough functions for the requested classes
        if self.N_classes > len(functions):
            raise ValueError(
                "Not enough functions to generate symbols for the requested number of classes"
            )
        # Loop over items to generate symbols
        t = 0
        for item in tqdm(range(0, self.items, 2), ncols=100):

            # Execute the lambda function for the current class_index and assign its output
            input_space[item] = np.clip(functions[t](), a_min=0, a_max=1)
            self.labels[item, t] = 1

            # Assign blank part after symbol-input
            input_space[item + 1] = np.clip(functions[4](), a_min=0, a_max=1)
            self.labels[item + 1, 4] = 1

            if t == self.N_classes - 1:
                t = 0
            else:
                t += 1

        # Reshape input_dims x input_dims to get time x input_dims**2
        self.data = np.reshape(input_space, (int(self.items), input_dims**2))

    def float_2_pos_spike(self):
        # Correct the dimensions for the 2D-array: time x neurons
        poisson_input = np.zeros((self.time, self.N_input_neurons))

        # Correct the dimensions for the 2D-array: time x neurons
        for i in range(self.items):  # Iterating over items
            for j in range(self.N_input_neurons):  # Iterating over neurons
                print(
                    poisson_input[
                        i * self.timesteps : i * self.timesteps + self.timesteps, j
                    ].shape()
                )
                poisson_input[
                    i * self.timesteps : i * self.timesteps + self.timesteps, j
                ] = target_rate_code(num_steps=self.timesteps, rate=self.data[i, j])

        # Extend labels to match the poisson_input
        labels_bin = np.repeat(self.labels, self.timesteps, axis=0)

        if self.retur:
            return poisson_input, labels_bin, self.data, self.labels
