# Create an interactive ipython script that communicates with train.property

# Import libraries
import matplotlib.pyplot as plt
from ipywidgets import FloatSlider, IntSlider, Checkbox, Output, Button, VBox, HBox
import os
import sys
import numpy as np

# Set current working directories and add relevant directories to path
if os.path.exists(
    "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.0"
):
    os.chdir(
        "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.0\\main"
    )
else:
    sys.path.append(
        "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\version1.0\\main"
    )

from train import *


class ipy_widget:

    def init_gui(
        self,
        V_th,
        V_reset,
        P,
        C,
        R,
        A,
        B,
        w_p,
        beta,
        delta,
        time,
        V_rest,
        dt,
        tau_m,
        tau_const,
        update_frequency,
        plot_weights,
        plot_spikes,
        N_excit_neurons,
        N_inhib_neurons,
        N_input_neurons,
        MemPot,
        W_se,
        W_se_ideal,
        W_ee,
        W_ee_ideal,
        W_ei,
        W_ei_ideal,
        W_ie,
        W_ie_ideal,
    ):
        self.R_slider = FloatSlider(
            value=R, min=0.5, max=2.0, step=0.1, description="R"
        )
        self.A_slider = FloatSlider(
            value=A, min=0.001, max=0.1, step=0.001, description="A"
        )
        self.B_slider = FloatSlider(
            value=B, min=0.001, max=0.1, step=0.001, description="B"
        )
        self.P_slider = FloatSlider(
            value=P, min=0.05, max=1.0, step=0.05, description="P"
        )
        self.w_p_slider = FloatSlider(
            value=w_p, min=0.5, max=2.0, step=0.1, description="w_p"
        )
        self.beta_slider = FloatSlider(
            value=beta, min=0.001, max=0.1, step=0.001, description="beta"
        )
        self.delta_slider = FloatSlider(
            value=delta, min=0.001, max=0.1, step=0.001, description="delta"
        )
        self.time_slider = IntSlider(
            value=time, min=50, max=500, step=50, description="Time"
        )
        self.V_th_slider = IntSlider(
            value=V_th, min=-60, max=-40, step=1, description="V_th"
        )
        self.V_rest_slider = IntSlider(
            value=V_rest, min=-80, max=-60, step=1, description="V_rest"
        )
        self.V_reset_slider = IntSlider(
            value=V_reset, min=-90, max=-70, step=1, description="V_reset"
        )
        self.dt_slider = FloatSlider(
            value=dt, min=0.01, max=1.0, step=0.01, description="dt"
        )
        self.tau_m_slider = FloatSlider(
            value=tau_m, min=10, max=30, step=1, description="tau_m"
        )
        self.tau_const_slider = FloatSlider(
            value=tau_const, min=0.5, max=2.0, step=0.1, description="tau_const"
        )
        self.update_frequency_slider = IntSlider(
            value=update_frequency,
            min=1,
            max=50,
            step=1,
            description="Update frequency",
        )
        self.plot_weights_checkbox = Checkbox(
            value=plot_weights, description="Plot Weights"
        )
        self.plot_spikes_checkbox = Checkbox(
            value=plot_spikes, description="Plot Spikes"
        )

        self.output = Output
        self.plot_output = Output
        self.button = Button(description="run simulation")

        # Organize the sliders into groups using VBox and HBox
        param_box1 = VBox([self.R_slider, self.A_slider, self.B_slider, self.P_slider])
        param_box2 = VBox(
            [self.w_p_slider, self.beta_slider, self.delta_slider, self.time_slider]
        )
        param_box3 = VBox([self.V_th_slider, self.V_rest_slider, self.V_reset_slider])
        param_box4 = VBox(
            [
                self.dt_slider,
                self.tau_m_slider,
                self.tau_const_slider,
                self.update_frequency_slider,
            ]
        )
        checkboxes = VBox([self.plot_weights_checkbox, self.plot_spikes_checkbox])

        control_box = HBox([param_box1, param_box2, param_box3, param_box4, checkboxes])

        # Make variables globally available
        self.N_excit_neurons = N_excit_neurons
        self.N_inhib_neurons = N_inhib_neurons
        self.N_input_neurons = N_input_neurons
        self.MemPot = MemPot
        self.W_se = W_se
        self.W_se_ideal = W_se_ideal
        self.W_ee = W_ee
        self.W_ee_ideal = W_ee_ideal
        self.W_ei = W_ei
        self.W_ei_ideal = W_ei_ideal
        self.W_ie = W_ie
        self.W_ie_ideal = W_ie_ideal

    def update_plots(self, W_se, W_ee, spikes, t):
        with self.plot_output:
            clear_output(wait=True)  # Clear the existing plots
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # Plotting weights
            if self.plot_weights_checkbox.value:
                # Concatenate stimulation-excitation and excitatio-excitation weights
                weights = np.concatenate(W_se, W_ee, axis=1)
                axes[0].imshow(
                    weights, aspect="auto", interpolation="nearest", cmap="coolwarm"
                )
                axes[0].set_title(f"Weights at time {t}")
                axes[0].set_xlabel("Neurons")
                axes[0].set_ylabel("Weights")

            # Plotting spikes
            if self.plot_spikes_checkbox.value:
                axes[1].imshow(
                    spikes, aspect="auto", interpolation="nearest", cmap="binary"
                )
                axes[1].set_title(f"Spikes at time {t}")
                axes[1].set_xlabel("Time")
                axes[1].set_ylabel("Neurons")

            plt.tight_layout()
            plt.show()

    def on_button_clicked(self, b):
        with self.output:
            clear_output(wait=True)
            print("Simulation starting...")

            train_data(
                R=self.R_slider.value,
                A=self.A_slider.value,
                B=self.B_slider.value,
                P=self.P_slider.value,
                w_p=self.w_p_slider.value,
                beta=self.beta_slider.value,
                delta=self.delta_slider.value,
                time=self.time_slider.value,
                V_th=self.V_th_slider.value,
                V_rest=self.V_rest_slider.value,
                V_reset=self.V_reset_slider.value,
                dt=self.dt_slider.value,
                tau_m=self.tau_m_slider.value,
                tau_const=self.tau_const_slider.value,
                training_data=self.train_data,
                N_excit_neurons=self.N_excit_neurons,
                N_inhib_neurons=self.N_inhib_neurons,
                N_input_neurons=self.N_input_neurons,
                MemPot=self.MemPot,
                W_se=self.W_se,
                W_se_ideal=self.W_se_ideal,
                W_ee=self.W_ee,
                W_ee_ideal=self.W_ee_ideal,
                W_ei=self.W_ei,
                W_ei_ideal=self.W_ei_ideal,
                W_ie=self.W_ie,
                W_ie_ideal=self.W_ie_ideal,
                callback=lambda W_se, W_ee, spikes, t: self.update_plots(
                    W_se, W_ee, spikes, t
                ),
                update_frequency=self.update_frequency_slider.value,
            )

    def run_gui(self):
        self.button.on_click(self.on_button_clicked)
        display(self.control_box, self.button, self.output, self.plot_output)
