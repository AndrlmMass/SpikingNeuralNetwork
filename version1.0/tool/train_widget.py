import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QMainWindow,
    QSlider,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLabel,
    QAction,
)
from PyQt5.QtCore import Qt

# Set current working directories and add relevant directories to path
if os.path.exists(
    "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.0"
):
    sys.path.append(
        "C:\\Users\\andre\\OneDrive\\Documents\\NMBU_\\BONSAI\\SpikingNeuralNetwork\\version1.0\\main"
    )
else:
    sys.path.append(
        "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\version1.0\\main"
    )

from train import *


class MainWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    def init_gui(
        self,
        V_th,
        V_reset,
        P,
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
        train_data,
    ):
        # Main layout with horizontal orientation
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(
            0, 0, 0, 0
        )  # Remove margins around the main layout
        main_layout.setSpacing(0)  # Remove spacing between elements in the main layout

        # Left column for sliders
        slider_column = QVBoxLayout()
        slider_column.setContentsMargins(
            15, 0, 10, 10
        )  # Remove margins around the slider column
        slider_column.setSpacing(5)
        slider_column.setAlignment(Qt.AlignLeft)
        # Assume some range settings are already defined, you can adjust them as needed
        self.R_slider, self.R_label = self.create_slider(0.5, 2.0, 0.1, R, "R")
        self.A_slider, self.A_label = self.create_slider(0.001, 0.1, 0.001, A, "A")
        self.B_slider, self.B_label = self.create_slider(0.001, 0.1, 0.001, B, "B")
        self.P_slider, self.P_label = self.create_slider(0.05, 1.0, 0.05, P, "P")
        self.w_p_slider, self.w_p_label = self.create_slider(0.5, 2.0, 0.1, w_p, "w_p")
        self.beta_slider, self.beta_label = self.create_slider(
            0.001, 0.1, 0.001, beta, "beta"
        )
        self.delta_slider, self.delta_label = self.create_slider(
            0.001, 0.1, 0.001, delta, "delta"
        )
        self.V_th_slider, self.V_th_label = self.create_slider(
            -60, -40, 1, V_th, "V_th"
        )
        self.V_rest_slider, self.V_rest_label = self.create_slider(
            -80, -60, 1, V_rest, "V_rest"
        )
        self.V_reset_slider, self.V_reset_label = self.create_slider(
            -90, -70, 1, V_reset, "V_reset"
        )
        self.dt_slider, self.dt_label = self.create_slider(0.01, 1.0, 0.01, dt, "dt")
        self.tau_m_slider, self.tau_m_label = self.create_slider(
            10, 30, 1, tau_m, "tau_m"
        )
        self.tau_const_slider, self.tau_const_label = self.create_slider(
            0.5, 2.0, 0.1, tau_const, "tau_const"
        )
        self.update_frequency_slider, self.update_frequency_label = self.create_slider(
            1, 50, 1, update_frequency, "Update frequency"
        )

        # Add sliders to the slider column
        sliders = [
            self.R_slider,
            self.A_slider,
            self.B_slider,
            self.P_slider,
            self.w_p_slider,
            self.beta_slider,
            self.delta_slider,
            self.V_th_slider,
            self.V_rest_slider,
            self.V_reset_slider,
            self.dt_slider,
            self.tau_m_slider,
            self.tau_const_slider,
            self.update_frequency_slider,
        ]
        labels = [
            self.R_label,
            self.A_label,
            self.B_label,
            self.P_label,
            self.w_p_label,
            self.beta_label,
            self.delta_label,
            self.V_th_label,
            self.V_rest_label,
            self.V_reset_label,
            self.dt_label,
            self.tau_m_label,
            self.tau_const_label,
            self.update_frequency_label,
        ]

        for slider, label in zip(sliders, labels):
            slider_column.addWidget(label)
            slider_column.addWidget(slider)

        main_layout.addLayout(slider_column)

        self.figure_weights = plt.Figure()
        self.canvas_weights = FigureCanvas(self.figure_weights)
        self.ax_weights = self.figure_weights.add_subplot(111)

        self.figure_spikes = plt.Figure()
        self.canvas_spikes = FigureCanvas(self.figure_spikes)
        self.ax_spikes = self.figure_spikes.add_subplot(111)

        # Layout to place the canvas
        main_layout.addWidget(self.canvas_weights)
        main_layout.addWidget(self.canvas_spikes)

        self.N_excit_neurons = N_excit_neurons
        self.N_inhib_neurons = N_inhib_neurons
        self.N_input_neurons = N_input_neurons
        self.train_data = train_data
        self.MemPot = MemPot
        self.W_se = W_se
        self.W_se_ideal = W_se_ideal
        self.W_ee = W_ee
        self.W_ee_ideal = W_ee_ideal
        self.W_ei = W_ei
        self.W_ei_ideal = W_ei_ideal
        self.W_ie = W_ie
        self.W_ie_ideal = W_ie_ideal
        self.time = time

        # Define button and connect it
        self.button = QPushButton("Run Simulation")
        self.button.clicked.connect(self.on_button_clicked)
        main_layout.addWidget(self.button)

        # Set main layout
        self.setLayout(main_layout)

    def update_plots(self, W_se, W_ee, spikes, t):
        # Update the weights plot
        self.ax_weights.clear()
        combined_weights = np.concatenate((W_se[t], W_ee[t]), axis=1)
        self.ax_weights.matshow(combined_weights, aspect="auto")
        self.ax_weights.set_title(f"Weights at Time {t}")
        self.canvas_weights.draw()

        # Update the spikes plot
        self.ax_spikes.clear()
        self.ax_spikes.imshow(spikes[:t, :], aspect="auto", interpolation="nearest")
        self.ax_spikes.set_title(f"Spikes up to Time {t}")
        self.canvas_spikes.draw()

    def create_slider(self, min_val, max_val, step, initial_val, label_text):
        # Scale factor to convert float values to int
        scale = 1 / step

        # Create slider
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(min_val * scale))
        slider.setMaximum(int(max_val * scale))
        slider.setSingleStep(1)
        slider.setValue(int(initial_val * scale))
        slider.setFixedWidth(100)

        # Create label
        slider_label = QLabel(f"{label_text}: {initial_val:.2f}")
        slider.valueChanged.connect(
            lambda value, lbl=slider_label, lt=label_text, sc=scale: lbl.setText(
                f"{lt}: {value / sc:.2f}"
            )
        )

        return slider, slider_label

    def on_button_clicked(self):
        # Parse float values from labels
        R = float(self.R_label.text().split(": ")[1])
        A = float(self.A_label.text().split(": ")[1])
        B = float(self.B_label.text().split(": ")[1])
        P = float(self.P_label.text().split(": ")[1])
        w_p = float(self.w_p_label.text().split(": ")[1])
        beta = float(self.beta_label.text().split(": ")[1])
        delta = float(self.delta_label.text().split(": ")[1])
        V_th = float(self.V_th_label.text().split(": ")[1])
        V_rest = float(self.V_rest_label.text().split(": ")[1])
        V_reset = float(self.V_reset_label.text().split(": ")[1])
        dt = float(self.dt_label.text().split(": ")[1])
        tau_m = float(self.tau_m_label.text().split(": ")[1])
        tau_const = float(self.tau_const_label.text().split(": ")[1])
        update_frequency = float(self.update_frequency_label.text().split(": ")[1])

        # Call the train_data function with parsed label values
        train_data(
            R=R,
            A=A,
            B=B,
            P=P,
            w_p=w_p,
            beta=beta,
            delta=delta,
            time=self.time,
            V_th=V_th,
            V_rest=V_rest,
            V_reset=V_reset,
            dt=dt,
            tau_m=tau_m,
            tau_const=tau_const,
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
            callback=lambda spikes, W_se, W_ee, t: self.update_plots(
                W_se, W_ee, spikes, t
            ),
            update_frequency=update_frequency,
        )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set the central widget and the window's layout
        self.setWindowTitle("PyQt5 Interface")
        self.setGeometry(100, 100, 800, 600)  # Position and size: x, y, width, height

        central_widget = MainWidget(self)
        self.setCentralWidget(central_widget)

        # Optionally add menu, status bar, etc.
        self.statusBar().showMessage("Ready")
        self.setup_menu()

    def setup_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
