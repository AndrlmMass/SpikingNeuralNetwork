import sys
import os
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QSlider,
    QCheckBox,
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
        train_data,
    ):
        # Assume some range settings are already defined, you can adjust them as needed
        self.R_label, self.R_slider = self.create_slider(0.5, 2.0, 0.1, R, "R")
        self.A_label, self.A_slider = self.create_slider(0.001, 0.1, 0.001, A, "A")
        self.B_label, self.B_slider = self.create_slider(0.001, 0.1, 0.001, B, "B")
        self.P_label, self.P_slider = self.create_slider(0.05, 1.0, 0.05, P, "P")
        self.w_p_label, self.w_p_slider = self.create_slider(0.5, 2.0, 0.1, w_p, "w_p")
        self.beta_label, self.beta_slider = self.create_slider(
            0.001, 0.1, 0.001, beta, "beta"
        )
        self.delta_label, self.delta_slider = self.create_slider(
            0.001, 0.1, 0.001, delta, "delta"
        )
        self.time_label, self.time_slider = self.create_slider(
            50, 500, 50, time, "Time"
        )
        self.V_th_label, self.V_th_slider = self.create_slider(
            -60, -40, 1, V_th, "V_th"
        )
        self.V_rest_label, self.V_rest_slider = self.create_slider(
            -80, -60, 1, V_rest, "V_rest"
        )
        self.V_reset_label, self.V_reset_slider = self.create_slider(
            -90, -70, 1, V_reset, "V_reset"
        )
        self.dt_label, self.dt_slider = self.create_slider(0.01, 1.0, 0.01, dt, "dt")
        self.tau_m_label, self.tau_m_slider = self.create_slider(
            10, 30, 1, tau_m, "tau_m"
        )
        self.tau_const_label, self.tau_const_slider = self.create_slider(
            0.5, 2.0, 0.1, tau_const, "tau_const"
        )
        self.update_frequency_label, self.update_frequency_slider = self.create_slider(
            1, 50, 1, update_frequency, "Update frequency"
        )
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

        self.plot_weights_checkbox = QCheckBox("Plot Weights")
        self.plot_spikes_checkbox = QCheckBox("Plot Spikes")
        self.plot_weights_checkbox.setChecked(plot_weights)
        self.plot_spikes_checkbox.setChecked(plot_spikes)

        # Define button and connect it
        self.button = QPushButton("Run Simulation")
        self.button.clicked.connect(self.on_button_clicked)

        # Organize the sliders into groups using QVBoxLayout and QHBoxLayout
        param_box1 = QVBoxLayout()
        param_box2 = QVBoxLayout()
        param_box3 = QVBoxLayout()
        param_box4 = QVBoxLayout()
        checkboxes = QVBoxLayout()
        param_box1.addWidget(self.R_slider)
        param_box1.addWidget(self.A_slider)
        param_box1.addWidget(self.B_slider)
        param_box1.addWidget(self.P_slider)
        param_box2.addWidget(self.w_p_slider)
        param_box2.addWidget(self.beta_slider)
        param_box2.addWidget(self.delta_slider)
        param_box2.addWidget(self.time_slider)
        param_box3.addWidget(self.V_th_slider)
        param_box3.addWidget(self.V_rest_slider)
        param_box3.addWidget(self.V_reset_slider)
        param_box4.addWidget(self.dt_slider)
        param_box4.addWidget(self.tau_m_slider)
        param_box4.addWidget(self.tau_const_slider)
        param_box4.addWidget(self.update_frequency_slider)
        checkboxes.addWidget(self.plot_weights_checkbox)
        checkboxes.addWidget(self.plot_spikes_checkbox)

        # Main layout
        main_layout = QHBoxLayout()
        main_layout.addLayout(param_box1)
        main_layout.addLayout(param_box2)
        main_layout.addLayout(param_box3)
        main_layout.addLayout(param_box4)
        main_layout.addLayout(checkboxes)

        # Set main layout
        self.setLayout(main_layout)

    def create_slider(self, min_val, max_val, step, initial_val, label_text):
        # Scale factor to convert float values to int
        scale = 1 / step

        # Create slider
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(min_val * scale))
        slider.setMaximum(int(max_val * scale))
        slider.setSingleStep(1)
        slider.setValue(int(initial_val * scale))

        # Create label
        slider_label = QLabel(f"{label_text}: {initial_val:.2f}")
        slider.valueChanged.connect(
            lambda value, lbl=slider_label, lt=label_text, sc=scale: lbl.setText(
                f"{lt}: {value / sc:.2f}"
            )
        )

        return slider, slider_label

    def on_button_clicked(self):
        # Placeholder for running the simulation logic
        print("Simulation starting...")

        # Parse float values from labels
        R = float(self.R_label.text().split(": ")[1])
        A = float(self.A_label.text().split(": ")[1])
        B = float(self.B_label.text().split(": ")[1])
        P = float(self.P_label.text().split(": ")[1])
        w_p = float(self.w_p_label.text().split(": ")[1])
        beta = float(self.beta_label.text().split(": ")[1])
        delta = float(self.delta_label.text().split(": ")[1])
        time = int(
            self.time_label.text().split(": ")[1]
        )  # Assuming 'time' is intended as an integer
        V_th = int(self.V_th_label.text().split(": ")[1])
        V_rest = int(self.V_rest_label.text().split(": ")[1])
        V_reset = int(self.V_reset_label.text().split(": ")[1])
        dt = float(self.dt_label.text().split(": ")[1])
        tau_m = float(self.tau_m_label.text().split(": ")[1])
        tau_const = float(self.tau_const_label.text().split(": ")[1])
        update_frequency = int(self.update_frequency_label.text().split(": ")[1])

        # Call the train_data function with parsed label values
        train_data(
            R=R,
            A=A,
            B=B,
            P=P,
            w_p=w_p,
            beta=beta,
            delta=delta,
            time=time,
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
            callback=lambda W_se, W_ee, spikes, t: self.update_plots(
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
