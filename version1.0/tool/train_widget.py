import sys
import os
from PyQt5.QtWidgets import (
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
        self.slider_layout = QVBoxLayout()
        main_layout = QHBoxLayout()
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
        self.time_slider, self.time_label = self.create_slider(
            50, 500, 50, time, "Time"
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

        # Left column for sliders
        slider_column = QVBoxLayout()

        # Add sliders to the slider column
        sliders = [
            self.R_slider,
            self.A_slider,
            self.B_slider,
            self.P_slider,
            self.w_p_slider,
            self.beta_slider,
            self.delta_slider,
            self.time_slider,
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
            self.time_label,
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

        # Middle column for plotting (placeholder)
        plot_area = QVBoxLayout()
        plot_placeholder = QLabel("Plotting Area Placeholder")
        plot_placeholder.setFixedSize(400, 550)
        plot_placeholder.setStyleSheet("background-color: #DDDDDD")
        plot_area.addWidget(plot_placeholder)

        # Add the plot area to the main layout
        main_layout.addLayout(plot_area)

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

        # Define button and connect it
        self.button = QPushButton("Run Simulation")
        self.button.clicked.connect(self.on_button_clicked)

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
