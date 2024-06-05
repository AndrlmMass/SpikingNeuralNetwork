# Train network script
import numpy as np
from tqdm import tqdm
from numba import njit
import time
import threading
import os
import sys

# Set the current directory based on the existence of a specific path
if os.path.exists(
    "C:\\Users\\Bruker\\OneDrive\\Documents\\NMBU_\\BONSAI\\SNN\\SpikingNeuralNetwork\\version1.0"
):
    base_path = "C:\\Users\\Bruker\\OneDrive\\Documents\\NMBU_\\BONSAI\\SNN\\SpikingNeuralNetwork\\version1.0"
else:
    base_path = "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\version1.0"

os.chdir(base_path)
sys.path.append(os.path.join(base_path, "main"))
sys.path.append(os.path.join(base_path, "plot"))
sys.path.append(os.path.join(base_path, "tool"))
sys.path.append(os.path.join(base_path, "train_packages"))

from plot_training import *
from plot_network import *
from membrane_potential import (
    adjust_membrane_threshold,
    update_membrane_potential,
)
from weight_updating import exc_weight_update, inh_weight_update

import numpy as np
import os
from tqdm import tqdm


def load_model(folder, stop_event):
    W_se = np.load(f"model/{folder}/W_se.npy")
    W_ee = np.load(f"model/{folder}/W_ee.npy")
    W_ie = np.load(f"model/{folder}/W_ie.npy")
    W_ei = np.load(f"model/{folder}/W_ei.npy")
    spikes = np.load(f"model/{folder}/spikes.npy")
    MemPot = np.load(f"model/{folder}/MemPot.npy")
    pre_synaptic_trace = np.load(f"model/{folder}/pre_synaptic_trace.npy")
    post_synaptic_trace = np.load(f"model/{folder}/post_synaptic_trace.npy")
    slow_pre_synaptic_trace = np.load(f"model/{folder}/slow_pre_synaptic_trace.npy")
    z_istdp = np.load(f"model/{folder}/z_istdp_trace.npy")
    I_in_ls = np.load(f"model/{folder}/I_in_ls.npy")

    # Signal that loading is complete
    stop_event.set()

    return (
        W_se,
        W_ee,
        W_ie,
        W_ei,
        spikes,
        MemPot,
        pre_synaptic_trace,
        post_synaptic_trace,
        slow_pre_synaptic_trace,
        z_istdp,
        I_in_ls,
    )


def display_animation(stop_event):
    waiting_animation = [".  ", ".. ", "...", ".. ", ".  ", "   "]
    idx = 0
    while not stop_event.is_set():
        print(f"Reloading model{waiting_animation[idx]}", end="\r")
        idx = (idx + 1) % len(waiting_animation)
        time.sleep(0.2)  # Adjust the speed of the animation as needed


def train_data(
    R: float | int,
    A: float | int,
    P: float | int,
    w_p: float | int,
    beta: float | int,
    delta: float | int,
    time: int,
    V_th_: int,
    V_rest: int,
    V_reset: int,
    dt: float | int,
    tau_plus: float | int,
    tau_minus: float | int,
    tau_slow: float | int,
    tau_m: float | int,
    tau_ht: float | int,
    tau_hom: float | int,
    tau_const: float | int,
    tau_H: float | int,
    tau_stdp: float | int,
    tau_thr: float | int,
    learning_rate: float | int,
    training_data: np.ndarray,
    N_excit_neurons: int,
    N_inhib_neurons: int,
    N_input_neurons: int,
    MemPot: np.ndarray,
    max_weight: float | int,
    min_weight: float | int,
    W_se: np.ndarray,
    W_se_ideal: np.ndarray,
    W_se_2d: np.ndarray,
    W_se_plt_idx: np.ndarray,
    W_ee: np.ndarray,
    W_ee_ideal: np.ndarray,
    W_ee_2d: np.ndarray,
    W_ee_plt_idx: np.ndarray,
    W_ei: np.ndarray,
    W_ei_ideal: np.ndarray,
    W_ei_2d: np.ndarray,
    W_ei_plt_idx: np.ndarray,
    W_ie: np.ndarray,
    W_ie_ideal: np.ndarray,
    W_ie_2d: np.ndarray,
    W_ie_plt_idx: np.ndarray,
    gamma: float | int,
    save_model: bool,
    item_lim: int,
    items: int,
):
    # Get all local variables
    locs = locals()

    # Define keys to exclude
    exclude_keys = [
        "training_data",
        "MemPot",
        "W_se",
        "W_se_ideal",
        "W_se_plt_idx",
        "W_se_2d",
        "W_ee",
        "W_ee_ideal",
        "W_ee_plt_idx",
        "W_ee_2d",
        "W_ei",
        "W_ei_ideal",
        "W_ei_plt_idx",
        "W_ei_2d",
        "W_ie",
        "W_ie_ideal",
        "W_ie_plt_idx",
        "W_ie_2d",
    ]

    # Filter out the large arrays
    filtered_locs = {k: v for k, v in locs.items() if k not in exclude_keys}

    # Check if model exists
    for folder in os.listdir("model"):
        config_path = f"model/{folder}/config.npy"
        if os.path.exists(config_path):
            saved_config = np.load(config_path, allow_pickle=True).item()
            saved_config = {
                k: v for k, v in saved_config.items() if k not in exclude_keys
            }
            if filtered_locs == saved_config:
                stop_event = threading.Event()

                # Start the animation in a separate thread
                animation_thread = threading.Thread(
                    target=display_animation, args=(stop_event,)
                )
                animation_thread.start()

                # Load the model (this will run in the main thread)
                W_se_2d = np.load(f"model/{folder}/W_se.npy")
                W_ee_2d = np.load(f"model/{folder}/W_ee.npy")
                W_ie_2d = np.load(f"model/{folder}/W_ie.npy")
                W_ei_2d = np.load(f"model/{folder}/W_ei.npy")
                W_se_plt_idx = np.load(f"model/{folder}/W_se_plt_idx.npy")
                W_ee_plt_idx = np.load(f"model/{folder}/W_ee_plt_idx.npy")
                W_ie_plt_idx = np.load(f"model/{folder}/W_ie_plt_idx.npy")
                W_ei_plt_idx = np.load(f"model/{folder}/W_ei_plt_idx.npy")
                spikes = np.load(f"model/{folder}/spikes.npy")
                MemPot = np.load(f"model/{folder}/MemPot.npy")
                pre_synaptic_trace = np.load(f"model/{folder}/pre_synaptic_trace.npy")
                post_synaptic_trace = np.load(f"model/{folder}/post_synaptic_trace.npy")
                slow_pre_synaptic_trace = np.load(
                    f"model/{folder}/slow_pre_synaptic_trace.npy"
                )
                z_istdp = np.load(f"model/{folder}/z_istdp_trace.npy")
                I_in_ls = np.load(f"model/{folder}/I_in_ls.npy")
                V_th_array = np.load(f"model/{folder}/V_th_array.npy")

                # Signal that loading is complete
                stop_event.set()

                # Wait for the animation to finish
                animation_thread.join()

                print("Model reloaded successfully.       ")  # Clear the animation line

                return (
                    spikes,
                    MemPot,
                    W_se_2d,
                    W_se_ideal,
                    W_ee_2d,
                    W_ee_ideal,
                    W_ei_2d,
                    W_ei_ideal,
                    W_ie_2d,
                    W_ie_ideal,
                    pre_synaptic_trace,
                    post_synaptic_trace,
                    slow_pre_synaptic_trace,
                    z_istdp,
                    I_in_ls,
                    V_th_array,
                )

    # Initiate relevant arrays and variables
    num_neurons = N_excit_neurons + N_inhib_neurons + N_input_neurons
    spikes = np.zeros((time, num_neurons))
    pre_synaptic_trace = np.zeros((time, num_neurons))
    post_synaptic_trace = np.ones((time, N_excit_neurons))
    slow_pre_synaptic_trace = np.zeros((time, num_neurons))
    C = np.full(num_neurons, A)
    z_ht = np.ones(num_neurons)
    z_istdp = np.zeros(N_inhib_neurons)
    H = 0
    V_th_ = float(V_th_)
    V_th = np.full(num_neurons - N_input_neurons, V_th_)
    V_th_array = np.zeros((100))
    V_th_array[0] = V_th_

    # Add input data before training for input neurons
    spikes[:, :N_input_neurons] = training_data

    # Define update frequency for adaptive threshold
    update_freq = time // 100

    # Convert functions if njit is true
    njit_ = int(items > item_lim)

    if njit_:
        # adjust_membrane_threshold_func = njit(adjust_membrane_threshold)
        update_membrane_potential_func = njit(update_membrane_potential)
        exc_weight_update_func = njit(exc_weight_update)
        inh_weight_update_func = njit(inh_weight_update)
        print("Running njit")
    else:
        # adjust_membrane_threshold_func = adjust_membrane_threshold
        update_membrane_potential_func = update_membrane_potential
        exc_weight_update_func = exc_weight_update
        inh_weight_update_func = inh_weight_update
        print("Running without njit")

    # Initiate I_in list
    I_in_ls = np.zeros((time))

    # Loop through time and update membrane potential, spikes, and weights
    for t in tqdm(range(1, time), desc="Training network"):

        # Calculate Euler time unit
        euler_unit = int(t - update_freq > 0) * (t - update_freq)

        # Update adaptive membrane potential threshold
        if t % update_freq == 0:
            # V_th = adjust_membrane_threshold_func(
            #     spikes[t - 1],
            #     V_th,
            #     N_input_neurons,
            #     N_excit_neurons,
            #     N_inhib_neurons,
            #     dt,
            #     tau_thr,
            #     V_rest,
            # )

            V_th_array[t // update_freq] = np.mean(V_th)

        MemPot[t], I_in_i, I_in_e = update_membrane_potential_func(
            MemPot[t - 1],
            W_se,
            W_ee,
            W_ie,
            W_ei,
            spikes[t - 1],
            dt,
            N_excit_neurons,
            N_input_neurons,
            N_inhib_neurons,
            V_rest,
            R,
            tau_m,
        )
        I_in_ls[t] = np.mean(np.mean(I_in_e) + np.mean(I_in_i))

        # Update spikes based on membrane potential
        spike_mask = MemPot[t] > V_th
        spikes[t, N_input_neurons:] = spike_mask.astype(int)
        MemPot[t][spike_mask] = V_reset

        # Update excitatory weights
        (
            W_se,
            W_ee,
            W_se_ideal,
            W_ee_ideal,
            pre_synaptic_trace[t, :N_input_neurons],
            post_synaptic_trace[t],
            slow_pre_synaptic_trace[t, :N_input_neurons],
            z_ht[:N_input_neurons],
            C[:N_input_neurons],
            pre_synaptic_trace[t, N_input_neurons:-N_inhib_neurons],
            slow_pre_synaptic_trace[t, N_input_neurons:-N_inhib_neurons],
            z_ht[N_input_neurons:-N_inhib_neurons],
            C[N_input_neurons:-N_inhib_neurons],
        ) = exc_weight_update_func(
            dt,
            tau_const,
            W_se,
            W_ee,
            W_se_ideal,
            W_ee_ideal,
            P,
            t,
            w_p,
            spikes[t - 1],
            N_input_neurons,
            N_inhib_neurons,
            pre_synaptic_trace[t - 1],
            post_synaptic_trace[t - 1],
            slow_pre_synaptic_trace[euler_unit],
            tau_plus,
            tau_minus,
            tau_slow,
            tau_ht,
            tau_hom,
            A,
            beta,
            delta,
            update_freq,
            z_ht,
            C,
        )

        # Update inhibitory weights
        (
            W_ie,
            z_istdp,
            H,
        ) = inh_weight_update_func(
            H,
            dt,
            W_ie,
            z_istdp,
            tau_H,
            gamma,
            tau_stdp,
            learning_rate,
            spikes[t - 1, -N_inhib_neurons:],
            spikes[t - 1, N_input_neurons:-N_inhib_neurons],
            post_synaptic_trace[t],
        )

        # Assign the selected indices to the first row
        W_se_2d[t] = W_se[W_se_plt_idx[:, 0], W_se_plt_idx[:, 1]]
        W_ee_2d[t] = W_ee[W_ee_plt_idx[:, 0], W_ee_plt_idx[:, 1]]
        W_ie_2d[t] = W_ie[W_ie_plt_idx[:, 0], W_ie_plt_idx[:, 1]]

        # Ensure weights continue their value to the next time step
        W_ei_2d[t] = W_ei_2d[t - 1]

    if save_model:
        # Generate a random number for model folder
        rand_num = np.random.randint(0, 1000)

        # Check if random number folder exists
        while os.path.exists(f"model/model_{rand_num}"):
            rand_num = np.random.randint(0, 1000)

        os.makedirs(f"model/model_{rand_num}")

        # Save model weights, spikes, and MemPot
        np.save(f"model/model_{rand_num}/W_se.npy", W_se_2d)
        np.save(f"model/model_{rand_num}/W_ee.npy", W_ee_2d)
        np.save(f"model/model_{rand_num}/W_ie.npy", W_ie_2d)
        np.save(f"model/model_{rand_num}/W_ei.npy", W_ei_2d)
        np.save(f"model/model_{rand_num}/W_se_plt_idx.npy", W_se_plt_idx)
        np.save(f"model/model_{rand_num}/W_ee_plt_idx.npy", W_ee_plt_idx)
        np.save(f"model/model_{rand_num}/W_ie_plt_idx.npy", W_ie_plt_idx)
        np.save(f"model/model_{rand_num}/W_ei_plt_idx.npy", W_ei_plt_idx)
        np.save(f"model/model_{rand_num}/spikes.npy", spikes)
        np.save(f"model/model_{rand_num}/MemPot.npy", MemPot)
        np.save(f"model/model_{rand_num}/pre_synaptic_trace.npy", pre_synaptic_trace)
        np.save(f"model/model_{rand_num}/post_synaptic_trace.npy", post_synaptic_trace)
        np.save(
            f"model/model_{rand_num}/slow_pre_synaptic_trace.npy",
            slow_pre_synaptic_trace,
        )
        np.save(f"model/model_{rand_num}/z_istdp_trace.npy", z_istdp)
        np.save(f"model/model_{rand_num}/config.npy", filtered_locs)
        np.save(f"model/model_{rand_num}/I_in_ls.npy", I_in_ls)
        np.save(f"model/model_{rand_num}/V_th_array.npy", V_th_array)

    return (
        spikes,
        MemPot,
        W_se_2d,
        W_se_ideal,
        W_ee_2d,
        W_ee_ideal,
        W_ei_2d,
        W_ei_ideal,
        W_ie_2d,
        W_ie_ideal,
        pre_synaptic_trace,
        post_synaptic_trace,
        slow_pre_synaptic_trace,
        z_istdp,
        I_in_ls,
        V_th_array,
    )
