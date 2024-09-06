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
    "C:\\Users\\Bruker\\OneDrive\\Documents\\NMBU_\\BONSAI\\SNN\\SpikingNeuralNetwork\\src"
):
    base_path = "C:\\Users\\Bruker\\OneDrive\\Documents\\NMBU_\\BONSAI\\SNN\\SpikingNeuralNetwork\\src"
else:
    base_path = "C:\\Users\\andreama\\OneDrive - Norwegian University of Life Sciences\\Documents\\Projects\\BONXAI\\SpikingNeuralNetwork\\src"

os.chdir(base_path)
sys.path.append(os.path.join(base_path, "main"))
sys.path.append(os.path.join(base_path, "plot"))
sys.path.append(os.path.join(base_path, "tool"))
sys.path.append(os.path.join(base_path, "train_packages"))

from plot_training import *
from plot_network import *
from membrane_potential import update_membrane_potential_conduct
from weight_updating import exc_weight_update, inh_weight_update


def load_model(folder, stop_event):
    W_exc_2d = np.load(f"model/{folder}/W_exc.npy")
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
        W_exc_2d,
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
        print(f"Found model. Reloading {waiting_animation[idx]}", end="\r")
        idx = (idx + 1) % len(waiting_animation)
        time.sleep(0.2)  # Adjust the speed of the animation as needed


def train_data(
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
    tau_cons: float | int,
    tau_H: float | int,
    tau_istdp: float | int,
    tau_ampa: float | int,
    tau_nmda: float | int,
    tau_gaba: float | int,
    tau_thr: float | int,
    tau_d: float | int,
    tau_f: float | int,
    tau_a: float | int,
    tau_b: float | int,
    delta_a: float | int,
    delta_b: float | int,
    U_exc: float | int,
    U_inh: float | int,
    learning_rate: float | int,
    training_data: np.ndarray,
    N_excit_neurons: int,
    N_inhib_neurons: int,
    N_input_neurons: int,
    MemPot: np.ndarray,
    max_weight: float | int,
    min_weight: float | int,
    W_exc: np.ndarray,
    W_inh: np.ndarray,
    W_exc_ideal: np.ndarray,
    W_exc_2d: np.ndarray,
    W_exc_plt_idx: np.ndarray,
    gamma: float | int,  # Where is gamma used?
    alpha_exc: float | int,
    alpha_inh: float | int,
    save_model: bool,
    U_cons: float | int,
    force_retrain: bool,
):
    # Get all local variables
    locs = locals()

    # Define keys to exclude
    exclude_keys = [
        "training_data",
        "MemPot",
        "W_exc",
        "W_inh",
        "W_exc_ideal",
        "W_exc_plt_idx",
        "W_exc_2d",
        "save_model",
        "locs",
    ]

    # Filter out the large arrays
    filtered_locs = {k: v for k, v in locs.items() if k not in exclude_keys}

    if not force_retrain:
        # Check if model exists
        for folder in os.listdir("model"):
            config_path = f"model/{folder}/config.npy"
            if os.path.exists(config_path):
                saved_config = np.load(config_path, allow_pickle=True).item()
                saved_config = {
                    k: v for k, v in saved_config.items() if k not in exclude_keys
                }
                print("saved config: ", saved_config, "filtered config", filtered_locs)
                d = input("Should we continue?")
                if d == "n":
                    print("Training cancelled. Errors detected")
                    break
                if filtered_locs == saved_config:
                    stop_event = threading.Event()

                    # Start the animation in a separate thread
                    animation_thread = threading.Thread(
                        target=display_animation, args=(stop_event,)
                    )
                    animation_thread.start()

                    # Create dict of variable names and values

                    # Load the model (this will run in the main thread)
                    save_path = f"model/model_{folder}"

                    # Now you can access the variables like this:
                    W_exc_2d = np.load(save_path + "/W_exc_2d")
                    spikes = np.load(save_path + "/spikes")
                    MemPot = np.load(save_path + "/MemPot")
                    pre_synaptic_trace = np.load(save_path + "/pre_synaptic_trace")
                    post_synaptic_trace = np.load(save_path + "/post_synaptic_trace")
                    slow_pre_synaptic_trace = np.load(
                        save_path + "/slow_pre_synaptic_trace"
                    )
                    C = np.load(save_path + "/C")
                    z_ht = np.load(save_path + "/z_ht")
                    x = np.load(save_path + "/x")
                    u = np.load(save_path + "/u")
                    H = np.load(save_path + "/H")
                    z_i = np.load(save_path + "/z_i")
                    z_j = np.load(save_path + "/z_j")
                    filtered_locs = np.load(save_path + "/config")
                    V_th_array = np.load(save_path + "/V_th_array")
                    W_exc = np.load(save_path + "/W_exc")
                    W_inh = np.load(save_path + "/W_inh")
                    V_th = np.load(save_path + "/V_th")
                    g_nmda = np.load(save_path + "/g_nmda")
                    g_ampa = np.load(save_path + "/g_ampa")
                    g_gaba = np.load(save_path + "/g_gaba")
                    g_a = np.load(save_path + "/g_a")
                    g_b = np.load(save_path + "/g_b")

                    # Signal that loading is complete
                    stop_event.set()

                    # Wait for the animation to finish
                    animation_thread.join()

                    print(
                        "Model reloaded successfully.       "
                    )  # Clear the animation line

                    return (
                        W_exc_2d,
                        spikes,
                        MemPot,
                        post_synaptic_trace,
                        slow_pre_synaptic_trace,
                        C,
                        z_ht,
                        x,
                        u,
                        H,
                        z_i,
                        z_j,
                        filtered_locs,
                        V_th_array,
                        W_exc,
                        W_inh,
                        V_th,
                        g_nmda,
                        g_ampa,
                        g_gaba,
                        g_a,
                        g_b,
                    )

    # Initiate relevant traces and variables
    num_neurons = N_excit_neurons + N_inhib_neurons + N_input_neurons
    spikes = np.zeros((time, num_neurons))
    pre_synaptic_trace = np.zeros((time, num_neurons))
    post_synaptic_trace = np.zeros((time, N_excit_neurons))
    slow_pre_synaptic_trace = np.zeros((time, num_neurons))
    C = np.full(num_neurons, A)
    z_i = np.zeros(N_excit_neurons)
    z_j = np.zeros(N_inhib_neurons)
    z_ht = np.ones(num_neurons)
    x = np.ones((N_input_neurons + N_excit_neurons, 1))
    u = np.ones((N_input_neurons + N_excit_neurons, 1))
    H = 0
    V_th_ = float(V_th_)
    V_th = np.full(num_neurons - N_input_neurons, V_th_)
    V_th_array = np.zeros((100))  # for plotting
    V_th_array[0] = V_th_
    g_nmda = np.zeros((num_neurons - N_input_neurons, 1))
    g_ampa = np.zeros((num_neurons - N_input_neurons, 1))
    g_gaba = np.zeros((N_excit_neurons, 1))
    g_a = np.zeros((N_excit_neurons, 1))
    g_b = np.zeros((N_excit_neurons, 1))

    # Add input data before training for input neurons
    spikes[:, :N_input_neurons] = training_data

    # Define update frequency for adaptive threshold
    update_freq = time // 100

    # Convert functions if njit is true
    njit_ = False

    if njit_:
        update_membrane_potential_conduct_func = njit(update_membrane_potential_conduct)
        exc_weight_update_func = njit(exc_weight_update)
        inh_weight_update_func = njit(inh_weight_update)
        print("Running njit")
    else:
        update_membrane_potential_conduct_func = update_membrane_potential_conduct
        exc_weight_update_func = exc_weight_update
        inh_weight_update_func = inh_weight_update
        print("Running without njit")
    up = time // 100

    # Loop through time and update membrane potential, spikes, and weights
    for t in tqdm(range(1, time), desc="Training network"):

        # Calculate Euler time unit
        euler_unit = int(t - update_freq > 0) * (t - update_freq)

        # Update membrane potential
        MemPot[t], V_th, g_ampa, g_nmda, g_gaba, x, u, g_a, g_b = (
            update_membrane_potential_conduct_func(
                MemPot[t - 1],
                U_inh,
                U_exc,
                V_th,
                V_th_,
                W_exc,
                W_inh,
                spikes[t - 1],
                u,
                x,
                dt,
                N_input_neurons,
                N_inhib_neurons,
                V_rest,
                tau_m,
                alpha_exc,
                alpha_inh,
                tau_ampa,
                tau_nmda,
                tau_gaba,
                tau_thr,
                tau_d,
                tau_f,
                tau_a,
                tau_b,
                delta_a,
                delta_b,
                g_ampa,
                g_nmda,
                g_gaba,
                g_a,
                g_b,
                U_cons,
            )
        )

        # Update spikes based on membrane potential
        spike_mask = MemPot[t] > V_th
        spikes[t, N_input_neurons:] = spike_mask.astype(int)
        MemPot[t][spike_mask] = V_reset

        # Update excitatory weights
        (
            W_exc[:N_input_neurons],
            W_exc[N_input_neurons:-N_inhib_neurons],
            W_exc_ideal[:N_input_neurons],
            W_exc_ideal[N_input_neurons:-N_inhib_neurons],
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
            tau_cons,
            W_exc[:N_input_neurons],
            W_exc[N_input_neurons:-N_inhib_neurons],
            W_exc_ideal[:N_input_neurons],
            W_exc_ideal[N_input_neurons:-N_inhib_neurons],
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
            W_inh,
            z_i,
            z_j,
            H,
        ) = inh_weight_update_func(
            H,
            dt,
            W_inh,
            z_i,
            z_j,
            tau_H,
            gamma,
            tau_istdp,
            learning_rate,
            spikes[t - 1, -N_inhib_neurons:],
            spikes[t - 1, N_input_neurons:-N_inhib_neurons],
        )

        # Assign the selected indices to the first row
        W_exc_2d[t] = W_exc[W_exc_plt_idx[:, 0], W_exc_plt_idx[:, 1]]

    if save_model:
        # Generate a random number for model folder
        rand_num = np.random.randint(0, 1000)

        # Check if random number folder exists
        while os.path.exists(f"model/model_{rand_num}"):
            rand_num = np.random.randint(0, 1000)

        os.makedirs(f"model/model_{rand_num}")

        # Save main path
        save_path = f"model/model_{rand_num}"

        # Create a dictionary of file names and variables
        data_to_save = {
            "W_exc_2d": W_exc_2d,
            "spikes": spikes,
            "MemPot": MemPot,
            "pre_synaptic_trace": pre_synaptic_trace,
            "post_synaptic_trace": post_synaptic_trace,
            "slow_pre_synaptic_trace": slow_pre_synaptic_trace,
            "C": C,
            "z_ht": z_ht,
            "x": x,
            "u": u,
            "H": H,
            "z_i": z_i,
            "z_j": z_j,
            "config": filtered_locs,
            "V_th_array": V_th_array,
            "exc_weights": W_exc,
            "inh_weights": W_inh,
            "V_th": V_th,
            "g_nmda": g_nmda,
            "g_ampa": g_ampa,
            "g_gaba": g_gaba,
            "g_a": g_a,
            "g_b": g_b,
        }

        # Loop through the dictionary and save each variable
        for filename, data in data_to_save.items():
            np.save(f"{save_path}/{filename}.npy", data)

    return (
        W_exc_2d,
        spikes,
        MemPot,
        post_synaptic_trace,
        slow_pre_synaptic_trace,
        C,
        z_ht,
        x,
        u,
        H,
        z_i,
        z_j,
        filtered_locs,
        V_th_array,
        W_exc,
        W_inh,
        V_th,
        g_nmda,
        g_ampa,
        g_gaba,
        g_a,
        g_b,
    )
