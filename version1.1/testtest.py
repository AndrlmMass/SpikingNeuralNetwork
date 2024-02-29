import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
dt = 0.1  # timestep in ms
time = np.arange(0, 100, dt)  # simulation time from 0 to 100 ms

# Neuron parameters
V_rest = -65.0  # Resting membrane potential in mV
V_th = -50.0  # Threshold potential in mV
V_reset = -70.0  # Reset potential in mV

# Synapse parameters
tau_ampa = 2.0  # AMPA decay time constant in ms
tau_nmda = 100.0  # NMDA decay time constant in ms
tau_gaba = 10.0  # GABA decay time constant in ms
g_ampa_max = 0.05  # Max conductance for AMPA
g_nmda_max = 0.05  # Max conductance for NMDA
g_gaba_max = 0.05  # Max conductance for GABA
E_ampa = 0.0  # Reversal potential for AMPA in mV
E_nmda = 0.0  # Reversal potential for NMDA in mV
E_gaba = -70.0  # Reversal potential for GABA in mV

# Initializations
V = np.full(time.shape, V_rest)  # Membrane potential array
g_ampa = np.zeros(time.shape)  # Conductance for AMPA
g_nmda = np.zeros(time.shape)  # Conductance for NMDA
g_gaba = np.zeros(time.shape)  # Conductance for GABA

# Simulate synaptic inputs
input_times = np.array([10, 20, 50, 70])  # Times at which inputs are received
for input_time in input_times:
    input_index = int(input_time / dt)
    g_ampa[input_index] += g_ampa_max  # Simulate an AMPA receptor activation
    g_nmda[input_index] += g_nmda_max  # Simulate an NMDA receptor activation
    g_gaba[input_index] += g_gaba_max  # Simulate a GABA receptor activation

# Simulation loop
for i in range(1, len(time)):
    # Update conductances
    g_ampa[i] -= g_ampa[i - 1] * dt / tau_ampa
    g_nmda[i] -= g_nmda[i - 1] * dt / tau_nmda
    g_gaba[i] -= g_gaba[i - 1] * dt / tau_gaba

    # Calculate the postsynaptic current (PSC)
    I_ampa = g_ampa[i] * (E_ampa - V[i - 1])
    I_nmda = g_nmda[i] * (E_nmda - V[i - 1])
    I_gaba = g_gaba[i] * (E_gaba - V[i - 1])
    I_total = I_ampa + I_nmda + I_gaba

    # Update membrane potential
    V[i] = V[i - 1] + I_total * dt
    if V[i] >= V_th:
        V[i] = V_reset  # Reset potential after spike

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.title("Membrane Potential")
plt.plot(time, V, label="Membrane Potential")
plt.axhline(y=V_th, color="r", linestyle="--", label="Threshold")
plt.xlabel("Time (ms)")
plt.ylabel("Potential (mV)")
plt.legend()

plt.subplot(2, 1, 2)
plt.title("Synaptic Conductances")
plt.plot(time, g_ampa, label="AMPA")
plt.plot(time, g_nmda, label="NMDA")
plt.plot(time, g_gaba, label="GABA")
plt.xlabel("Time (ms)")
plt.ylabel("Conductance")
plt.legend()

plt.tight_layout()
plt.show()
