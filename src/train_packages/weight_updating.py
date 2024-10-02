import numpy as np


def exc_weight_update(
    dt,
    tau_cons,
    W_se,
    W_ee,
    W_se_ideal,
    W_ee_ideal,
    P,
    w_p,
    spikes,
    N_inp,
    N_inh,
    pre_trace,
    post_trace,
    post_euler_trace,
    slow_trace,
    tau_plus,
    tau_minus,
    tau_slow,
    tau_ht,
    tau_hom,
    A,
    beta,
    delta,
    z_ht,
    C,
):
    N_total = len(spikes)
    N_post = N_total - N_inp - N_inh

    # Update ideal weights W_se_ideal
    for i in range(N_post):
        for j in range(N_inp):
            diff = W_se[i, j] - W_se_ideal[i, j]
            cons_term = (
                P
                * W_se_ideal[i, j]
                * ((w_p / 2) - W_se_ideal[i, j])
                * (w_p - W_se_ideal[i, j])
            )
            W_se_ideal[i, j] += (dt / tau_cons) * (diff - cons_term)

    # Update ideal weights W_ee_ideal
    for i in range(N_post):
        for j in range(N_post):
            diff = W_ee[i, j] - W_ee_ideal[i, j]
            cons_term = (
                P
                * W_ee_ideal[i, j]
                * ((w_p / 2) - W_ee_ideal[i, j])
                * (w_p - W_ee_ideal[i, j])
            )
            W_ee_ideal[i, j] += (dt / tau_cons) * (diff - cons_term)

    # Update spike variables
    post_spikes = spikes[N_inp : N_inp + N_post]  # Size N_post
    pre_spikes_se = spikes[:N_inp]  # Size N_inp
    pre_spikes_ee = post_spikes  # Since spikes[N_inp:N_inp + N_post]

    # Update synaptic traces
    for i in range(N_inp):
        pre_trace[i] += dt * ((-pre_trace[i] / tau_plus) + pre_spikes_se[i])

    for i in range(N_post):
        idx = N_inp + i
        pre_trace[idx] += dt * ((-pre_trace[idx] / tau_plus) + pre_spikes_ee[i])

    for i in range(N_post):
        post_trace[i] += dt * ((-post_trace[i] / tau_minus) + post_spikes[i])
        slow_trace[i] += dt * ((-slow_trace[i] / tau_slow) + post_spikes[i])

    # Update z_ht and C
    for i in range(N_post):
        z_ht[i] += dt * (-z_ht[i] / tau_ht + post_spikes[i])
        C[i] += dt * (-C[i] / tau_hom + z_ht[i] ** 2)

    # Compute B
    B = np.empty(N_post)
    for i in range(N_post):
        if C[i] <= 1.0:
            B[i] = A * C[i]
        else:
            B[i] = A

    # Compute triplet_LTP and heterosynaptic
    triplet_LTP = np.zeros((N_post, N_inp))
    heterosynaptic = np.zeros((N_post, N_inp))
    for i in range(N_post):
        for j in range(N_inp):
            pre_idx = j
            triplet_LTP[i, j] = A * pre_trace[pre_idx] * slow_trace[i]
            heterosynaptic[i, j] = (
                beta * post_euler_trace[i] ** 3 * (W_se[i, j] - W_se_ideal[i, j])
            )

    # Compute transmitter
    transmitter = np.empty(N_post)
    for i in range(N_post):
        transmitter[i] = B[i] * post_trace[i] - delta

    # Compute delta_w_se and update W_se
    for i in range(N_post):
        for j in range(N_inp):
            delta_w = dt * (
                post_spikes[i] * (triplet_LTP[i, j] - heterosynaptic[i, j])
                - pre_spikes_se[j] * transmitter[i]
            )
            W_se[i, j] += delta_w
            if W_se[i, j] < 0.0:
                W_se[i, j] = 0.0
            elif W_se[i, j] > 5.0:
                W_se[i, j] = 5.0

    # Compute triplet_LTP_ee and heterosynaptic_ee
    triplet_LTP_ee = np.zeros((N_post, N_post))
    heterosynaptic_ee = np.zeros((N_post, N_post))
    for i in range(N_post):
        for j in range(N_post):
            pre_idx = N_inp + j
            triplet_LTP_ee[i, j] = A * pre_trace[pre_idx] * slow_trace[i]
            heterosynaptic_ee[i, j] = (
                beta * post_trace[i] ** 3 * (W_ee[i, j] - W_ee_ideal[i, j])
            )

    # Compute delta_w_ee and update W_ee
    for i in range(N_post):
        for j in range(N_post):
            delta_w = dt * (
                post_spikes[i] * (triplet_LTP_ee[i, j] - heterosynaptic_ee[i, j])
                - pre_spikes_ee[j] * transmitter[i]
            )
            W_ee[i, j] += delta_w
            if W_ee[i, j] < 0.0:
                W_ee[i, j] = 0.0
            elif W_ee[i, j] > 5.0:
                W_ee[i, j] = 5.0

    return (
        W_se,
        W_ee,
        W_se_ideal,
        W_ee_ideal,
        pre_trace[:N_inp],
        post_trace,
        slow_trace,
        z_ht,
        C,
        pre_trace[N_inp:],
    )


def inh_weight_update(
    H,
    dt,
    W_inh,
    z_i,
    z_j,
    tau_H,
    gamma,
    tau_stdp,
    learning_rate,
    pre_spikes,
    post_spikes,
):
    N_post = len(post_spikes)

    # Update synaptic traces using Euler's method
    for i in range(N_post):
        z_i[i] += dt * (-z_i[i] / tau_stdp + post_spikes[i])
        z_j[i] += dt * (-z_j[i] / tau_stdp + pre_spikes[i])

    # Update H using Euler's method
    H += dt * (-H / tau_H + np.sum(post_spikes))
    G = H - gamma

    # Calculate delta weights
    delta_w = np.zeros((N_post, N_post))
    for i in range(N_post):
        for j in range(N_post):
            delta_w[i, j] = (
                dt
                * learning_rate
                * G
                * (pre_spikes[i] * (z_i[j] + 1.0) + z_j[i] * post_spikes[j])
            )

    # Update weights with constraints
    for i in range(N_post):
        for j in range(N_post):
            W_inh[i, j] += delta_w[i, j]
            if W_inh[i, j] < 0.0:
                W_inh[i, j] = 0.0
            elif W_inh[i, j] > 5.0:
                W_inh[i, j] = 5.0

    return W_inh, z_i, z_j, H
