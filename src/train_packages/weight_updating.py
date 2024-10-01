import jax.numpy as jnp
from functools import partial
from jax import jit


@partial(
    jit,
    static_argnames=[
        "dt",
        "tau_cons",
        "P",
        "w_p",
        "N_inp",
        "N_inh",
        "tau_plus",
        "tau_minus",
        "tau_slow",
        "tau_ht",
        "tau_hom",
        "A",
        "beta",
        "delta",
    ],
)
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
    slow_post_trace,
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
    # Update ideal weights
    W_se_ideal = W_se_ideal + (dt / tau_cons) * (
        W_se
        - W_se_ideal
        - P * W_se_ideal * ((w_p / 2) - W_se_ideal) * (w_p - W_se_ideal)
    )

    # Update spike variables
    post_spikes = spikes[N_inp:-N_inh]
    pre_spikes_se = spikes[:N_inp]

    # Extract traces
    pre_trace_se = pre_synaptic_trace[:N_inp]
    z_ht_se = z_ht
    C_se = C

    # Update synaptic traces
    pre_trace_se = pre_trace_se + dt * ((-pre_trace_se / tau_plus) + pre_spikes_se)
    post_trace = post_trace + dt * ((-post_trace / tau_minus) + post_spikes_se)
    post_euler_trace = post_euler_trace + dt * ((-post_euler_trace / tau_minus) + post_spikes_se)
    slow_trace = slow_trace + dt * ((-slow_trace / tau_slow) + post_spikes_se)

    # Update z_ht, C, and B variables
    z_ht_se = z_ht_se + dt * (-z_ht_se / tau_ht + post_spikes_se)
    C_se = C_se + dt * (-C_se / tau_hom + z_ht_se**2)
    B = jnp.where(C_se <= 1 / A, A * C_se, A)

    # Get learning components
    triplet_LTP = A * pre_trace_se * slow_trace
    heterosynaptic = beta * post_euler_trace**3 * (W_se - W_se_ideal)
    transmitter = B * post_trace - delta

    # Compute the differential update for weights using Euler's method
    delta_w_se = dt * (
        post_spikes_se * (triplet_LTP - heterosynaptic) - pre_spikes_se * transmitter
    )

    # Update the weights
    W_se = W_se + delta_w_se
    W_se = jnp.clip(W_se, 0.0, 5.0)

    ## W_ee weights ##

    # Update ideal weights
    W_ee_ideal = W_ee_ideal + (dt / tau_cons) * (
        W_ee
        - W_ee_ideal
        - P * W_ee_ideal * ((w_p / 2) - W_ee_ideal) * (w_p - W_ee_ideal)
    )

    # Update spike variables
    pre_spikes_ee = spikes[N_inp:-N_inh]

    # Extract traces
    pre_trace_ee = pre_synaptic_trace[N_inp:-N_inh]

    # Update synaptic traces
    pre_trace_ee = pre_trace_ee + dt * (-pre_trace_ee / tau_plus + pre_spikes_ee)

    # Get learning components
    triplet_LTP_ee = A * pre_trace_ee * slow_trace
    heterosynaptic_ee = beta * post_trace**3 * (W_ee - W_ee_ideal)
    transmitter_ee = B_ee * post_trace - delta

    # Compute the differential update for weights using Euler's method
    delta_w_ee = dt * (
        post_spikes * (triplet_LTP_ee - heterosynaptic_ee) - pre_spikes_ee * transmitter_ee
    )

    # Update the weights
    W_ee = W_ee + delta_w_ee
    W_ee = jnp.clip(W_ee, 0.0, 5.0)

    return (
        W_se,
        W_ee,
        W_se_ideal,
        W_ee_ideal,
        pre_trace_se,
        post_trace_se,
        slow_trace_se,
        z_ht_se,
        C_se,
        pre_trace_ee,
    )


@jit
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
    # Update synaptic traces using Euler's method
    z_i = z_i + dt * (-z_i / tau_stdp + post_spikes)
    z_j = z_j + dt * (-z_j / tau_stdp + pre_spikes)

    # Update H using Euler's method
    H = H + dt * (-H / tau_H + jnp.sum(post_spikes))
    G = H - gamma

    # Reshape arrays for matrix operations
    z_i_reshaped = jnp.expand_dims(z_i, axis=1)
    z_j_reshaped = jnp.expand_dims(z_j, axis=1)
    post_spikes_reshaped = jnp.expand_dims(post_spikes, axis=1)
    pre_spikes_reshaped = jnp.expand_dims(pre_spikes, axis=1)

    # Calculate delta weights
    delta_w = (
        dt
        * learning_rate
        * G
        * (
            jnp.dot(pre_spikes_reshaped, (z_i_reshaped.T + 1))
            + jnp.dot(z_j_reshaped, post_spikes_reshaped.T)
        )
    )

    # Update weights with constraints
    W_inh = W_inh + delta_w
    W_inh = jnp.clip(W_inh, 0.0, 5.0)

    return W_inh, z_i, z_j, H
