import numpy as np


def update_weights(weights, S): ...


def update_membrane_potential(
    mp,
    weights,
    membrane,
    resting_potential,
    membrane_resistance,
    membrane_conductance,
    dt,
):
    I_in = np.dot(weights, mp)
    U_delta = (
        (I_in - ((membrane - resting_potential) / membrane_resistance))
        * dt
        / membrane_conductance
    )
    membrane += U_delta

    return


def train_network(self, weights, mp, spikes, S_traces):

    for t in range(1, self.T):
        # update membrane potential

        ...
        # update weights
