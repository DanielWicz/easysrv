from scipy.stats import multinomial
from typing import List
from scipy.stats import norm
import tensorflow as tf
import numpy as np

import unittest
from easysrv import SRV

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class SampleMSM:
    def __init__(self):
        self.p_init = np.array([0, 1, 0])
        self.p_transition = np.array(
            [
                [0.90, 0.05, 0.02, 0.03],
                [0.01, 0.90, 0.07, 0.02],
                [0.05, 0.03, 0.9, 0.02],
                [0.10, 0.03, 0.02, 0.85],
            ]
        )

    def markov_sequence(self, sequence_length=10000):
        """
        Generate a Markov sequence based on p_init and p_transition.
        """
        p_init = self.p_init
        p_transition = self.p_transition

        if p_init is None:
            p_init = equilibrium_distribution(p_transition)
        initial_state = list(multinomial.rvs(1, p_init)).index(1)

        states = [initial_state]
        for _ in range(sequence_length - 1):
            p_tr = p_transition[states[-1]]
            new_state = list(multinomial.rvs(1, p_tr)).index(1)
            states.append(new_state)
        return states

    def gaussian_emissions(self, states, mus, sigmas):
        emissions = []
        for state in states:
            loc = mus[state]
            scale = sigmas[state]
            e = norm.rvs(loc=loc, scale=scale)
            emissions.append(e)
        return np.array(emissions)

    def sample(self, n=10000, dims=2):
        dims = dims
        sep = 50
        mus = np.random.randn(4, dims) * sep
        sigmas = np.abs(np.random.randn(4, dims)) * sep
        gaussian_ems = self.gaussian_emissions(
            self.markov_sequence(sequence_length=n), mus=mus, sigmas=sigmas
        )
        return gaussian_ems


msm = SampleMSM()
sample_n = 10000
num = 40
data = [msm.sample(n=sample_n, dims=10) for x in range(num)]
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(
            16,
            kernel_regularizer=tf.keras.regularizers.L2(0.0001),
            kernel_initializer="lecun_normal",
        ),
        tf.keras.layers.Activation("swish"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(
            16,
            kernel_regularizer=tf.keras.regularizers.L2(0.0001),
            kernel_initializer="lecun_normal",
        ),
        tf.keras.layers.Activation("swish"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(5, kernel_regularizer=tf.keras.regularizers.L2(0.0001)),
    ]
)
model(data[0])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
srv = SRV(model=model, optimizer=optimizer, epochs=5, lagtime=1, split=0.1)
srv.fit(data, verbose=False)


class TestFitting(unittest.TestCase):
    def test_remove_one_mode(self):
        out = srv.transform(data, remove_modes=[0])
        if out[0].shape[0] != sample_n:
            raise Exception(
                "Sample output dimension after trasform is different than assumed in the test"
            )
        if out[0].shape[-1] != 5 - 1:
            raise Exception(
                "Last output dimension after transform is different than assumed in the test"
            )
        if len(out) != num:
            raise Exception(
                "Number of samples after transform is different than assumed in the test"
            )

    def test_remove_two_subsequent_modes(self):
        out = srv.transform(data, remove_modes=[0, 1])
        if out[0].shape[0] != sample_n:
            raise Exception(
                "Sample output dimension after trasform is different than assumed in the test"
            )
        if out[0].shape[-1] != 5 - 2:
            raise Exception(
                "Last output dimension after transform is different than assumed in the test"
            )
        if len(out) != num:
            raise Exception(
                "Number of samples after transform is different than assumed in the test"
            )

    def test_remove_two_notsubsequent_modes(self):
        out = srv.transform(data, remove_modes=[1, 3])
        if out[0].shape[0] != sample_n:
            raise Exception(
                "Sample output dimension after trasform is different than assumed in the test"
            )
        if out[0].shape[-1] != 5 - 2:
            raise Exception(
                "Last output dimension after transform is different than assumed in the test"
            )
        if len(out) != num:
            raise Exception(
                "Number of samples after transform is different than assumed in the test"
            )

    def test_remove_all_except_last_mode(self):
        out = srv.transform(data, remove_modes=[0, 1, 2, 3])
        if out[0].shape[0] != sample_n:
            raise Exception(
                "Sample output dimension after trasform is different than assumed in the test"
            )
        if out[0].shape[-1] != 5 - 4:
            raise Exception(
                "Last output dimension after transform is different than assumed in the test"
            )
        if len(out) != num:
            raise Exception(
                "Number of samples after transform is different than assumed in the test"
            )


if __name__ == "__main__":
    unittest.main()
