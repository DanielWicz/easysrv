import unittest
from easysrv import SRV
import tensorflow as tf
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestSVRCreation(unittest.TestCase):
    def test_create(self):
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
                tf.keras.layers.Dense(
                    2, kernel_regularizer=tf.keras.regularizers.L2(0.0001)
                ),
            ]
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model(tf.random.normal(shape=(20, 40)))
        srv = SRV(model=model, optimizer=optimizer, epochs=5, lagtime=1, split=0.1)


if __name__ == "__main__":
    unittest.main()
