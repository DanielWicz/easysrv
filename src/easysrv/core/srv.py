import tensorflow as tf
import numpy as np
import scipy
from easysrv.utils import calc_cov, metric_VAMP2


class SRV:
    """SRV implementation from:
    Arguments:
        model: Tensorflow Sequential Model, where the number of outputs signifies the number of slow processes.
        optimizer: Tensorflow optimizer object.
        epochs: Number of epochs used to train.
        lagtime (small tau): lagtime used for training of the SRV.
        split: split between training and validation set.
    """

    def __init__(self, model=None, optimizer=None, epochs=None, lagtime=1, split=0.1):
        self.model = model
        self.training_data = None
        self.split = split
        self.optimizer = optimizer
        self.epochs = epochs
        self.ae_lagtime = lagtime

        # eigenvectors and their stats
        self.eigenvectors_ = None
        self.eigenvalues_ = None
        self.norms_ = None
        self.means_ = None

        self.dense_out = model.layers[-1].output_shape[-1]
        if self.dense_out < 2:
            raise ValueError("Output/num of slow processes has to be 2 or more")
        # stats

        self.train_loss = []
        self.train_val_loss = []
        self.validation_vamp = []

    def fit(self, data, verbose=True):
        """Fits SRV to given data
        Arguments:
            data: A list of features extracted from trajectories as numpy arrays.
                  Also a generator that returns a numpy array will work.

        Returns:
            A dictionary with history of eigenvalues, training loss and validation loss.
        """
        split_size = int(len(data) * self.split)
        self.validation_data = data[:split_size]
        self.training_data = data[split_size:]
        self.train(verbose=verbose)
        self._calc_basis()
        history = {
            "eigenvalues": list(self.eigenvalues_.numpy()),
            "VAMP2 valid score": self.validation_vamp,
            "Training loss": self.train_loss,
            "Val Training loss": self.train_val_loss,
        }
        return history

    def transform(self, data, remove_modes=[]):
        """Transforms data onto SRV slow processes. Order is the same as in the data.
        Arguments:
            data: A list of features extracted from trajectories as numpy arrays.
                  Also a generator that returns a numpy array will work.
            remove_modes: Remove modes indexed from 0 to n, where 0 is the slowest.
                          For example you can specify [0, 2] to remove slowest and third slowest.
        Returns:
            Transformed data onto slow processes as a list of numpy arrays
        """

        all_data = []
        for datapoint in data:
            z = self.model(datapoint) - self.means_
            eigenvectors = self.eigenvectors_
            # remove some eigenvectors
            all_eig_ind = [
                i for i in range(eigenvectors.shape[0]) if i not in remove_modes
            ]
            eigv = tf.transpose(tf.gather(tf.transpose(eigenvectors), all_eig_ind))
            z_product = tf.matmul(z, eigv)
            norms = tf.transpose(tf.gather(tf.transpose(self.norms_), all_eig_ind))
            z_norm = z_product / norms
            all_data.append(z_norm.numpy())

        return all_data

    def fit_transform(self, data, verbose=True, remove_modes=[]):
        """Fits SRV to given data and transforms them onto slow processes, order is the same as in the data.
        Arguments:
            data: A list of features extracted from trajectories as numpy arrays.
                  Also a generator that returns a numpy array will work.
            remove_modes: Remove modes indexed from 0 to n, where 0 is the slowest.
                          For example you can specify [0, 2] to remove slowest and third slowest.
        Returns:
            Transformed data onto slow processes as a list of numpy arrays
        """

        self.fit(data, verbose=verbose)
        return self.transform(data, remove_modes=remove_modes)

    def train(self, verbose=True):
        """Trains SRV for a given model.
        Arguments:
            None
        Returns:
            None
        """
        for i in range(self.epochs):
            loss = []
            val_loss = []
            vamp2_met = []
            for batch in self.training_data:
                batch_shift = tf.convert_to_tensor(batch[self.ae_lagtime :])
                batch_back = tf.convert_to_tensor(batch[: -self.ae_lagtime])
                batch_ae_loss, g_norm_ae = self._train_step_vamp(
                    batch_shift,
                    batch_back,
                )
                loss.append(batch_ae_loss)
            for batch in self.validation_data:
                batch_shift = self.model(tf.convert_to_tensor(batch[self.ae_lagtime :]))
                batch_back = self.model(tf.convert_to_tensor(batch[: -self.ae_lagtime]))
                vamp2_score = metric_VAMP2(batch_back, batch_shift)
                batch_val_loss = self.loss_func_vamp(batch_back, batch_shift)
                vamp2_met.append(vamp2_score)
                val_loss.append(batch_val_loss)
            mean_loss = float(np.mean(loss))
            mean_val_loss = float(np.mean(val_loss))
            mean_vamp2 = float(np.mean(vamp2_met))
            self.train_loss.append(mean_loss)
            self.train_val_loss.append(mean_val_loss)
            self.validation_vamp.append(mean_vamp2)
            if verbose:
                print("Loss for epoch {0} is {1}".format(i, mean_loss))
                print("Val Loss for epoch {0} is {1}".format(i, mean_val_loss))
                print("Validation VAMP2 for epoch {0} is {1}".format(i, mean_vamp2))

        self._calc_basis()

    @tf.function
    def _calc_basis_estimate(self, batch_shift, batch_back):
        zt0 = self.model(batch_back)
        ztt = self.model(batch_shift)
        ztt_nom = tf.cast(ztt, tf.float64)
        zt0_nom = tf.cast(zt0, tf.float64)
        ztt = ztt_nom - tf.reduce_mean(ztt_nom, axis=0)
        zt0 = zt0_nom - tf.reduce_mean(zt0_nom, axis=0)
        # weights = self._koopman_weight(zt0, ztt)
        cov_01 = calc_cov(zt0, ztt)
        cov_10 = calc_cov(ztt, zt0)
        cov_11 = calc_cov(ztt, ztt)
        cov_00 = calc_cov(zt0, zt0)

        return cov_01, cov_10, cov_11, cov_00, zt0_nom, ztt_nom

    def _calc_basis(self):
        """Calculates basis vectors for the SRV method, based on the average
        approximated covariance matrices.
        Arguments:
            None
        Returns:
            None
        """
        M = self.dense_out
        cov_01 = tf.zeros(shape=(M, M), dtype=tf.float64)
        cov_10 = tf.zeros(shape=(M, M), dtype=tf.float64)
        cov_11 = tf.zeros(shape=(M, M), dtype=tf.float64)
        cov_00 = tf.zeros(shape=(M, M), dtype=tf.float64)
        zt0_buffer = []
        ztt_buffer = []
        obs = 0
        for batch in self.training_data:
            batch_shift = tf.convert_to_tensor(batch[self.ae_lagtime :])
            batch_back = tf.convert_to_tensor(batch[: -self.ae_lagtime])
            assert tf.rank(batch_back) == 2
            assert tf.rank(batch_shift) == 2
            (
                cov_01e,
                cov_10e,
                cov_11e,
                cov_00e,
                zt0_nom,
                ztt_nom,
            ) = self._calc_basis_estimate(batch_shift, batch_back)
            zt0_buffer.append(tf.cast(zt0_nom, tf.float32))
            ztt_buffer.append(tf.cast(ztt_nom, tf.float32))
            # weights = self._koopman_weight(zt0, ztt)
            cov_01 += cov_01e
            cov_10 += cov_10e
            cov_11 += cov_11e
            cov_00 += cov_00e
            obs += 1

        zt0_concat = tf.concat(zt0_buffer, axis=0)
        ztt_concat = tf.concat(ztt_buffer, axis=0)
        x_concat = tf.concat([ztt_concat, zt0_concat], axis=0)
        self.means_ = tf.reduce_mean(x_concat, axis=0)

        cov_01 = cov_01 / obs
        cov_10 = cov_10 / obs
        cov_11 = cov_11 / obs
        cov_00 = cov_00 / obs
        self.cov_0 = 0.5 * (cov_00 + cov_11)
        self.cov_1 = 0.5 * (cov_01 + cov_10)
        assert self.cov_0.shape[0] == zt0_nom.shape[1]
        cov_1_numpy = self.cov_1.numpy()
        cov_0_numpy = self.cov_0.numpy()
        assert cov_1_numpy.dtype == np.float64
        assert cov_0_numpy.dtype == np.float64
        eigvals, eigvecs = scipy.linalg.eigh(cov_1_numpy, b=cov_0_numpy)
        # sorts descending
        idx = np.argsort(np.abs(eigvals))[::-1]
        # remove the slowest process
        self.eigenvectors_ = tf.convert_to_tensor(eigvecs[:, idx], dtype=tf.float32)
        self.eigenvalues_ = tf.convert_to_tensor(eigvals[idx], dtype=tf.float32)
        # transform features trough eigenvectors/basis vectors
        self.means_ = tf.cast(self.means_, tf.float32)
        x_concat = tf.cast(x_concat, tf.float32)
        z = tf.matmul((x_concat - self.means_), self.eigenvectors_)
        self.norms_ = tf.math.sqrt(tf.reduce_mean(z * z, axis=0))

    @tf.function
    def loss_func_vamp(self, shift, back, reversible=False, constr_cov=True):
        """Calculates the VAMP-2 score with respect to the network lobes.

        Based on:
            https://github.com/markovmodel/deeptime/blob/master/vampnet/vampnet/vampnet.py
            https://github.com/hsidky/srv/blob/master/hde/hde.py
        Arguments:
            shift: tensorflow tensor, shifted by tau and truncated in the
                    batch dimension.

            back: tensorlfow tensor, not shifted by tau in the batch dimension and
                    truncated so by size of tau.
        Returns:
            loss_score: tensorflow tensor with shape (1, ).
        """
        N = tf.shape(shift)[0]
        ztt, zt0 = (shift, back)
        zt0 = tf.cast(zt0, tf.float64)
        ztt = tf.cast(ztt, tf.float64)
        N = tf.cast(N, tf.float64)
        # shape (batch_size, output)
        zt0_mean = tf.reduce_mean(zt0, axis=0, keepdims=True)
        ztt_mean = tf.reduce_mean(ztt, axis=0, keepdims=True)
        # Remove the mean from the data
        # shape (batch_size, output)
        zt0 = zt0 - zt0_mean
        ztt = ztt - ztt_mean
        # shape (output, output)
        cov_01 = calc_cov(zt0, ztt)
        cov_10 = calc_cov(ztt, zt0)
        cov_00 = calc_cov(zt0, zt0)
        cov_11 = calc_cov(ztt, ztt)
        cov_0 = 0.5 * (cov_00 + cov_11)
        cov_1 = 0.5 * (cov_01 + cov_10)
        L = tf.linalg.cholesky(cov_0)
        Linv = tf.linalg.inv(L)
        add_loss = 0
        A = tf.matmul(tf.matmul(Linv, cov_1), Linv, transpose_b=True)
        # make sure that all matrices are positive definitive
        lambdas, eig_v = tf.linalg.eigh(A)
        if constr_cov:
            add_loss += (
                -tf.reduce_mean(
                    tf.math.sign(tf.linalg.eigh(cov_11)[0])
                    + tf.math.sign(tf.linalg.eigh(cov_00)[0])
                )
                / 2
            )
        lambdas, eig_v = tf.linalg.eigh(A)
        # train only with respect to positive eigenvalues
        loss = -1 - tf.reduce_sum(tf.math.sign(lambdas) * lambdas**2) + add_loss
        loss = tf.cast(loss, tf.float32)

        return loss

    @tf.function
    def _train_step_vamp(self, shift, back):
        """One train step as the TF 2 graph for autoencoder.
        Arguments:
            shift: Feature matrix which is shifted by tau
            back: Feature matrix which is not shifted by tau
        Returns:
            Tuple with loss and global norm
        """
        with tf.GradientTape() as tape:
            # standarization of obs
            out_shift = self.model(shift, training=True)
            out_back = self.model(back, training=True)
            regul_loss = tf.reduce_sum(self.model.losses)
            loss = self.loss_func_vamp(out_shift, out_back)
            loss += regul_loss
            trainable_var = self.model.trainable_variables
        grads = tape.gradient((loss), trainable_var)
        global_norm = tf.linalg.global_norm(grads)
        self.optimizer.apply_gradients(zip(grads, trainable_var))

        return loss, global_norm
