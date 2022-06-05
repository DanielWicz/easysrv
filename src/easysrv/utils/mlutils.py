import tensorflow as tf
import tensorflow_probability as tfp


def rao_blackwell_ledoit_wolf(cov, N):
    """Rao-Blackwellized Ledoit-Wolf shrinkaged estimator of the covariance
    matrix.
    Arguments:
        cov: array, shape=(n, n), Sample covariance matrix (e.g. estimated with np.cov(X.T))
        N : int, Number of data points.
    Returns:
        Corrected covariance matrix
    [1] Chen, Yilun, Ami Wiesel, and Alfred O. Hero III. "Shrinkage
    estimation of high dimensional covariance matrices" ICASSP (2009)
    Based on: https://github.com/msmbuilder/msmbuilder/blob/master/msmbuilder/decomposition/tica.py
    """
    p = len(cov)
    assert cov.shape == (p, p)

    alpha = (N - 2) / (N * (N + 2))
    beta = ((p + 1) * N - 2) / (N * (N + 2))

    trace_cov2 = tf.reduce_sum(cov * cov)
    U = (p * trace_cov2 / tf.linalg.trace(cov) ** 2) - 1
    rho = tf.minimum(alpha + beta / U, 1)

    F = (tf.linalg.trace(cov) / p) * tf.eye(p, dtype=tf.float64)
    return (1 - rho) * cov + rho * F, rho


def calc_cov(x, y, rblw=False, use_shrinkage=False, no_normalize=False):
    """Calculates covariance matrix from a batch of tensorflow data.
    x: first set of variables as an tensorflow array
    y: second set of variables as an tensorflow array
    rblw: if to use shrinkage correction rblw
    use_shrinkage: if to use shrinkage at all
    no_normalize: do not normalize by the number of samples
    """
    x = tf.cast(x, tf.float64)
    y = tf.cast(y, tf.float64)
    N = tf.cast(tf.shape(x)[0], tf.float64)
    feat = tf.cast(tf.shape(x)[1], tf.float64)
    if not no_normalize:
        cov = 1 / N * tf.matmul(x, y, transpose_a=True)
    else:
        cov = tf.matmul(x, y, transpose_a=True)
    if rblw and use_shrinkage:
        cov_shrink, shrinkage = rao_blackwell_ledoit_wolf(cov, N)
        return cov_shrink
    else:
        return cov


def metric_VAMP2(shift, back):
    """Returns the sum of the squared top k eigenvalues of the vamp matrix,
    with k determined by the wrapper parameter k_eig, and the vamp matrix
    defined as:
        V = cov_00 ^ -1/2 * cov_01 * cov_11 ^ -1/2
    Can be used as a metric function in model.fit()

    Arguments:
        shift: shifted data
        back: not shifted data
    Returns:
        sum of the squared k highest eigenvalues in the vamp matrix (VAMP2 score)
    """
    N = tf.shape(shift)[0]
    zt0 = back
    ztt = shift

    # shape (batch_size, output)
    zt0 = zt0 - tf.reduce_mean(zt0, axis=0, keepdims=True)
    # shape (batch_size, output)
    ztt = ztt - tf.reduce_mean(ztt, axis=0, keepdims=True)
    # Calculate the covariance matrices
    # shape (output, output)
    cov_01 = calc_cov(zt0, ztt, rblw=False, use_shrinkage=False)
    cov_10 = calc_cov(ztt, zt0, rblw=False, use_shrinkage=False)
    cov_00 = calc_cov(zt0, zt0)
    cov_11 = calc_cov(ztt, ztt)
    cov_00_inv = tf.linalg.sqrtm(tf.linalg.inv(cov_00))
    cov_11_inv = tf.linalg.sqrtm(tf.linalg.inv(cov_11))
    vamp_matrix = tf.matmul(tf.matmul(cov_00_inv, cov_01), cov_11_inv)
    # Select the all singular values of the VAMP matrix
    diag = tf.linalg.svd(vamp_matrix, compute_uv=False)
    eig_sum_sq = 1 + tf.reduce_sum(diag**2)

    return eig_sum_sq


class AutoClipper:
    """
    Source: https://github.com/pseeth/autoclip/blob/master/autoclip_tf.py
    """

    def __init__(self, clip_percentile, history_size=10000):
        self.clip_percentile = clip_percentile
        self.grad_history = tf.Variable(tf.zeros(history_size), trainable=False)
        self.i = tf.Variable(0, trainable=False)
        self.history_size = history_size

    def __call__(self, grads_and_vars):
        grad_norms = [self._get_grad_norm(g) for g, _ in grads_and_vars]
        total_norm = tf.norm(grad_norms)
        assign_idx = tf.math.mod(self.i, self.history_size)
        self.grad_history = self.grad_history[assign_idx].assign(total_norm)
        self.i = self.i.assign_add(1)
        clip_value = tfp.stats.percentile(
            self.grad_history[: self.i], q=self.clip_percentile
        )
        return [(tf.clip_by_norm(g, clip_value), v) for g, v in grads_and_vars]

    def _get_grad_norm(self, t, axes=None, name=None):
        values = tf.convert_to_tensor(
            t.values if isinstance(t, tf.IndexedSlices) else t, name="t"
        )

        # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
        l2sum = tf.math.reduce_sum(values * values, axes, keepdims=True)
        pred = l2sum > 0
        # Two-tap tf.where trick to bypass NaN gradients
        l2sum_safe = tf.where(pred, l2sum, tf.ones_like(l2sum))
        return tf.squeeze(tf.where(pred, tf.math.sqrt(l2sum_safe), l2sum))
