"""Calculate relative mean squared error."""
from tensorflow import keras
import tensorflow as tf


class RelMSE(keras.losses.Loss):
    """Computes relative mean squared error between labels and preds.

    # Arguments
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) name for the loss.
        norm_ord: (Optional) The 'ord' parameter for backend norm method
        norm_opts: (Optional) Additional parameters for norm method
    Normalization is based on the 'true' values
    """

    def __init__(self, denom_nonzero=1e-5, **kwargs):
        self.denom_nonzero = denom_nonzero
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):

        # Compute the MSE and the L2 norm of the true values
        mse = tf.reduce_mean(tf.square(y_pred - y_true), axis=-1)
        true_norm = tf.reduce_mean(tf.square(y_true), axis=-1)
        # Ensure there are no 'zero' values in the denominator before division
        true_norm += self.denom_nonzero

        # Compute relative MSE
        err = tf.truediv(mse, true_norm)
        err = tf.reduce_mean(err, axis=-1)

        # Return the error
        return err
