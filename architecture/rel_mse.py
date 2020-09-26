#from tensorflow.keras.losses import LossFunctionWrapper
from tensorflow import keras
import tensorflow as tf


class rel_mse(keras.losses.Loss):
    """Computes normalized mean of squares of errors between labels and preds.
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
        # Compute the MSE and the L2 norm of the true values (with 1/batch_size prefactor)
        mse = tf.reduce_mean(tf.square(y_pred-y_true), axis=-1)
        true_norm = tf.reduce_mean(tf.square(y_true), axis=-1)
        # Ensure there are no 'zero' values in the denominator before division
        true_norm += self.denom_nonzero

        # Compute normalized MSE (normalized to true L2 norm)
        err = tf.truediv(mse, true_norm)
        err = tf.reshape(err, [-1])
        #err = tf.reduce_mean(err, axis=-1)

        # Return the error
        return err

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "denom_nonzero": self.denom_nonzero}