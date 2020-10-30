"""Creates the network architecture for linearizing PDEs."""
import numpy as np
import tensorflow as tf
from tensorflow import keras

from DenseResBlock import DenseResBlock
from RelMSE import RelMSE


class NetworkArch(keras.Model):
    """Subclass the Keras Model class."""

    def __init__(self,
                 n_inputs=128,
                 n_latent=21,
                 len_time=51,
                 num_shifts=50,
                 num_shifts_middle=50,
                 outer_encoder=DenseResBlock(),
                 outer_decoder=DenseResBlock(),
                 inner_config=dict(),
                 inner_loss_weights=[1, 1],
                 L_diag=False,
                 train_autoencoder_only=False,
                 **kwargs):
        """
        Create network architecture for linearizing PDEs.

        Arguments:
            n_inputs -- the number of inputs to the network
                (spatial discretization of the PDE)
            n_latent -- the dimensionality of the latent space
                (i.e. number of Koopman eigenfunctions in expansion)
            len_time -- number of time steps for each trajectory in data
            num_shifts -- the number of time steps in the future each network
                input will predict when calculating prediction loss
            num_shifts_middle -- the number of time steps in the future each
                network input will predict when calculating the linearity loss
            outer_encoder -- a Keras Layer or Model with the architecture for
                the outer encoder
            outer_decoder -- a Keras Layer or Model with the architecture for
                the outer decoder (typically the same as the outer encoder)
            inner_config -- Python dictionary with keyword arguments for
                the inner encoder and decoder
            inner_loss_weights -- Python list of length 2 with weights for the
                inner autoencoder loss and linearity loss
            L_diag -- Boolean that determines whether the dynamics matrix L is
                constrained to be diagonal
            train_autoencoder_only -- Boolean that determines whether only the
                autoencoder losses are used for training. It is recommendeded
                that you do several epochs of pretraining with only the
                autoencoder losses and then set this option to False to include
                the prediction and linearity losses.
            **kwargs -- additional keyword arguments. Can be used to name the
                Model.
        """
        super().__init__(**kwargs)

        self.n_inputs = n_inputs
        self.n_latent = n_latent
        self.len_time = len_time
        self.num_shifts = num_shifts
        self.num_shifts_middle = num_shifts_middle
        self.outer_encoder = outer_encoder
        self.outer_decoder = outer_decoder

        # Create the inner encoder layer
        self.inner_encoder = keras.layers.Dense(
            n_latent,
            name='inner_encoder',
            activation=None,
            use_bias=False,
            kernel_initializer=identity_init,
            **inner_config)

        # The dynamics matrix, initialized as identity
        self.L = tf.Variable(tf.eye(n_latent), trainable=True)

        # Create the inner decoder layer
        self.inner_decoder = keras.layers.Dense(
            n_inputs,
            name='inner_decoder',
            activation=None,
            use_bias=False,
            kernel_initializer=identity_init,
            **inner_config)

        self.RelMSE = RelMSE(name='RelMSE')
        self.inner_loss_weights = inner_loss_weights
        self.L_diag = L_diag
        self.train_autoencoder_only = train_autoencoder_only

    def call(self, inputs):
        """
        Run given inputs through the newtork.

        Arguments:
            inputs:
            inputs -- A Numpy array or Tensorflow tensor with shape
                (number_of_trajectories, len_time, n_inputs)
            outputs:
            autoencoder_output -- The output of running each time
                step of each trajectory through the autoencoder
            outer_auto_output -- The output of running each time step
                of each trajectory through the outer autoencoder
            predictions -- The predictions for num_steps steps into the
                future for all trajectories and all time steps for which
                num_steps steps into the future are in the data

        """
        # For the shape of the outputs for prediction and linearity loss
        len_pred = self.len_time - self.num_shifts
        len_lin = self.len_time - self.num_shifts_middle

        # Create arrays for prediction and linearity
        pred_inputs = inputs[:, :len_pred, :]
        lin_inputs = inputs[:, :len_lin, :]

        # Inputs for "exact" solutions for prediction and linearity loss
        pred_exact = stack_predictions(inputs, self.num_shifts)
        lin_advanced = stack_predictions(inputs, self.num_shifts_middle)

        #  Reshape inputs as 2D arrays
        auto_inputs, pred_inputs, lin_inputs, lin_advanced = reshape_inputs(
            (inputs, pred_inputs, lin_inputs, lin_advanced))

        # Autoencoder
        partially_encoded = self.outer_encoder(auto_inputs)
        fully_encoded = self.inner_encoder(partially_encoded)
        partially_decoded = self.inner_decoder(fully_encoded)
        autoencoder_output = self.outer_decoder(partially_decoded)

        autoencoder_output = tf.reshape(autoencoder_output,
                                        [-1, self.len_time, self.n_inputs])

        # Outer Autoencoder
        outer_auto_output = self.outer_decoder(partially_encoded)

        outer_auto_output = tf.reshape(outer_auto_output,
                                       [-1, self.len_time, self.n_inputs])

        # Inner Autoencoder Loss
        self.add_loss(self.inner_loss_weights[0]
                      * self.RelMSE(partially_encoded, partially_decoded))

        # If training autoencoder only, output results
        if self.train_autoencoder_only:
            predictions = 0 * pred_exact
            return autoencoder_output, outer_auto_output, predictions

        # Set dynamics matrix L
        if self.L_diag:
            Lmat = tf.linalg.diag(tf.linalg.diag_part(self.L))
        else:
            Lmat = self.L

        # Prediction
        predictions_list = []
        part_encoded_pred = self.outer_encoder(pred_inputs)
        current_encoded = self.inner_encoder(part_encoded_pred)
        for shift in range(self.num_shifts):
            advanced_encoded = tf.matmul(current_encoded, Lmat)
            adv_part_decoded = self.inner_decoder(advanced_encoded)
            advanced_decoded = self.outer_decoder(adv_part_decoded)
            predictions_list.append(tf.reshape(advanced_decoded,
                                               [-1, len_pred, self.n_inputs]))
            current_encoded = tf.identity(advanced_encoded)
        predictions = tf.concat(predictions_list, axis=1)

        # Linearity predictions
        linearity_list = []
        part_encoded_lin = self.outer_encoder(lin_inputs)
        current_encoded = self.inner_encoder(part_encoded_lin)
        for shift in range(self.num_shifts_middle):
            advanced_encoded = tf.matmul(current_encoded, Lmat)
            current_encoded = tf.identity(advanced_encoded)
            linearity_list.append(tf.reshape(current_encoded,
                                             [-1, len_lin, self.n_latent]))
        lin_pred = tf.concat(linearity_list, axis=1)

        lin_part_encoded = self.outer_encoder(lin_advanced)
        lin_exact = self.inner_encoder(lin_part_encoded)
        lin_exact = tf.reshape(
            lin_exact,
            [-1, self.num_shifts_middle * len_lin, self.n_latent])

        # Add Linearity loss
        self.add_loss(self.inner_loss_weights[1]
                      * self.RelMSE(lin_exact, lin_pred))

        return autoencoder_output, outer_auto_output, predictions


def identity_init(shape, dtype=tf.float32):
    """Initialize weight matrices as identity-like matrices."""
    n_rows = shape[0]
    n_cols = shape[1]
    if n_rows >= n_cols:
        A = np.zeros((n_rows, n_cols), dtype=np.float32)
        for col in range(n_cols):
            for row in range(col, col + n_rows - n_cols + 1):
                A[row, col] = 1.0 / (n_rows - n_cols + 1)
    else:
        A = np.zeros((n_rows, n_cols), dtype=np.float32)
        for row in range(n_rows):
            for col in range(row, row + n_cols - n_rows + 1):
                A[row, col] = 1.0 / (n_cols - n_rows + 1)
    return A


def reshape_inputs(inputs):
    """Reshape inputs to be 2D arrays."""
    input_list = []
    for data in inputs:
        input_list.append(tf.reshape(data, [-1, data.shape[-1]]))
    return tuple(input_list)


def stack_predictions(data, num_shifts):
    """Stack inputs for linearity or prediction loss."""
    len_pred = data.shape[1] - num_shifts
    prediction_list = []
    for j in range(num_shifts):
        prediction_list.append(data[:, j + 1:j + 1 + len_pred, :])
    prediction_tensor = tf.concat(prediction_list, axis=1)

    return prediction_tensor
