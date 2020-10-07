import tensorflow as tf
from tensorflow import keras


class ConvResBlock(keras.layers.Layer):
    """A residual block of convolutional layers."""

    def __init__(self,
                 n_inputs=128,
                 num_filters=[8, 16, 32, 64],
                 convlay_config=dict(kernel_size=4, strides=1, padding='SAME',
                                     activation='relu',
                                     kernel_initializer='he_normal'),
                 poollay_config=dict(pool_size=2, strides=2, padding='VALID'),
                 dense_config=dict(activation='relu',
                                   kernel_initializer='he_normal'),
                 output_config=dict(activation=None,
                                    kernel_initializer='he_normal'),
                 **kwargs):
        super().__init__(**kwargs)

        self.n_inputs = n_inputs

        # Construct a list of the convolutional and
        # pooling layers used in this block
        self.conv_layers = [tf.keras.layers.Conv1D(filters=num_filters[0],
                                                   **convlay_config)]
        for filters in num_filters[1:]:
            self.conv_layers.append(
                tf.keras.layers.AveragePooling1D(**poollay_config))
            self.conv_layers.append(tf.keras.layers.Conv1D(filters=filters,
                                                           **convlay_config))

        # Construct a list of the dense layers in this block
        self.dense_layers = [tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(n_inputs,
                                                   **dense_config),
                             tf.keras.layers.Dense(n_inputs,
                                                   **output_config)]

    def call(self, inputs):
        x = tf.expand_dims(inputs, axis=-1)
        for layer in self.conv_layers:
            x = layer(x)
        for layer in self.dense_layers:
            x = layer(x)
        return inputs + x
