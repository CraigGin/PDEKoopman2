import tensorflow as tf
from tensorflow import keras

class DenseResBlock(keras.layers.Layer):
    """A residual block of dense layers."""
    def __init__(self, 
                 n_inputs=128,
                 num_hidden=4,
                 hidden_config=dict(activation='relu',
                                    kernel_initializer='he_normal'),
                 output_config=dict(activation=None, 
                                    kernel_initializer='he_normal'),
                 **kwargs):
        super().__init__(**kwargs)
        
        self.n_inputs = n_inputs
        self.num_hidden = num_hidden
        
        # Construct a list of the hidden layers used in this block
        self.layers = [keras.layers.Dense(n_inputs, 
                                         name='hidden{}'.format(i), 
                                         **hidden_config) 
                       for i in range(num_hidden)]
        # Append the output layer
        self.layers.append(keras.layers.Dense(n_inputs,
                                              name='output', 
                                              **output_config))

    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return inputs + x

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                'n_inputs': self.n_inputs,
                'num_hidden': self.num_hidden,
                'layers': self.layers}

